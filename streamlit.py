import streamlit as st
import subprocess
import os
import shutil
import tempfile
import time
import ast
import uuid
from typing import Tuple
from dotenv import load_dotenv

# Optional: Cohere + chromadb (same libs as your snippet)
try:
    from langchain_cohere.embeddings import CohereEmbeddings
    import chromadb
except Exception:
    # We'll surface dependency errors in the UI instead of failing import-time
    CohereEmbeddings = None
    chromadb = None

load_dotenv()

st.set_page_config(page_title="Repo Function / Class Explorer", layout="wide")
st.title("🔎 Repo Function & Class Explorer — Streamlit edition")
st.markdown("Upload or paste a GitHub repo URL and the app will clone, parse Python files using `ast`, embed function/method/import chunks, and let you query them.")

# --- Helpers (based on your original logic, adapted) ---

def get_embedding_model():
    if CohereEmbeddings is None:
        raise RuntimeError("Cohere / langchain_cohere is not installed.")
    model = CohereEmbeddings(model="embed-v4.0", max_retries=3)
    return model


def safe_create_chroma_collection(name: str = "latest"):
    if chromadb is None:
        raise RuntimeError("chromadb is not installed.")
    client = chromadb.Client()
    try:
        collection = client.get_collection(name)
    except Exception:
        # fallback to create
        try:
            collection = client.create_collection(name=name)
        except Exception:
            # if create also fails, raise error
            raise
    return client, collection


# small utility to read file contents
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# AST helpers (kept similar to original but simplified)

def get_all_functions(code: str, tree: ast.AST, file_name: str = "<in-memory>") -> Tuple:
    functions_list = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    functions_name_list = [x.name for x in functions_list]
    function_contents = [ast.get_source_segment(code, x) or "" for x in functions_list]
    functions_lines_index = [x.lineno for x in functions_list]

    chunks = []
    docs = []
    for i, fn in enumerate(functions_list):
        chunk = {
            "type": "function",
            "name": functions_name_list[i],
            "file": file_name,
            "line": functions_lines_index[i],
            "docstring": ast.get_docstring(fn) or "",
            "code": function_contents[i] or "",
        }
        chunks.append(chunk)
        docs.append(f"{chunk['name']} {chunk['docstring']} {chunk['code']}")
    return chunks, docs


def get_all_classes(code: str, tree: ast.AST, file_name: str = "<in-memory>") -> Tuple:
    classes_list = [x for x in tree.body if isinstance(x, ast.ClassDef)]
    chunk_list = []
    docs = []

    for cls in classes_list:
        methods = [y for y in cls.body if isinstance(y, ast.FunctionDef)]
        for m in methods:
            name = f"{cls.name}.{m.name}"
            content = ast.get_source_segment(code, m) or ""
            chunk = {
                "type": "method",
                "name": name,
                "class": cls.name,
                "file": file_name,
                "line": m.lineno,
                "docstring": ast.get_docstring(m) or "",
                "code": content,
            }
            chunk_list.append(chunk)
            docs.append(f"{chunk['name']} {chunk['docstring']} {chunk['code']}")
    return chunk_list, docs


def get_all_imports(code: str, tree: ast.AST, file_name: str = "<in-memory>") -> Tuple:
    imports = [x for x in tree.body if isinstance(x, ast.Import) or isinstance(x, ast.ImportFrom)]
    chunk_list = []
    docs = []
    for im in imports:
        src = ast.get_source_segment(code, im) or ""
        # try to create a readable name
        name = src.split()[1] if len(src.split()) > 1 else src
        chunk = {
            "type": "imports",
            "name": name,
            "file": file_name,
            "line": im.lineno,
            "code": src,
        }
        chunk_list.append(chunk)
        docs.append(f"{chunk['name']} {chunk['type']} {chunk['code']}")
    return chunk_list, docs


# Core parsing + embedding pipeline

def parse_and_index_repo(repo_path: str, model, collection, sleep_between_files: float = 0.0):
    indexed_count = 0
    for root, dirs, files in os.walk(repo_path):
        # skip common trash
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'saved_model'}]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            full = os.path.join(root, fname)
            try:
                code = read_file(full)
                tree = ast.parse(code)

                fn_chunks, fn_docs = get_all_functions(code, tree, file_name=os.path.relpath(full, repo_path))
                cls_chunks, cls_docs = get_all_classes(code, tree, file_name=os.path.relpath(full, repo_path))
                im_chunks, im_docs = get_all_imports(code, tree, file_name=os.path.relpath(full, repo_path))

                all_chunks = fn_chunks + cls_chunks + im_chunks
                all_docs = fn_docs + cls_docs + im_docs

                if len(all_docs) == 0:
                    continue

                # embed documents (batch)
                embeddings = model.embed_documents(all_docs)

                for i, chunk in enumerate(all_chunks):
                    myid = str(uuid.uuid4())
                    # chroma expects lists for ids/embeddings/metadatas
                    collection.add(ids=[myid], embeddings=[embeddings[i]], metadatas=[chunk])
                    indexed_count += 1

                if sleep_between_files:
                    time.sleep(sleep_between_files)

            except Exception as e:
                # keep going — surface the error in the UI
                st.error(f"Error processing {full}: {e}")
    return indexed_count


# --- Streamlit UI layout ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Controls")
    repo_url = st.text_input("GitHub repo URL (https or ssh)")
    uploaded = st.file_uploader("Or upload a .zip of a repo (optional)", type=["zip"])

    create_index_button = st.button("Clone & Index Repository")
    clear_index_button = st.button("Clear collection (danger)")

    st.markdown("---")
    st.subheader("Query the collection")
    user_query = st.text_input("Enter a natural-language query (eg: 'Which function contains train pipeline logic')")
    n_results = st.number_input("Number of results", min_value=1, max_value=10, value=3)
    query_button = st.button("Search")

    st.markdown("---")
    st.caption("Notes: Private repos require credentials available to the running environment. Cohere API key must be set in env for embedding. Chroma client must be available.")

with col2:
    st.header("Output")
    log_area = st.empty()
    result_area = st.empty()


# Create chroma collection and embedding model lazily when needed
client = None
collection = None
model = None

if create_index_button:
    with st.spinner("Preparing to index repository..."):
        try:
            if chromadb is None or CohereEmbeddings is None:
                raise RuntimeError("Required libraries missing. Please install `chromadb` and `langchain_cohere`.")

            model = get_embedding_model()
            client, collection = safe_create_chroma_collection(name="latest")

            # create a temporary directory to clone into
            tmpdir = tempfile.mkdtemp(prefix="st_repo_")
            cloned_path = None

            if uploaded is not None:
                # unzip into tempdir
                import zipfile
                zpath = os.path.join(tmpdir, uploaded.name)
                with open(zpath, "wb") as f:
                    f.write(uploaded.getvalue())
                with zipfile.ZipFile(zpath, 'r') as zf:
                    zf.extractall(tmpdir)
                cloned_path = tmpdir
            elif repo_url:
                # use git clone
                try:
                    # clone shallow to save time
                    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, tmpdir], stderr=subprocess.STDOUT)
                    cloned_path = tmpdir
                except subprocess.CalledProcessError as e:
                    st.error(f"git clone failed: {e}")
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    cloned_path = None
            else:
                st.error("Provide a GitHub url or upload a zip file.")

            if cloned_path:
                # try to find the repo root (if zip had a top-level folder)
                entries = os.listdir(cloned_path)
                if len(entries) == 1 and os.path.isdir(os.path.join(cloned_path, entries[0])):
                    repo_root = os.path.join(cloned_path, entries[0])
                else:
                    repo_root = cloned_path

                st.success(f"Repository ready at {repo_root}")
                pbar = st.progress(0)
                start = time.time()
                count = parse_and_index_repo(repo_root, model, collection, sleep_between_files=0.0)
                elapsed = time.time() - start
                pbar.progress(100)
                st.success(f"Indexed {count} chunks in {elapsed:.1f}s")

        except Exception as e:
            st.exception(e)


if clear_index_button:
    try:
        if chromadb is None:
            raise RuntimeError("chromadb not installed")
        client = chromadb.Client()
        try:
            cur = client.get_collection("latest")
            cur.delete()  # try delete all items
            st.warning("Collection (latest) cleared.")
        except Exception:
            # recreate collection to ensure clean state
            client.create_collection(name="latest")
            st.warning("Collection reset.")
    except Exception as e:
        st.error(f"Error clearing collection: {e}")


if query_button:
    try:
        if model is None:
            model = get_embedding_model()
        if client is None or collection is None:
            client, collection = safe_create_chroma_collection(name="latest")

        q_emb = model.embed_query(user_query)
        results = collection.query(query_embeddings=[q_emb], n_results=n_results)

        # results is a dict-like structure — display neatly
        result_area.subheader("Search results")
        # We will render result metadata and distance scores (if present)
        rows = []
        # chroma returns keys: ids, distances, metadatas, documents (varies by backend)
        ids = results.get('ids', [])
        distances = results.get('distances', [])
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])

        import pandas as pd
        table_rows = []
        for i in range(len(ids)):
            meta = metadatas[0][i] if metadatas and len(metadatas) and len(metadatas[0]) > i else {}
            doc = documents[0][i] if documents and len(documents) and len(documents[0]) > i else ""
            dist = distances[0][i] if distances and len(distances) and len(distances[0]) > i else None
            table_rows.append({
                "id": ids[0][i] if ids and len(ids) and len(ids[0]) > i else "",
                "score": float(dist) if dist is not None else None,
                "file": meta.get('file', ''),
                "line": meta.get('line', ''),
                "type": meta.get('type', ''),
                "name": meta.get('name', ''),
                "code": meta.get('code', '')[:800],
            })

        df = pd.DataFrame(table_rows)
        # Render results as readable code snippets with metadata instead of a raw table
        if len(table_rows) == 0:
            st.info("No results found.")
        else:
            for r in table_rows:
                header = f"{r.get('type','')} — {r.get('name','')}  ({r.get('file','')}:{r.get('line','')})"
                with st.expander(header, expanded=False):
                    if r.get('score') is not None:
                        st.write(f"**Score:** {r['score']}")
                    st.write(f"**File:** {r.get('file','')}")
                    st.write(f"**Line:** {r.get('line','')}")
                    st.write(f"**Type:** {r.get('type','')}")
                    # Show full code (no truncation) with Python syntax highlighting
                    st.code(r.get('code',''), language='python')

    except Exception as e:
        st.exception(e)


st.markdown("---")
st.caption("Built by a helpful script. Be careful when indexing private repositories — credentials must be present in the runtime environment.")
