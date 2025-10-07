import ast
from typing import Tuple
from langchain_cohere.embeddings import CohereEmbeddings
import chromadb
from dotenv import load_dotenv
import numpy as np
import uuid
import os
import subprocess
import shutil
import time
load_dotenv()


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="latest")

def getEmbeddingModel():
    model = CohereEmbeddings(model = "embed-v4.0",max_retries=3)
    return model

model = getEmbeddingModel()

code = """
def login_user(email, password):
    return True

def logout_user(session_id):
    return False

class Homo:
    def __init__(self):
        pass
    def isHomo(self):
        pass
    def isntHomo(self):
        pass
"""

def read_file(path):
    with open(path) as f:
        reader = f.read()
    return reader



tree = ast.parse(code)

def get_function_line_end(x):
    """This function returns the end line number of functions in tree"""
    return x.end_lineno

def get_function_line_start(x):
    """This function returns the line number of functions in tree"""
    return x.lineno
def get_classes_line_start(x):
    """This function returns the line number of functions in tree"""
    return x.lineno

def get_all_functions(tree , file_name = "test.py")->Tuple:
    """This function returns the line number of functions in tree with their respective line starts"""
    
    # Fetch the functions from the ast
    functions_list = [x for x in tree.body if isinstance(x,ast.FunctionDef)] 

    functions_name_list = [x.name for x in tree.body if isinstance(x,ast.FunctionDef)] 
    
    function_contents = [ast.get_source_segment(code ,x) for x in functions_list]

    # Fetch function line starts and ends from the ast 
    functions_lines_index = [get_function_line_start(x) for x in functions_list] 

    # Fetch the contents of each function
    for function in range(len(functions_name_list)):
        chunk = {
            "type":"function",
            "name":functions_name_list[function],
            "file":file_name,
            "line":functions_lines_index[function],
            "docstring":ast.get_docstring(functions_list[function]) or " ",
            "code":function_contents[function] or " "
        }

        print("Function chunk")
        print(chunk)

        myuuid = uuid.uuid4()
        if function_contents[function]:
            embedding = model.embed_documents([f"{chunk['name']} {chunk['docstring']} {chunk['code']}"])[0]
            collection.add(
                ids = [f"{myuuid}"],
                embeddings = embedding,
                metadatas = [chunk]
            )
            time.sleep(5)

    return (functions_name_list , function_contents)

def get_all_classes(tree , file_name = "test.py")->Tuple:
    classes_list = [x for x in tree.body if isinstance(x,ast.ClassDef)] 
    print(classes_list)


    class_methods = [[y for y in x.body if isinstance(y,ast.FunctionDef)] for x in classes_list]
    

    class_methods_lines = [[y.lineno for y in x.body if isinstance(y,ast.FunctionDef)] for x in classes_list] 

    classes_method_contents = [[ast.get_source_segment(code ,y) for y in x] for x in class_methods]

    class_methods_names = [[str(x.name + "." + y.name) for y in x.body if isinstance(y,ast.FunctionDef)] for x in classes_list]

    for i in range(len(classes_method_contents)):
        if len(classes_method_contents[i]) == 0:
            continue
        else:
            for j,content in enumerate(classes_method_contents[i]): 
                if(content):
                    myuuid = uuid.uuid4()

                    chunk = {
                        "type":"method",
                        "name":class_methods_names[i][j],
                        "class":classes_list[i].name,
                        "file":file_name,
                        "line":class_methods_lines[i][j],
                        "docstring":ast.get_docstring(class_methods[i][j]) or " ",
                        "code":content or " "
                    }

                    embedding = model.embed_documents([f"{chunk['name']} {chunk['docstring']} {chunk['code']}"])[0]
                    print(chunk)

                    collection.add(
                        ids = [f"{myuuid}"],
                        embeddings = embedding,
                        metadatas = [chunk]
                    )
                    time.sleep(5)


    # classes_list = [x.name for x in tree.body if isinstance(x,ast.ClassDef)] 

    return (class_methods_names , classes_method_contents)



def get_all_imports(tree,file_name = "test.py"):
    
    all_imports  = [x for x in tree.body if isinstance(x,ast.Import) or isinstance(x,ast.ImportFrom)]
    # all_imports_name = [x.name for x in all_imports]
    all_imports_name = [ast.get_source_segment(code , x) for x in all_imports]
    all_imports_lines = [x.lineno for x in all_imports]
    
    embedding_document_array = []
    chunk_array = []

    for i,chunks in enumerate(all_imports_name):
        if chunks and file_name and all_imports_lines[i]:
            
            chunk = {
                "type":"imports",
                "name":chunks.split(' ')[1],
                "file":file_name,
                "line":all_imports_lines[i],
                "code": chunks
            }

            chunk_array.append(chunk)

            embedding_document_string = f"{chunk['name']} {chunk['type']} {chunk['code']}"

            embedding_document_array.append(embedding_document_string)

        embeddings = model.embed_documents(embedding_document_array)
        for i in range(len(chunk_array)):
            myuuid = uuid.uuid4()
            collection.add(
                ids = [f"{myuuid}"],
                embeddings = embeddings[i],
                metadatas = [chunk_array[i]]
            )

    return (all_imports , all_imports_name)






# for x in ast.walk(tree):
    
#     print(x)
repo = input("Enter the name of the repository: ")

# Reset temp_clone
if os.path.exists('temp_clone'):
    shutil.rmtree('temp_clone')
os.mkdir("temp_clone")

# Clone repo
os.chdir("temp_clone")
subprocess.run(["git", "clone", repo])

# Get repo directory name (git clones into a folder named after the repo)
repo_name = os.listdir(".")[0]
os.chdir(repo_name)

# Walk and list files
ignore_dirs = {'.git', '__pycache__'}

for root, dirs, files in os.walk(os.getcwd()):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for file_name in files:
        print(os.path.join(root, file_name))
        code = read_file(os.path.join(root, file_name))
        tree = ast.parse(code)
        get_all_classes(tree,file_name)
        get_all_functions(tree,file_name)
        get_all_imports(tree,file_name)

query_embedd = model.embed_query("Which function contains imports logic")

results = collection.query(
    query_embeddings=query_embedd,
    n_results=5
)

print(results)