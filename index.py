import ast
from typing import Tuple
from langchain_cohere.embeddings import CohereEmbeddings
import chromadb
from dotenv import load_dotenv
import numpy as np
import uuid
load_dotenv()


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="latest")

def getEmbeddingModel():
    model = CohereEmbeddings(model = "embed-v4.0")
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


with open("test.py") as f:
    reader = f.read()

code = reader


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

    # classes_list = [x.name for x in tree.body if isinstance(x,ast.ClassDef)] 

    return (class_methods_names , classes_method_contents)

# print(get_all_functions(tree)[0])


get_all_functions(tree)
get_all_classes(tree)

query_embeddings = model.embed_query("setup database")
result = collection.query(query_embeddings=[query_embeddings],n_results = 2,include = ["metadatas","distances"])

print(result)
# if isinstance(item, ast.FunctionDef):
#     chunks.append({
#         'type': 'method',
#         'name': f"{class_name}.{item.name}",
#         'class': class_name,
#         'file': filepath,
#         'line': item.lineno,
#         'docstring': ast.get_docstring(item) or '',
#         'code': ast.get_source_segment(code, item)
#     })