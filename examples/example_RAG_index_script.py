"""
Start indexing your local codebase into retrieavable vector databases with chromadb.

Requirements:
    chromadb (`pip install chromadb`)
    langchain_text_splitters (`pip install langchain_text`)
    A local embedder model. By default, Chroma will pull an omnx runtime `all-miniLM-L6-v2` instance that they built to remove dependencies.
    
Usage:
    Replace `model_path = "/path/to/your/local/model_folder"` if you have a local model available. IMPORTANT: You must also use this model during runtime retrieval.
    At example/: 
        python example_RAG_index_script.py [PATH_TO_ROOT_OF_YOUR_DIRECTORY] [PACKAGE_NAME] [LOCAL_CHROMA_DB_PATH]
"""
import sys
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_path = None  # Replace with your local model path if you have one
local_ef = None
if model_path:
    from chromadb.utils import embedding_functions
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_path
    )

# Code-Aware Splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", 
    chunk_size=800, 
    chunk_overlap=100
)

def ingest_code(repo_path, package_name, chroma_path):
    documents = []
    metadatas = []
    ids = []

    count = 0
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code = f.read()
                    chunks = splitter.split_text(code)

                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({"source": rel_path, "chunk": i})
                        ids.append(f"{rel_path}_{i}")
                        count += 1

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=package_name, embedding_function=local_ef)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Successfully indexed {count} chunks from {repo_path}")

if __name__ == "__main__":
    from pathlib import Path
    path = sys.argv[1] if len(sys.argv) > 1 else "."

    path_P = Path(path)
    if not path_P.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    
    path = str(path_P.resolve())
    
    package_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    chroma_path = sys.argv[3] if len(sys.argv) > 3 else str((path_P / "chromadb_data").resolve())

    ingest_code(path, package_name, chroma_path)
