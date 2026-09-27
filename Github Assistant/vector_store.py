import os
import sys
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from text_splitter import split_documents
from document_loader import load_documents


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_DB_DIR = "vector_db"


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs={
            "device": "mps",
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )


def create_vector_store(chunks):
    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name="github_code"
    )

    return vector_store


def build_vector_store(url):
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
        print("Old vector database deleted.")

    documents = load_documents(url)

    print(f"Documents loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    vector_store = create_vector_store(chunks)

    print(f"Vectors stored: {len(chunks)}")
    print(f"Vector database: {os.path.abspath(VECTOR_DB_DIR)}")

    return vector_store


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vector_store.py <github_repo_url>")
        sys.exit(1)

    repo_url = sys.argv[1]

    build_vector_store(repo_url)