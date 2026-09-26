import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "github_code"


embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True}
)


vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    thinking_budget=0
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("""
You are a GitHub repository assistant.

Answer the question using only the provided repository context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the repository."

Mention relevant file names when useful.

Context:
{context}

Question:
{question}
""")

format_docs = RunnableLambda(
    lambda documents: "\n\n".join(
        f"FILE: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in documents
    )
)

parallel_chain = {
    "question": RunnablePassthrough(),
    "context": retriever | format_docs
}

main_chain = prompt | model | parser
chain = parallel_chain | main_chain

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAnswer:")
    print(answer)