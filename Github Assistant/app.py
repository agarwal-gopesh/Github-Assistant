import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from vector_store import build_vector_store, VECTOR_DB_DIR


load_dotenv()

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "github_code"


st.set_page_config(
    page_title="GitHub Assistant",
    page_icon="⌘",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #888;
            margin-bottom: 2rem;
        }

        .repo-status {
            padding: 0.7rem 0.9rem;
            border-radius: 0.6rem;
            background: rgba(128, 128, 128, 0.1);
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_rag_chain():
    embeddings = create_embeddings()

    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.7,
        thinking_budget=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a GitHub repository assistant.

Answer the question using only the provided repository context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the repository."

Mention relevant file names when useful.

Context:
{context}

Question:
{question}
"""
    )

    format_docs = RunnableLambda(
        lambda documents: "\n\n".join(
            f"FILE: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in documents
        )
    )

    parallel_chain = {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs,
    }

    main_chain = prompt | model | StrOutputParser()

    return parallel_chain | main_chain


if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "repository_url" not in st.session_state:
    st.session_state.repository_url = None


# Sidebar
with st.sidebar:
    st.markdown("## GitHub Assistant")
    st.caption("Chat with any GitHub repository")

    st.divider()

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repository",
    )

    load_button = st.button(
        "Load Repository",
        use_container_width=True,
        type="primary",
    )

    if load_button:
        if not repo_url.strip():
            st.error("Please enter a GitHub repository URL.")
        elif "github.com/" not in repo_url:
            st.error("Please enter a valid GitHub repository URL.")
        else:
            try:
                with st.status("Loading repository...", expanded=True) as status:
                    st.write("Cloning and filtering files...")
                    build_vector_store(repo_url.strip())

                    st.write("Loading the RAG pipeline...")
                    st.session_state.rag_chain = create_rag_chain()

                    st.session_state.repository_url = repo_url.strip()
                    st.session_state.messages = []

                    status.update(
                        label="Repository ready",
                        state="complete",
                        expanded=False,
                    )

                st.success("Repository loaded successfully.")

            except Exception as error:
                st.session_state.rag_chain = None
                st.session_state.repository_url = None
                st.error(f"Could not load the repository: {error}")

    if st.session_state.repository_url:
        st.divider()
        st.markdown("**Current repository**")
        st.caption(st.session_state.repository_url)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# Main area
st.markdown('<div class="app-title">GitHub Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions about your GitHub repository</div>',
    unsafe_allow_html=True,
)


if st.session_state.repository_url is None:
    st.info("Load a GitHub repository from the sidebar to start chatting.")

    st.markdown("### Try asking")

    example_questions = [
        "How does authentication work?",
        "What technologies are used in this project?",
        "Explain the main project structure.",
        "Where is the database connection handled?",
    ]

    for question in example_questions:
        st.markdown(f"- {question}")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about the repository..."
)

if question:
    if st.session_state.rag_chain is None:
        st.warning("Please load a GitHub repository first.")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the repository..."):
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as error:
                    st.error(f"Something went wrong: {error}")
