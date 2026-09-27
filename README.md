# 🤖 GitHub Assistant – AI-Powered Repository RAG

> An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to interact with GitHub repositories using natural language.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Embeddings-yellow)](https://huggingface.co/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-blue)](https://ai.google.dev/)

---

## 📌 Overview

**GitHub Assistant** is a RAG-based AI application that allows users to interact with a GitHub repository using natural language.

Instead of manually navigating through a large codebase, users can provide a GitHub repository URL and ask questions about the project.

The application processes the repository, filters unnecessary files, converts relevant files into documents, splits the code into meaningful chunks, generates embeddings, stores them in a vector database, and retrieves the most relevant code when a question is asked.

The retrieved context is then passed to Google Gemini to generate a repository-grounded answer.

---

## 🎯 Purpose

Understanding an unfamiliar GitHub repository can be time-consuming.

Developers often need to search through:

- Multiple directories
- Different programming languages
- Authentication logic
- API implementations
- Database connections
- Machine learning models
- Configuration files
- Utility functions
- Documentation

The goal of GitHub Assistant is to make this process easier by allowing developers to **ask questions directly about the repository**.

For example:

```text
How does authentication work?

Where is the database connection handled?

What technologies are used in this project?

Explain the main project structure.

How does this API work?

Where is the machine learning model loaded?
```

---

## 🚀 Features

- Load a public GitHub repository using its URL
- Automatically clone the repository
- Filter irrelevant files and directories
- Ignore binary and generated files
- Detect programming languages
- Convert repository files into LangChain Documents
- Language-aware code chunking
- Generate semantic embeddings
- Use Qwen3-Embedding-0.6B for embeddings
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Generate answers using Google Gemini
- Include relevant file paths in retrieved context
- Streamlit-based web interface
- Local embedding generation
- Apple Silicon MPS acceleration

---

## 🧠 How It Works

The project follows a Retrieval-Augmented Generation architecture.

```text
                 GitHub Repository URL
                          │
                          ▼
                  Clone Repository
                          │
                          ▼
                Filter Relevant Files
                          │
                          ▼
                 Create Documents
                          │
                          ▼
              Language-Aware Chunking
                          │
                          ▼
                 Generate Embeddings
                          │
                          ▼
                     ChromaDB
                  Vector Database
                          │
                          ▼
                    User Query
                          │
                          ▼
               Semantic Similarity Search
                          │
                          ▼
                Relevant Code Chunks
                          │
                          ▼
                 Gemini + Context
                          │
                          ▼
                    Final Answer
```

---

## 🔄 RAG Pipeline

### 1️⃣ Repository Loading

The user enters a GitHub repository URL through the Streamlit interface.

The application:

1. Validates the GitHub URL
2. Clones the repository using GitPython
3. Uses a shallow clone
4. Scans the repository
5. Filters unnecessary files and directories

Directories such as:

```text
.git
node_modules
venv
__pycache__
build
dist
IDE folders
cache directories
```

are excluded from the processing pipeline.

Binary files, generated files, lock files, secrets, and selected data files are also filtered.

---

### 2️⃣ Document Creation

Every relevant repository file is converted into a LangChain `Document`.

Each document contains:

```text
Document
├── page_content
│   └── Source code / file content
│
└── metadata
    ├── source
    └── language
```

The metadata allows the application to identify the original file associated with retrieved code.

---

### 3️⃣ Language-Aware Chunking

Large source files are divided into smaller chunks before generating embeddings.

Current configuration:

```text
Chunk Size    : 1000
Chunk Overlap : 200
```

For supported programming languages, LangChain's language-aware `RecursiveCharacterTextSplitter` is used.

This helps preserve the structure of source code while creating retrieval-friendly chunks.

---

### 4️⃣ Embedding Generation

Each chunk is converted into a numerical vector representation.

The project currently uses:

```text
Qwen/Qwen3-Embedding-0.6B
```

The embedding model runs locally using Hugging Face embeddings.

On Apple Silicon, the project uses:

```text
MPS (Metal Performance Shaders)
```

for hardware acceleration.

Batch processing is also used to improve embedding efficiency when indexing larger repositories.

---

### 5️⃣ Vector Database

The generated embeddings are stored in:

```text
ChromaDB
```

ChromaDB allows the application to perform semantic similarity search.

For example:

```text
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Relevant Code Chunks
      ↓
Gemini
```

Unlike traditional keyword search, semantic search allows the system to retrieve code based on the meaning of the question.

---

### 6️⃣ Retrieval

When a user asks a question, the application searches the ChromaDB vector database.

The current retriever returns the:

```text
Top 5 relevant chunks
```

The retrieved chunks are then formatted along with their source file paths.

---

### 7️⃣ Answer Generation

The retrieved repository context is passed to Google Gemini.

The prompt instructs the model to:

- Use the provided repository context
- Avoid inventing information
- Mention relevant files when useful
- Clearly state when the requested information cannot be found

This creates a grounded RAG response instead of asking the LLM to answer purely from its pretrained knowledge.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| Streamlit | Web application |
| LangChain | RAG pipeline orchestration |
| GitPython | GitHub repository cloning |
| Hugging Face | Embedding integration |
| Qwen3-Embedding-0.6B | Embedding model |
| ChromaDB | Vector database |
| Google Gemini | Answer generation |
| MPS | Apple Silicon acceleration |
| python-dotenv | Environment variable management |

---

## 📂 Project Structure

```text
Github-Assistant/
│
├── Github Assistant/
│   │
│   ├── app.py
│   │   └── Streamlit application and UI
│   │
│   ├── repo_loader.py
│   │   └── GitHub repository cloning and file filtering
│   │
│   ├── document_loader.py
│   │   └── Converts repository files into LangChain Documents
│   │
│   ├── text_splitter.py
│   │   └── Language-aware code chunking
│   │
│   ├── vector_store.py
│   │   └── Embedding generation and ChromaDB creation
│   │
│   ├── rag.py
│   │   └── Retrieval-Augmented Generation pipeline
│   │
│   └── .streamlit/
│       └── Streamlit configuration
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Complete Application Flow

```text
User enters GitHub URL
          │
          ▼
Repository is cloned
          │
          ▼
Unnecessary files are filtered
          │
          ▼
Relevant files → Documents
          │
          ▼
Documents → Code chunks
          │
          ▼
Chunks → Embeddings
          │
          ▼
Embeddings → ChromaDB
          │
          ▼
User asks a question
          │
          ▼
Question → Embedding
          │
          ▼
Retrieve relevant chunks
          │
          ▼
Chunks + Question → Gemini
          │
          ▼
AI-generated answer
```

---

## 💻 Web Application

The application provides a Streamlit interface where users can:

- Enter a GitHub repository URL
- Load the repository
- Index the repository
- Ask questions about the codebase
- Receive AI-generated answers based on retrieved repository context

The interface is designed to provide a simple ChatGPT-like experience for interacting with source code.

---

## 📊 Example

Suppose a repository contains:

```text
src/
├── auth/
├── database/
├── models/
├── routes/
└── utils/
```

The user can ask:

```text
How does authentication work?
```

The system performs:

```text
Question
   ↓
Question Embedding
   ↓
ChromaDB Search
   ↓
Retrieve Authentication-Related Chunks
   ↓
Provide Context to Gemini
   ↓
Generate Answer
```

The generated answer can also reference relevant source files.

---

# ⚠️ Challenges & Limitations

### 1️⃣ Retrieval Quality

The quality of the final answer depends heavily on the quality of retrieval.

If the relevant code is not retrieved, even a strong LLM may not be able to produce the correct answer.

Currently, the system retrieves the top 5 relevant chunks.

This works well for many targeted questions but can struggle with questions that require information from many different parts of a repository.

Future improvements could include hybrid search and reranking.

---

### 2️⃣ LLM Answer Quality

The current project uses an accessible/free-tier Gemini API setup.

Because of model and API limitations, some generated answers may not be as accurate or detailed as answers produced by more capable models.

Using stronger and more mature models could improve generation quality.

However, improving the LLM alone is not enough.

The retrieval pipeline, chunking strategy, and context selection also have a major impact on the final answer.

---

### 3️⃣ Large Repository Indexing Time

Large repositories can take several minutes to process.

The complete indexing pipeline includes:

```text
Clone
  ↓
Filter Files
  ↓
Load Documents
  ↓
Split Documents
  ↓
Generate Embeddings
  ↓
Store Vectors
```

The embedding stage is particularly computationally expensive because embeddings are generated locally.

The project uses Qwen3-Embedding-0.6B to improve semantic retrieval quality, but the larger model also increases indexing time compared with smaller embedding models.

---

### 4️⃣ Repository-Wide Questions

Some questions require information from the entire repository.

For example:

```text
What programming languages are used throughout this repository?
```

A semantic retriever returning only a few chunks may not be sufficient to answer such questions accurately.

Future versions can address this using repository metadata, hierarchical retrieval, or repository-level summaries.

---

### 5️⃣ No Conversational RAG Yet

Although the interface has a chat-style design, the current RAG pipeline processes questions independently.

Previous questions and answers are not yet used as context for future questions.

Conversational RAG is planned as a future improvement.

---

### 6️⃣ Large Codebases

Very large repositories can contain thousands of files and potentially millions of tokens.

Sending the entire repository directly to an LLM is neither efficient nor practical.

The current system therefore relies on retrieval to select only the most relevant pieces of the codebase.

This makes the system more efficient, but it also introduces the possibility of missing relevant information.

---

## 🚧 Future Improvements

- 🔹 Conversational RAG
- 🔹 Hybrid keyword + semantic search
- 🔹 Reranking retrieved chunks
- 🔹 Better code-aware chunking
- 🔹 Function-level indexing
- 🔹 Class-level indexing
- 🔹 Repository-level summaries
- 🔹 Repository caching
- 🔹 Incremental indexing
- 🔹 Faster embedding generation
- 🔹 Better retrieval evaluation
- 🔹 Streaming responses
- 🔹 Improved source-code references
- 🔹 Support for private repositories
- 🔹 More advanced code understanding models

---

## 📚 What I Learned

Building this project helped me understand that a RAG system is much more than simply connecting an LLM to a vector database.

Through this project I worked with:

- Retrieval-Augmented Generation
- Document loaders
- Document metadata
- Code-aware text splitting
- Embeddings
- Vector databases
- Semantic similarity search
- Retriever configuration
- Prompt engineering
- LangChain LCEL
- Local model inference
- Apple Silicon MPS acceleration
- Streamlit application development
- GitHub repository processing
- File filtering
- Real-world codebase preprocessing

One of my biggest takeaways was:

```text
Better LLM ≠ Automatically Better RAG

Data
 ↓
Filtering
 ↓
Chunking
 ↓
Embeddings
 ↓
Retrieval
 ↓
Context
 ↓
Prompt
 ↓
LLM
 ↓
Final Answer
```

Every stage contributes to the final quality of the system.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/agarwal-gopesh/Github-Assistant.git
```

### 2. Navigate to the project

```bash
cd Github-Assistant
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Gemini API

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Never commit your API key or `.env` file to GitHub.

### 7. Run the application

```bash
python -m streamlit run "Github Assistant/app.py"
```

---

## 🔐 Security

The application is designed to avoid processing unnecessary sensitive files such as:

- `.env`
- Credential files
- Secret files
- Git metadata
- Generated files

API keys should always be stored using environment variables and should never be committed to the repository.

---

## 🎯 Future Vision

The long-term goal is to turn GitHub Assistant into a more capable AI developer tool that can understand an entire codebase rather than simply retrieving isolated code chunks.

Potential future capabilities include:

```text
Repository
    ↓
Repository Understanding
    ↓
Architecture Mapping
    ↓
Dependency Understanding
    ↓
Code Retrieval
    ↓
Reasoning
    ↓
AI Developer Assistant
```

This could eventually allow the system to explain architecture, trace code execution, identify dependencies, explain bugs, and assist developers in navigating unfamiliar codebases.

---

## 👨‍💻 Author

### Gopesh Agarwal

AI / ML | Generative AI | RAG | MLOps

GitHub:

https://github.com/agarwal-gopesh

LinkedIn:

https://www.linkedin.com/in/gopesh-agarwal-81a744378/

---

## 🔗 Project Repository

https://github.com/agarwal-gopesh/Github-Assistant

---

⭐ If you found this project interesting, consider giving the repository a star!
