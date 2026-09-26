"""
Stage 3: split the Documents from document_loader into smaller chunks.

Programming languages use LangChain's language-aware RecursiveCharacterTextSplitter.
Everything else (markdown, json, yaml, css, plain text, unknown) uses the plain
RecursiveCharacterTextSplitter. Embeddings and vector search come later.
"""

import sys

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from document_loader import load_documents

# Tune these two numbers for your embedding model later.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# The "language" values produced by document_loader that have a LangChain
# language-aware splitter. Everything not listed here falls back to the plain one.
LANGUAGE_SPLITTERS = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "tsx": Language.TS,
    "java": Language.JAVA,
    "kotlin": Language.KOTLIN,
    "scala": Language.SCALA,
    "c": Language.C,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "go": Language.GO,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "php": Language.PHP,
    "swift": Language.SWIFT,
    "lua": Language.LUA,
    "r": Language.R,
    "powershell": Language.POWERSHELL,
    "protobuf": Language.PROTO,
    "latex": Language.LATEX,
}

# Built once and reused, since splitters hold no per-document state.
_plitter_cache = {}


def get_splitter(language: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware splitter, or the plain one for unsupported types."""
    if language not in _plitter_cache:
        splitter = None
        langchain_language = LANGUAGE_SPLITTERS.get(language)
        if langchain_language is not None:
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=langchain_language,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
            except ValueError:
                # Some Language members exist in the enum but have no
                # separators yet, so quietly fall back to the plain splitter.
                splitter = None

        if splitter is None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

        _plitter_cache[language] = splitter

    return _plitter_cache[language]


def split_document(document: Document) -> list:
    """Split one Document into chunk Documents, keeping its metadata."""
    language = document.metadata.get("language", "unknown")
    return get_splitter(language).split_documents([document])


def split_documents(documents: list) -> list:
    """Split many Documents and return one flat list of chunk Documents."""
    chunks = []
    for document in documents:
        chunks.extend(split_document(document))
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python text_splitter.py https://github.com/owner/repo")

    try:
        documents = load_documents(sys.argv[1])
    except ValueError as error:
        sys.exit(f"Error: {error}")

    chunks = split_documents(documents)

    print(f"Documents loaded : {len(documents)}")
    print(f"Chunks created   : {len(chunks)}")
    if documents:
        average = len(chunks) / len(documents)
        print(f"Chunks per file  : {average:.1f} average")

    missing = [chunk for chunk in chunks if "source" not in chunk.metadata]
    print(f"Chunks missing metadata : {len(missing)}")

    print("\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\n  source   : {chunk.metadata['source']}")
        print(f"  language : {chunk.metadata['language']}")
        print(f"  chars    : {len(chunk.page_content)}")
        print("  preview  : " + chunk.page_content[:200].replace("\n", " ") + "...")
