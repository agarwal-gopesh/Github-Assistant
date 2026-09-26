"""
Stage 2: turn the filtered file paths from repo_loader into LangChain Documents.

One Document per file, with the file text in page_content and the path plus
detected language in metadata. Chunking and embeddings come in later stages.
"""

import os
import sys

from langchain_core.documents import Document

from repo_loader import load_repository_files

# Extension -> human readable language name.
LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".pyi": "python",
    ".ipynb": "jupyter_notebook",
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".groovy": "groovy",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".m": "objective_c", ".mm": "objective_c",
    ".dart": "dart", ".lua": "lua", ".r": "r", ".pl": "perl",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".ps1": "powershell", ".bat": "batch",
    ".sql": "sql", ".graphql": "graphql", ".proto": "protobuf",
    ".tf": "terraform", ".vue": "vue", ".svelte": "svelte",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "sass", ".sass": "sass", ".less": "less",
    ".xml": "xml",
    ".md": "markdown", ".markdown": "markdown",
    ".rst": "restructuredtext", ".adoc": "asciidoc", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "config", ".cfg": "config", ".conf": "config",
    ".tex": "latex", ".diff": "diff", ".patch": "diff",
}

# Files that have no extension, so we match on the name instead.
LANGUAGE_BY_FILENAME = {
    "readme": "markdown",
    "license": "text",
    "licence": "text",
    "makefile": "makefile",
    "dockerfile": "dockerfile",
    "changelog": "text",
    "contributing": "text",
}


def detect_language(file_path: str) -> str:
    """Guess the language from the extension, falling back to the filename."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in LANGUAGE_BY_EXTENSION:
        return LANGUAGE_BY_EXTENSION[extension]

    name = os.path.splitext(os.path.basename(file_path).lower())[0]
    if name in LANGUAGE_BY_FILENAME:
        return LANGUAGE_BY_FILENAME[name]

    return "unknown"


def read_file(file_path: str) -> str:
    """Read a file as text, replacing any undecodable bytes."""
    with open(file_path, "rb") as handle:
        return handle.read().decode("utf-8", errors="replace")


def create_document(file_path: str) -> Document:
    """Build one Document from a single file."""
    return Document(
        page_content=read_file(file_path),
        metadata={
            "source": file_path,
            "language": detect_language(file_path),
        },
    )


def documents_from_paths(file_paths: list) -> list:
    """Convert file paths into Documents, skipping any file we cannot read."""
    documents = []
    for file_path in file_paths:
        try:
            documents.append(create_document(file_path))
        except OSError as error:
            print(f"Warning: skipping {file_path} ({error})")
    return documents


def load_documents(url: str, destination_root: str = "repos") -> list:
    """Clone a GitHub repo and return one Document per relevant file."""
    file_paths, _ = load_repository_files(url, destination_root)
    return documents_from_paths(file_paths)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python document_loader.py https://github.com/owner/repo")

    try:
        documents = load_documents(sys.argv[1])
    except ValueError as error:
        sys.exit(f"Error: {error}")

    print(f"Documents created : {len(documents)}")

    languages = {}
    for document in documents:
        name = document.metadata["language"]
        languages[name] = languages.get(name, 0) + 1
    print("Languages         : " + ", ".join(
        f"{name}={count}" for name, count in sorted(languages.items(), key=lambda pair: -pair[1])
    ))

    print("\nFirst 3 Documents:")
    for document in documents[:3]:
        print(f"\n  source   : {document.metadata['source']}")
        print(f"  language : {document.metadata['language']}")
        print(f"  chars    : {len(document.page_content)}")
        print("  preview  : " + document.page_content[:200].replace("\n", " ") + "...")
