"""
Stage 1: clone a GitHub repo, scan it, and keep only the useful text files.
This stage only returns a clean list of file paths that the later RAG stages will read.
"""

import os
import shutil
import sys
from urllib.parse import urlparse

from git import GitError, Repo

# Extensions we keep: source code, docs, and config files.
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ipynb",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm", ".dart",
    ".lua", ".r", ".pl", ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".sql", ".graphql", ".proto", ".tf", ".vue", ".svelte",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".xml",
    ".md", ".markdown", ".rst", ".txt", ".adoc",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".tex", ".diff", ".patch",
}

# Extension-less files that are still worth keeping.
TEXT_FILENAMES = {
    "readme", "license", "licence", "copying", "notice", "authors",
    "changelog", "contributing", "codeowners", "makefile", "dockerfile",
    "procfile", "justfile",
}

# Folders we never walk into.
IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "jspm_packages", "vendor", "site-packages",
    "venv", ".venv", "env", "virtualenv",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".nox", ".eggs", ".ipynb_checkpoints",
    "dist", "build", "out", "target", ".next", ".nuxt", ".output",
    ".parcel-cache", ".svelte-kit", ".turbo", ".cache",
    "bin", "obj", ".gradle", ".dart_tool",
    "htmlcov", "coverage", "logs",
    ".idea", ".vscode", ".vs",
    ".terraform", ".serverless", "Pods", "DerivedData",
}

# Files we never keep: lock files, OS junk, and secrets.
IGNORED_FILES = {
    ".ds_store", "thumbs.db", "desktop.ini",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "cargo.lock", "composer.lock", "gemfile.lock", "uv.lock",
    ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig",
    ".coverage", ".env", "id_rsa", "id_rsa.pub", "credentials", ".npmrc", ".netrc",
}

# Text file types we skip because they only hold data, not code or docs.
IGNORED_EXTENSIONS = {
    ".csv",
    ".tsv",
}

# Binary media, archives, fonts, and model files.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".tif", ".tiff",
    ".webp", ".svg", ".svgz", ".heic", ".avif",
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle",
    ".npy", ".npz", ".onnx", ".pt", ".pth", ".h5", ".safetensors",
    ".ckpt", ".model", ".parquet", ".arrow", ".feather",
    ".dll", ".exe", ".so", ".dylib", ".a", ".lib", ".class", ".jar", ".war",
    ".pdb", ".pyc", ".pyo", ".o", ".obj", ".wasm", ".dex", ".pyz",
    ".whl", ".egg", ".deb", ".rpm", ".dmg", ".iso", ".msi",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
}

# Text files that are machine-generated, so not useful as source context.
GENERATED_PATTERNS = {
    ".min.js", ".min.css", ".bundle.js", ".chunk.js", ".d.ts.map",
    "_pb2.py", "_pb2_grpc.py", ".pb.go", "_generated.go", "_version.py",
    ".designer.cs", ".g.cs", ".g.i.cs",
}

MAX_FILE_SIZE_MB = 5


def parse_github_url(url: str) -> tuple:
    """Turn a GitHub URL into (clone_url, repo_name).

    Accepts https://github.com/owner/repo, the .git form, git@github.com:owner/repo,
    and the short owner/repo form.
    """
    if not url or not url.strip():
        raise ValueError("Repository URL cannot be empty.")

    url = url.strip().rstrip("/")

    if url.startswith("git@"):
        # git@github.com:owner/repo.git -> https://github.com/owner/repo
        url = "https://" + url[4:].replace(":", "/", 1)
    elif "://" not in url:
        if url.lower().startswith(("github.com/", "www.github.com/")):
            url = "https://" + url
        else:
            url = "https://github.com/" + url

    url = url[:-4] if url.endswith(".git") else url

    parsed = urlparse(url)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError(
            f"'{url}' is not a GitHub URL. Expected https://github.com/owner/repo"
        )

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            f"'{url}' is missing the owner or repository name. "
            "Expected https://github.com/owner/repo"
        )

    owner, repo = parts[0], parts[1]
    return f"https://github.com/{owner}/{repo}.git", repo


def clone_repository(url: str, destination_root: str = "repos") -> str:
    """Clone the repo shallowly and return its local path."""
    clone_url, repo_name = parse_github_url(url)

    os.makedirs(destination_root, exist_ok=True)
    target_path = os.path.join(destination_root, repo_name)

    if os.path.isdir(target_path):
        shutil.rmtree(target_path)  # always start from a clean copy

    try:
        Repo.clone_from(clone_url, target_path, depth=1)
    except GitError as error:
        message = str(error).lower()
        if "not found" in message:
            raise ValueError(
                f"Repository '{repo_name}' was not found. Check the URL, "
                "or it may be private (then a token is required)."
            ) from error
        if "authentication" in message or "permission" in message:
            raise ValueError(
                f"Access denied for '{repo_name}'. It may be private."
            ) from error
        raise ValueError(f"Could not clone '{repo_name}': {error}") from error

    return os.path.abspath(target_path)


def is_binary_file(file_path: str) -> bool:
    """True if the file content is not readable text."""
    try:
        with open(file_path, "rb") as handle:
            chunk = handle.read(1024)
    except OSError:
        return True

    if b"\x00" in chunk:  # null byte never appears in real text
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def is_relevant_file(file_path: str) -> bool:
    """True if this file is source code, documentation, or config worth keeping."""
    name = os.path.basename(file_path).lower()
    extension = os.path.splitext(name)[1]

    if name in IGNORED_FILES or extension in BINARY_EXTENSIONS:
        return False

    if extension in IGNORED_EXTENSIONS:
        return False

    if any(pattern in name for pattern in GENERATED_PATTERNS):
        return False

    if os.path.getsize(file_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False

    if extension in TEXT_EXTENSIONS:
        return True

    if os.path.splitext(name)[0] in TEXT_FILENAMES:
        return True

    # Unknown extension: keep it only if the content is plain text.
    return not is_binary_file(file_path)


def load_repository_files(url: str, destination_root: str = "repos") -> tuple:
    """Clone a GitHub repo and return (kept_paths, summary).

    summary keys: total_files, files_kept, files_ignored, ignored_dirs, repo_path
    """
    repo_path = clone_repository(url, destination_root)

    kept_files = []
    ignored_dirs = set()
    total_files = 0

    for dir_path, dir_names, file_names in os.walk(repo_path):
        # Prune ignored folders so os.walk never descends into them.
        kept_dirs = []
        for dir_name in dir_names:
            if dir_name.lower() in IGNORED_DIRS:
                ignored_dirs.add(os.path.relpath(os.path.join(dir_path, dir_name), repo_path))
            else:
                kept_dirs.append(dir_name)
        dir_names[:] = kept_dirs

        for file_name in file_names:
            total_files += 1
            full_path = os.path.join(dir_path, file_name)
            if is_relevant_file(full_path):
                kept_files.append(os.path.abspath(full_path))

    kept_files.sort()

    summary = {
        "total_files": total_files,
        "files_kept": len(kept_files),
        "files_ignored": total_files - len(kept_files),
        "ignored_dirs": sorted(ignored_dirs),
        "repo_path": repo_path,
    }
    return kept_files, summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python repo_loader.py https://github.com/owner/repo")

    try:
        files, stats = load_repository_files(sys.argv[1])
    except ValueError as error:
        sys.exit(f"Error: {error}")

    print(f"Total files found : {stats['total_files']}")
    print(f"Files kept        : {stats['files_kept']}")
    print(f"Files ignored     : {stats['files_ignored']}")
    if stats["ignored_dirs"]:
        print(f"Folders skipped   : {', '.join(stats['ignored_dirs'][:10])}")
    print("\nKept files:")
    for path in files:
        print("  -", path)
