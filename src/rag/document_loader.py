"""Load and chunk PDF, TXT, and CSV documents using LangChain."""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _get_loader(file_path: str):
    """Return the appropriate LangChain loader for a file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        return CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


def load_and_chunk(file_path: str) -> list[dict]:
    """Load a document and split it into chunks.

    Returns a list of dicts with keys: text, metadata.
    Each chunk gets a deterministic ID based on source and index.
    """
    loader = _get_loader(file_path)
    documents = loader.load()
    chunks = _splitter.split_documents(documents)

    source_name = Path(file_path).name
    results = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": source_name,
            "chunk_index": i,
            **{k: v for k, v in chunk.metadata.items() if k != "source"},
        }
        results.append({
            "id": f"{source_name}__chunk_{i}",
            "text": chunk.page_content,
            "metadata": metadata,
        })

    return results
