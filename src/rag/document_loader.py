"""Load and chunk PDF, TXT, and CSV documents."""

import csv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _load_csv_fast(file_path: str) -> list[dict]:
    """Load CSV by batching rows into chunks directly — skips LangChain's one-doc-per-row overhead."""
    source_name = Path(file_path).name
    results = []
    chunk_idx = 0
    buffer = []
    buffer_len = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []
        header_line = ",".join(header)

        for row in reader:
            line = ",".join(row)
            line_len = len(line) + 1  # +1 for newline

            if buffer_len + line_len > CHUNK_SIZE and buffer:
                text = header_line + "\n" + "\n".join(buffer)
                results.append({
                    "id": f"{source_name}__chunk_{chunk_idx}",
                    "text": text,
                    "metadata": {"source": source_name, "chunk_index": chunk_idx},
                })
                chunk_idx += 1
                buffer = []
                buffer_len = 0

            buffer.append(line)
            buffer_len += line_len

        # Flush remaining
        if buffer:
            text = header_line + "\n" + "\n".join(buffer)
            results.append({
                "id": f"{source_name}__chunk_{chunk_idx}",
                "text": text,
                "metadata": {"source": source_name, "chunk_index": chunk_idx},
            })

    return results


def load_and_chunk(file_path: str) -> list[dict]:
    """Load a document and split it into chunks.

    Returns a list of dicts with keys: id, text, metadata.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    source_name = Path(file_path).name

    # CSV: fast direct chunking, no LangChain overhead
    if ext == ".csv":
        return _load_csv_fast(file_path)

    # PDF / TXT: use LangChain loaders
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    chunks = _splitter.split_documents(documents)

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
