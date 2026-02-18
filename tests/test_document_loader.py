"""Tests for the document loader module."""

from pathlib import Path

import pytest
from rag.document_loader import load_and_chunk, SUPPORTED_EXTENSIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_supported_extensions():
    """Module declares supported file types."""
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS


def test_load_txt():
    """TXT file loads and chunks correctly."""
    chunks = load_and_chunk(str(DATA_DIR / "sample.txt"))
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)
    assert all("id" in c for c in chunks)


def test_load_csv():
    """CSV file loads and chunks correctly."""
    chunks = load_and_chunk(str(DATA_DIR / "sample.csv"))
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "sample.csv"


def test_load_pdf():
    """PDF file loads and chunks correctly."""
    pdf_path = DATA_DIR / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("sample.pdf not found")
    chunks = load_and_chunk(str(pdf_path))
    assert len(chunks) > 0


def test_chunk_ids_are_deterministic():
    """Same file produces same chunk IDs on repeated loads."""
    chunks1 = load_and_chunk(str(DATA_DIR / "sample.txt"))
    chunks2 = load_and_chunk(str(DATA_DIR / "sample.txt"))
    ids1 = [c["id"] for c in chunks1]
    ids2 = [c["id"] for c in chunks2]
    assert ids1 == ids2


def test_chunk_id_format():
    """Chunk IDs follow the expected format: {filename}__chunk_{i}."""
    chunks = load_and_chunk(str(DATA_DIR / "sample.txt"))
    for i, chunk in enumerate(chunks):
        assert chunk["id"] == f"sample.txt__chunk_{i}"


def test_chunk_metadata_has_source():
    """Every chunk's metadata includes the source filename."""
    chunks = load_and_chunk(str(DATA_DIR / "sample.txt"))
    for chunk in chunks:
        assert "source" in chunk["metadata"]
        assert chunk["metadata"]["source"] == "sample.txt"


def test_unsupported_extension():
    """Unsupported file types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_and_chunk("document.docx")
