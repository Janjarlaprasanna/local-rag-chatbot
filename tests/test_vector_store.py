"""Tests for the vector store module."""

import pytest
from rag.vector_store import (
    add_documents,
    query,
    list_sources,
    get_document_count,
    clear_collection,
    get_collection,
)


@pytest.fixture(autouse=True)
def clean_collection():
    """Clear collection before and after each test."""
    clear_collection()
    yield
    clear_collection()


SAMPLE_CHUNKS = [
    {
        "id": "test.txt__chunk_0",
        "text": "Acme Corp was founded in 2018 by Dr. Sarah Chen.",
        "metadata": {"source": "test.txt", "chunk_index": 0},
    },
    {
        "id": "test.txt__chunk_1",
        "text": "The engineering team uses Python and Go for backend services.",
        "metadata": {"source": "test.txt", "chunk_index": 1},
    },
    {
        "id": "report.pdf__chunk_0",
        "text": "Q3 revenue reached $7.2 million with 40% year-over-year growth.",
        "metadata": {"source": "report.pdf", "chunk_index": 0},
    },
]


def test_add_documents():
    """Adding documents returns correct count."""
    count = add_documents(SAMPLE_CHUNKS)
    assert count == 3


def test_add_empty_list():
    """Adding empty list returns 0."""
    assert add_documents([]) == 0


def test_document_count():
    """Document count reflects added chunks."""
    assert get_document_count() == 0
    add_documents(SAMPLE_CHUNKS)
    assert get_document_count() == 3


def test_query_returns_results():
    """Query returns relevant results."""
    add_documents(SAMPLE_CHUNKS)
    results = query("When was Acme Corp founded?", top_k=2)
    assert len(results) > 0
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("distance" in r for r in results)


def test_query_relevance():
    """Most relevant result should match the query topic."""
    add_documents(SAMPLE_CHUNKS)
    results = query("When was Acme Corp founded?", top_k=1)
    assert "2018" in results[0]["text"] or "founded" in results[0]["text"]


def test_query_empty_store():
    """Querying empty store returns empty list."""
    results = query("anything")
    assert results == []


def test_list_sources():
    """Sources are correctly listed."""
    add_documents(SAMPLE_CHUNKS)
    sources = list_sources()
    assert "test.txt" in sources
    assert "report.pdf" in sources
    assert len(sources) == 2


def test_clear_collection():
    """Clearing collection removes all documents."""
    add_documents(SAMPLE_CHUNKS)
    assert get_document_count() == 3
    clear_collection()
    assert get_document_count() == 0


def test_upsert_deduplication():
    """Re-adding same IDs updates rather than duplicates."""
    add_documents(SAMPLE_CHUNKS)
    assert get_document_count() == 3
    add_documents(SAMPLE_CHUNKS)
    assert get_document_count() == 3
