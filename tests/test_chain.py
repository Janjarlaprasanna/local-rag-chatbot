"""Tests for the RAG chain module (mocked LLM — no Ollama needed)."""

from unittest.mock import patch

from rag.chain import build_prompt, ask, ask_stream


MOCK_DOCS = [
    {
        "text": "Acme Corp was founded in 2018 by Dr. Sarah Chen.",
        "metadata": {"source": "sample.txt", "chunk_index": 0},
        "distance": 0.15,
    },
    {
        "text": "The company is headquartered in Austin, Texas.",
        "metadata": {"source": "sample.txt", "chunk_index": 1},
        "distance": 0.22,
    },
]


def test_build_prompt_includes_context():
    """Prompt contains the document context."""
    prompt = build_prompt("When was Acme founded?", MOCK_DOCS)
    assert "Acme Corp was founded in 2018" in prompt
    assert "Austin, Texas" in prompt


def test_build_prompt_includes_question():
    """Prompt contains the user's question."""
    prompt = build_prompt("When was Acme founded?", MOCK_DOCS)
    assert "When was Acme founded?" in prompt


def test_build_prompt_includes_source_labels():
    """Prompt labels each context chunk with its source."""
    prompt = build_prompt("test question", MOCK_DOCS)
    assert "[Source: sample.txt]" in prompt


def test_build_prompt_empty_docs():
    """Prompt handles empty document list gracefully."""
    prompt = build_prompt("test question", [])
    assert "No relevant documents found" in prompt


def test_build_prompt_has_instructions():
    """Prompt includes anti-hallucination instructions."""
    prompt = build_prompt("test", MOCK_DOCS)
    assert "ONLY" in prompt or "only" in prompt.lower()
    assert "don't know" in prompt.lower() or "don't have enough" in prompt.lower()


@patch("rag.chain.vector_query", return_value=MOCK_DOCS)
@patch("rag.chain.generate", return_value="Acme Corp was founded in 2018.")
def test_ask_returns_answer(mock_gen, mock_query):
    """ask() returns answer with sources."""
    result = ask("When was Acme founded?")
    assert "answer" in result
    assert "sources" in result
    assert "num_chunks" in result
    assert result["answer"] == "Acme Corp was founded in 2018."
    assert "sample.txt" in result["sources"]
    assert result["num_chunks"] == 2


@patch("rag.chain.vector_query", return_value=MOCK_DOCS)
@patch("rag.chain.generate", return_value="Acme Corp was founded in 2018.")
def test_ask_calls_with_correct_question(mock_gen, mock_query):
    """ask() passes the question to vector_query."""
    ask("When was Acme founded?")
    mock_query.assert_called_once_with("When was Acme founded?", top_k=5)


@patch("rag.chain.vector_query", return_value=MOCK_DOCS)
@patch("rag.chain.generate_stream", return_value=iter(["Acme ", "was ", "founded."]))
def test_ask_stream_yields_tokens(mock_stream, mock_query):
    """ask_stream() yields string tokens then a metadata dict."""
    tokens = list(ask_stream("When was Acme founded?"))

    # String tokens
    string_tokens = [t for t in tokens if isinstance(t, str)]
    assert len(string_tokens) == 3
    assert "".join(string_tokens) == "Acme was founded."

    # Final metadata
    metadata = [t for t in tokens if isinstance(t, dict)]
    assert len(metadata) == 1
    assert "sources" in metadata[0]
    assert "sample.txt" in metadata[0]["sources"]
