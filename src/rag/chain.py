"""Core RAG chain: retrieve → prompt → generate."""

from collections.abc import Generator

from rag.config import RAG_PROMPT_TEMPLATE, TOP_K
from rag.vector_store import query as vector_query
from rag.llm import generate, generate_stream


def build_prompt(question: str, context_docs: list[dict]) -> str:
    """Build the RAG prompt from a question and retrieved documents."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc["metadata"].get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc['text']}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def ask(question: str, top_k: int = TOP_K) -> dict:
    """Run the full RAG pipeline and return the answer.

    Returns dict with keys: answer, sources, num_chunks.
    """
    context_docs = vector_query(question, top_k=top_k)
    prompt = build_prompt(question, context_docs)
    answer = generate(prompt)

    sources = list({doc["metadata"].get("source", "unknown") for doc in context_docs})

    return {
        "answer": answer,
        "sources": sorted(sources),
        "num_chunks": len(context_docs),
    }


def ask_stream(question: str, top_k: int = TOP_K) -> Generator[str | dict, None, None]:
    """Stream the RAG answer token by token.

    Yields string tokens, then a final dict with metadata.
    """
    context_docs = vector_query(question, top_k=top_k)
    prompt = build_prompt(question, context_docs)

    sources = sorted({doc["metadata"].get("source", "unknown") for doc in context_docs})

    for token in generate_stream(prompt):
        yield token

    # Final metadata yield
    yield {
        "sources": sources,
        "num_chunks": len(context_docs),
    }
