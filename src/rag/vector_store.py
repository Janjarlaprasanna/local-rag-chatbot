"""ChromaDB vector store operations."""

import chromadb

from rag.config import CHROMA_DB_DIR, CHROMA_COLLECTION, TOP_K, BATCH_SIZE
from rag.embeddings import LocalEmbeddingFunction

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """Return a singleton ChromaDB PersistentClient."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the default collection with local embeddings."""
    client = get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=LocalEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[dict], progress_callback=None) -> int:
    """Add document chunks to the vector store in batches.

    Args:
        chunks: List of dicts with keys: id, text, metadata.
        progress_callback: Optional callable(done, total) for progress updates.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_collection()
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, total), total)

    return total


def query(question: str, top_k: int = TOP_K) -> list[dict]:
    """Query the vector store for relevant chunks."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )

    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return documents


def list_sources() -> list[str]:
    """Return a sorted list of unique source names in the store."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({m.get("source", "unknown") for m in all_metadata})
    return sources


def get_document_count() -> int:
    """Return total number of chunks in the store."""
    return get_collection().count()


def clear_collection():
    """Delete and recreate the collection."""
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
