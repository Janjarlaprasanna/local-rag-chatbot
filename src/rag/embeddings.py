"""Sentence-transformers embedding wrapper compatible with ChromaDB."""

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a singleton SentenceTransformer instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function using sentence-transformers."""

    def __call__(self, input: Documents) -> Embeddings:
        model = get_model()
        embeddings = model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
