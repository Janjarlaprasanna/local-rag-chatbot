"""Tests for the embedding module."""

import numpy as np
from rag.embeddings import get_model, embed_texts, embed_query, LocalEmbeddingFunction


def test_model_loads():
    """Model loads successfully and is reusable (singleton)."""
    model1 = get_model()
    model2 = get_model()
    assert model1 is model2


def test_embedding_dimensions():
    """Embeddings have correct dimensionality (384 for MiniLM)."""
    vec = embed_query("hello world")
    assert len(vec) == 384


def test_embed_texts_batch():
    """Batch embedding returns correct number of vectors."""
    texts = ["first sentence", "second sentence", "third sentence"]
    vectors = embed_texts(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_semantic_similarity():
    """Semantically similar texts have higher cosine similarity than dissimilar ones."""
    vecs = embed_texts([
        "The cat sat on the mat",
        "A kitten was resting on the rug",
        "Quantum computing uses qubits for calculations",
    ])

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_related = cosine_sim(vecs[0], vecs[1])
    sim_unrelated = cosine_sim(vecs[0], vecs[2])
    assert sim_related > sim_unrelated


def test_chroma_embedding_function():
    """LocalEmbeddingFunction works as a ChromaDB-compatible callable."""
    fn = LocalEmbeddingFunction()
    result = fn(["test document"])
    assert len(result) == 1
    assert len(result[0]) == 384
