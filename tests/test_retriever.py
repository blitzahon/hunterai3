"""Retriever tests use a fake embedder so they don't require downloading
a sentence-transformers model just to run the test suite.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import Retriever
from rag.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic hash-based 'embedding' — good enough to test wiring,
    not meant to capture real semantic meaning."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=8)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.array(vectors)


def test_index_and_retrieve_roundtrip():
    embedder = FakeEmbedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)

    retriever.index_documents(
        ["Paris is the capital of France. It sits on the Seine river."],
        source_names=["geo.txt"],
    )

    assert len(store.chunks) >= 1
    assert store.metadata[0]["source"] == "geo.txt"

    results = retriever.retrieve("capital of France", top_k=1)
    assert len(results) == 1
    assert "chunk" in results[0]


def test_index_multiple_documents():
    embedder = FakeEmbedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)

    retriever.index_documents(
        ["Document one content here.", "Document two content here."],
        source_names=["doc1.txt", "doc2.txt"],
    )

    sources = {m["source"] for m in store.metadata}
    assert sources == {"doc1.txt", "doc2.txt"}


if __name__ == "__main__":
    test_index_and_retrieve_roundtrip()
    test_index_multiple_documents()
    print("All retriever tests passed.")
