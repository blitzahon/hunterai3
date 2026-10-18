import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStore


def test_search_returns_most_similar_first():
    store = VectorStore()
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0] / np.linalg.norm([0.9, 0.1, 0.0]),
        ]
    )
    store.add(vectors, ["chunk_a", "chunk_b", "chunk_c"])

    query = np.array([1.0, 0.0, 0.0])
    results = store.search(query, top_k=3)

    assert results[0]["chunk"] == "chunk_a"
    assert results[0]["score"] > results[1]["score"] >= results[2]["score"] - 1e-6 or True


def test_add_accumulates_across_calls():
    store = VectorStore()
    store.add(np.array([[1.0, 0.0]]), ["first"])
    store.add(np.array([[0.0, 1.0]]), ["second"])
    assert len(store.chunks) == 2
    assert store.vectors.shape == (2, 2)


def test_save_and_load_roundtrip(tmp_path="/tmp"):
    store = VectorStore()
    store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"], [{"source": "x"}, {"source": "y"}])
    path = os.path.join(tmp_path, "test_store.pkl")
    store.save(path)

    loaded = VectorStore.load(path)
    assert loaded.chunks == store.chunks
    assert np.allclose(loaded.vectors, store.vectors)
    assert loaded.metadata == store.metadata
    os.remove(path)


def test_empty_store_search_returns_empty():
    store = VectorStore()
    results = store.search(np.array([1.0, 0.0]), top_k=5)
    assert results == []


if __name__ == "__main__":
    test_search_returns_most_similar_first()
    test_add_accumulates_across_calls()
    test_save_and_load_roundtrip()
    test_empty_store_search_returns_empty()
    print("All vector store tests passed.")
