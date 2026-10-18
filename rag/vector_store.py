"""A flat, in-memory vector store with cosine-similarity search.

For small-to-medium corpora (thousands of chunks) a plain numpy matrix and
matrix multiplication is fast enough. You don't need FAISS/HNSW-style
indexing until you're well into the hundreds of thousands of vectors.
"""

import os
import pickle

import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, chunks: list[str], metadata: list[dict] | None = None):
        metadata = metadata or [{} for _ in chunks]
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        # vectors are pre-normalized, so dot product == cosine similarity
        sims = self.vectors @ query_vector
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {"chunk": self.chunks[i], "score": float(sims[i]), "metadata": self.metadata[i]}
            for i in top_idx
        ]

    def save(self, path: str):
        target_path = os.path.abspath(path)
        directory = os.path.dirname(target_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(
                {"vectors": self.vectors, "chunks": self.chunks, "metadata": self.metadata}, f
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.vectors = data["vectors"]
        store.chunks = data["chunks"]
        store.metadata = data["metadata"]
        return store
