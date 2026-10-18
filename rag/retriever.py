"""Glues the embedder and vector store together for indexing and search.

Note: this module intentionally avoids importing rag.embedder at module
level, since that pulls in sentence-transformers (a heavy dependency).
Any object with an `.embed(texts) -> np.ndarray` method works here,
including the lightweight FakeEmbedder used in tests.
"""

from typing import TYPE_CHECKING

from rag.chunker import chunk_text

if TYPE_CHECKING:
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: "Embedder", store: "VectorStore"):
        self.embedder = embedder
        self.store = store

    def index_documents(self, documents: list[str], source_names: list[str] | None = None):
        source_names = source_names or [f"doc_{i}" for i in range(len(documents))]
        for doc, name in zip(documents, source_names):
            chunks = chunk_text(doc)
            if not chunks:
                continue
            vectors = self.embedder.embed(chunks)
            metadata = [{"source": name, "chunk_index": i} for i in range(len(chunks))]
            self.store.add(vectors, chunks, metadata)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, top_k)
