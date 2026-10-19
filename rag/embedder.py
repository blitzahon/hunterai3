"""Turns text into normalized embedding vectors using Vertex AI.

No local model and no torch: embeddings come from Google's embedding API,
authenticated by the Cloud Run service account (or by
`gcloud auth application-default login` on your PC).
"""

import os

import numpy as np
from google import genai
from google.genai import types

DEFAULT_MODEL = "text-embedding-005"


class Embedder:
    """Turns text into normalized embedding vectors.

    Vectors are normalized to unit length so cosine similarity later
    reduces to a plain dot product, which is cheaper at query time.
    """

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size or int(os.getenv("EMBED_BATCH_SIZE", "16"))

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Set it to your GCP project ID."
            )
        # Embedding models are regional, so this is separate from the Gemini
        # location (which may be "global").
        location = os.getenv("EMBEDDING_LOCATION", "us-central1")
        self.client = genai.Client(vertexai=True, project=project, location=location)
        print(f"[Embedder] Using Vertex AI model {self.model_name} in {location}")

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                vectors.extend(self._embed_batch(batch, task_type))
            except Exception:
                if len(batch) == 1:
                    raise
                # Some embedding models accept only one text per request.
                for text in batch:
                    vectors.extend(self._embed_batch([text], task_type))

        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks for indexing."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Embed search queries (a different task type gives better retrieval)."""
        return self._embed(texts, "RETRIEVAL_QUERY")