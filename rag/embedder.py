"""Wraps a sentence-transformers model to turn text into vectors.

Automatically uses a GPU if one is available (CUDA on Windows/Linux with
an NVIDIA card, or MPS on Apple Silicon), and falls back to CPU otherwise.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Turns text into normalized embedding vectors.

    Vectors are pre-normalized to unit length so cosine similarity later
    reduces to a plain dot product, which is cheaper at query time.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.device = device or _detect_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] Using device: {self.device}")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )
