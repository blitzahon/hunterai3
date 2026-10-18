"""Splits raw text into overlapping, sentence-aware chunks for embedding."""

import re


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, breaking on sentence boundaries.

    Naive fixed-character chunking cuts sentences in half and loses meaning.
    This breaks on sentence boundaries and carries a bit of overlap forward
    so facts that sit near a chunk boundary aren't stranded in isolation.

    Args:
        text: The raw document text.
        chunk_size: Target max characters per chunk.
        overlap: Approximate characters of trailing context to carry into
            the next chunk.

    Returns:
        A list of chunk strings.
    """
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))

            # keep trailing sentences for overlap continuity
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current, current_len = overlap_sents, overlap_len

        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks
