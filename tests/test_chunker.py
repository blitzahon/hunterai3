import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import chunk_text


def test_short_text_single_chunk():
    text = "This is a short sentence. Here is another one."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_chunks():
    sentence = "This is a filler sentence used to pad the text out. "
    text = sentence * 40  # well over chunk_size
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_no_sentences_lost():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    joined = " ".join(chunks)
    assert "First sentence" in joined
    assert "Second sentence" in joined
    assert "Third sentence" in joined


def test_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=800, overlap=150) == []


if __name__ == "__main__":
    test_short_text_single_chunk()
    test_long_text_splits_into_multiple_chunks()
    test_no_sentences_lost()
    test_empty_text_returns_empty_list()
    print("All chunker tests passed.")
