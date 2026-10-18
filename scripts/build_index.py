"""One-off script: reads data/raw/, builds the index, saves it to disk.

Run this whenever you add new documents or change chunking/embedding
settings. Re-run wipes and rebuilds the full index (simple, avoids
duplicate-chunk bookkeeping).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader

from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.vector_store import VectorStore

RAW_DIR = "data/raw"
INDEX_PATH = "data/index/store.pkl"


def read_txt(path: str) -> str | None:
    # Windows text files are often saved as cp1252/latin-1, not utf-8
    # (smart quotes, em-dashes, etc. trip a strict utf-8 read). Try
    # utf-8 first, then fall back gracefully instead of crashing.
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return None


def read_pdf(path: str) -> str | None:
    try:
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception as e:
        print(f"  Failed to extract text from PDF: {e}")
        return None


def build_index(raw_dir: str = RAW_DIR, index_path: str = INDEX_PATH) -> dict:
    """Build or rebuild the persisted vector index from raw documents."""
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)

    docs, names = [], []
    for fname in sorted(os.listdir(raw_dir)):
        if fname.startswith("."):
            continue
        path = os.path.join(raw_dir, fname)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            content = read_pdf(path)
        elif ext in (".txt", ".md"):
            content = read_txt(path)
        else:
            print(f"  Skipping {fname}: unsupported file type ({ext}).")
            continue

        if content is None:
            print(f"  Skipping {fname}: could not extract readable text.")
            continue

        docs.append(content)
        names.append(fname)

    if not docs:
        return {
            "documents": 0,
            "chunks": 0,
            "index_path": index_path,
            "status": "empty",
        }

    retriever.index_documents(docs, names)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    store.save(index_path)

    return {
        "documents": len(names),
        "chunks": len(store.chunks),
        "index_path": index_path,
        "status": "built",
    }


def main():
    result = build_index()
    if result["status"] == "empty":
        print(f"No documents found in {RAW_DIR}/. Add some .txt or .pdf files and re-run.")
        return

    print(f"Indexed {result['documents']} document(s) into {result['chunks']} chunks.")
    print(f"Saved to {result['index_path']}")


if __name__ == "__main__":
    main()