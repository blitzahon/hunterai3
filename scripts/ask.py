"""CLI: python scripts/ask.py "your question"

Loads the pre-built index (run build_index.py first) and asks the agent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from rag.agent import RAGAgent
from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.vector_store import VectorStore

INDEX_PATH = "data/index/store.pkl"


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        return

    if not os.path.exists(INDEX_PATH):
        print(f"No index found at {INDEX_PATH}. Run scripts/build_index.py first.")
        return

    question = " ".join(sys.argv[1:])

    embedder = Embedder()
    store = VectorStore.load(INDEX_PATH)
    retriever = Retriever(embedder, store)
    agent = RAGAgent(retriever)

    print(agent.answer(question))


if __name__ == "__main__":
    main()
