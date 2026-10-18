# RAG Agent

A retrieval-augmented generation agent built from scratch — no LangChain or
managed vector database. It uses a flat NumPy matrix for vectors,
sentence-transformers for embeddings, and Groq for answers.

## Product setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then add GROQ_API_KEY
```

Set environment variables when needed:

```bash
set RAG_HOST=127.0.0.1
set RAG_PORT=5000
set RAG_DEBUG=false
set RAG_INDEX_PATH=data/index/store.pkl
```

## Product workflow

1. Drop your source documents (`.txt`, `.md`, and `.pdf` files) into `data/raw/`.
2. Build or rebuild the index:

   ```bash
   python scripts/build_index.py
   ```

3. Start the FastAPI server:

   ```bash
   uvicorn api_fastapi:app --host 0.0.0.0 --port 5000
   ```

4. Ask questions through the API or the UI:

   ```bash
   python scripts/ask.py "What does doc1 say about deployment?"
   ```

The index is saved to `data/index/store.pkl` so you only re-embed when you
add new documents or change chunking/embedding settings.

## HTTP product surface

- `GET /api/health` — returns runtime/product status and index presence
- `POST /api/ask` — answers a question with prior history support
- `GET /api/documents` — lists the currently uploaded raw document files
- `POST /api/reindex` — rebuilds the vector store from `data/raw`

## How it works

- **`rag/chunker.py`** — splits text into overlapping, sentence-aware chunks
- **`rag/embedder.py`** — wraps sentence-transformers to turn text into vectors
- **`rag/vector_store.py`** — flat numpy matrix + cosine similarity search, with save/load
- **`rag/retriever.py`** — glues embedder + store together for indexing and search
- **`rag/tools.py`** — the `retrieve` tool schema exposed to Claude
- **`rag/agent.py`** — the agent loop: Claude decides when to call `retrieve`,
  can call it multiple times for multi-hop questions, and skips it entirely
  for questions that don't need the knowledge base

## GPU support

`rag/embedder.py` auto-detects and uses a GPU if available (CUDA on
NVIDIA, MPS on Apple Silicon), otherwise falls back to CPU. It prints
which device it picked on startup, e.g. `[Embedder] Using device: cuda`.

If it prints `cpu` but you have an NVIDIA GPU, your `torch` install is
CPU-only. Fix it:

```cmd
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(Swap `cu121` for the CUDA version matching your driver — check with
`nvidia-smi`. Use `cu118` for older drivers.)

## Web UI

The included static UI is served by FastAPI; no Node.js build is needed.

```cmd
uvicorn api_fastapi:app --host 0.0.0.0 --port 5000
```

Open http://localhost:5000.


## Extending


- Swap `all-MiniLM-L6-v2` in `rag/embedder.py` for a larger sentence-transformers
  model, or point `Embedder` at an API-based embedding model (e.g. Voyage AI)
- Add hybrid search (BM25 + vectors) in `rag/retriever.py` for exact-term recall
- Add a reranking step between retrieval and generation for precision
