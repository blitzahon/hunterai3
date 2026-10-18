"""FastAPI backend exposing the RAG agent over HTTP.

Run with: uvicorn api_fastapi:app --host 0.0.0.0 --port 5000 --reload
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import logging
from typing import Optional
from pydantic import BaseModel

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename

from config import settings
from services.job_queue import enqueue_job, get_job, list_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

load_dotenv()

INDEX_PATH = settings.index_path
RAW_DIR = "data/raw"
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf"}

# Pydantic models for request/response
class AskRequest(BaseModel):
    question: str
    history: Optional[list] = []

class AskResponse(BaseModel):
    answer: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    index_exists: bool
    app_name: str
    app_version: str
    documents_directory: str

class Document(BaseModel):
    name: str
    size: int
    type: str

class DocumentsResponse(BaseModel):
    documents: list[Document]

class UploadResponse(BaseModel):
    status: str
    job_id: str
    documents: list[str]

class ReindexResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    index_path: str

# Global agent instance - lazy loaded
_agent = None

def get_agent():
    """Lazy-load the RAG agent on first use to avoid hanging at startup."""
    global _agent
    if _agent is None:
        logger.info("🤖 Initializing RAG agent (lazy-loaded)...")
        if not os.path.exists(INDEX_PATH):
            logger.warning(f"⚠️ Index not found at {INDEX_PATH}")
            raise FileNotFoundError("No index found. Run scripts/build_index.py first.")

        try:
            from rag.agent import RAGAgent
            from rag.embedder import Embedder
            from rag.retriever import Retriever
            from rag.vector_store import VectorStore
            
            logger.info("📥 Loading embedder...")
            embedder = Embedder()
            logger.info("✅ Embedder loaded")
            
            logger.info("📥 Loading vector store...")
            store = VectorStore.load(INDEX_PATH)
            logger.info("✅ Vector store loaded")
            
            logger.info("📥 Initializing retriever...")
            retriever = Retriever(embedder, store)
            logger.info("✅ Retriever initialized")
            
            logger.info("📥 Initializing agent...")
            _agent = RAGAgent(retriever)
            logger.info("✅ Agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}", exc_info=True)
            raise
    return _agent

def create_app() -> FastAPI:
    app = FastAPI(title="HunterAI RAG API", version="1.0.0")
    
    logger.info("🚀 Initializing FastAPI app...")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize auth DB if configured
    if settings.db_url:
        logger.info(f"📦 Initializing database: {settings.db_url}")
        try:
            from services.auth import init_db
            init_db(settings.db_url)
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.warning(f"⚠️ Database initialization failed (continuing): {e}")

    def verify_api_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> Optional[dict]:
        """Dependency to verify API key when DB is configured."""
        # If no DB, allow for backward compatibility (tests/dev)
        if not settings.db_url:
            return None

        token = None
        if authorization:
            if authorization.startswith("ApiKey "):
                token = authorization[len("ApiKey "):].strip()
            elif authorization.startswith("Bearer "):
                token = authorization[len("Bearer "):].strip()
        
        if not token:
            token = x_api_key

        if not token:
            raise HTTPException(status_code=401, detail="Missing API key")

        from services.auth import verify_api_key as verify_token
        ws = verify_token(token)
        if not ws:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return ws

    @app.on_event("startup")
    async def startup_event():
        """Log startup completion - don't load agent here."""
        logger.info("🎉 FastAPI startup complete - agent will load on first request")

    @app.get("/")
    async def index():
        return FileResponse("static/index.html")

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(request: AskRequest):
        question = request.question.strip()
        history = request.history or []

        if not question:
            raise HTTPException(status_code=400, detail="No question provided.")

        if not os.path.exists(INDEX_PATH):
            raise HTTPException(status_code=400, detail="No index found. Run scripts/build_index.py first.")

        try:
            answer = get_agent().answer(question, history=history)
            return AskResponse(answer=answer)
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"The assistant could not answer that question. {str(e)}")

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            index_exists=os.path.exists(INDEX_PATH),
            app_name=settings.app_name,
            app_version=settings.app_version,
            documents_directory=RAW_DIR,
        )

    @app.get("/api/documents", response_model=DocumentsResponse)
    async def documents():
        if not os.path.exists(RAW_DIR):
            return DocumentsResponse(documents=[])

        files = []
        for name in sorted(os.listdir(RAW_DIR)):
            if name.startswith("."):
                continue
            path = os.path.join(RAW_DIR, name)
            if os.path.isfile(path):
                files.append(Document(
                    name=name,
                    size=os.path.getsize(path),
                    type=os.path.splitext(name)[1].lower(),
                ))
        return DocumentsResponse(documents=files)

    @app.post("/api/documents/upload", response_model=UploadResponse)
    async def upload_documents(files: list[UploadFile] = File(...), workspace: Optional[dict] = Depends(verify_api_key)):
        """Save uploaded files and enqueue an indexing job.
        
        This returns immediately with a job id the frontend can poll.
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files were provided.")

        os.makedirs(RAW_DIR, exist_ok=True)

        saved_names = []
        for file in files:
            if not file or not file.filename:
                continue

            original_name = secure_filename(file.filename)
            ext = os.path.splitext(original_name)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")

            target_path = os.path.join(RAW_DIR, original_name)
            content = await file.read()
            with open(target_path, "wb") as f:
                f.write(content)
            saved_names.append(original_name)

        if not saved_names:
            raise HTTPException(status_code=400, detail="No valid files were saved.")

        # Enqueue an indexing job instead of running synchronously
        try:
            job_id = enqueue_job("index", {"files": saved_names})
            return UploadResponse(
                status="queued",
                job_id=job_id,
                documents=saved_names,
            )
        except Exception as e:
            logger.error(f"❌ Failed to enqueue indexing job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to enqueue indexing job. {str(e)}")

    @app.post("/api/reindex", response_model=ReindexResponse)
    async def reindex(workspace: Optional[dict] = Depends(verify_api_key)):
        try:
            from scripts.build_index import build_index
            result = build_index()
            return ReindexResponse(
                status="ok",
                documents=result.get("documents", 0),
                chunks=result.get("chunks", 0),
                index_path=result.get("index_path"),
            )
        except Exception as e:
            logger.error(f"❌ Index rebuild failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Index rebuild failed. {str(e)}")

    @app.get("/api/jobs")
    async def list_all_jobs(workspace: Optional[dict] = Depends(verify_api_key)):
        jobs = list_jobs()
        return {"jobs": jobs}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str, workspace: Optional[dict] = Depends(verify_api_key)):
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"job": job}

    # Mount static files
    app.mount("", StaticFiles(directory="static", html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)
