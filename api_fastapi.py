"""FastAPI backend exposing the RAG agent over HTTP.

Run with: uvicorn api_fastapi:app --host 0.0.0.0 --port 5000 --reload
Then open http://localhost:5000 in your browser.
"""

import os
import secrets
import sys
import threading
import time
import logging
from typing import Optional
from pydantic import BaseModel

from dotenv import load_dotenv

# Load .env before importing modules that read settings at import time.
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Request, Response, Body
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename

from config import settings
from services import auth, google_auth
from services.auth import init_db, verify_api_key, create_api_key
from services.job_queue import enqueue_job, get_job, list_jobs

# Configure logging
if os.getenv("GAE_ENV") or os.getenv("CLOUD_RUN_EXECUTION") or os.getenv("GOOGLE_CLOUD_PROJECT"):
    # We are running in Google Cloud (App Engine, Cloud Run, GKE, or Compute Engine)
    import google.cloud.logging
    client = google.cloud.logging.Client()
    client.setup_logging()
    logger = logging.getLogger(__name__)
else:
    # Fallback for local development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        stream=sys.stdout,
        force=True
    )
    logger = logging.getLogger(__name__)

INDEX_PATH = settings.index_path
RAW_DIR = "data/raw"
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024  # Cloud Run caps requests at 32 MB
INDEX_ON_UPLOAD = os.getenv("INDEX_ON_UPLOAD", "true").lower() in {"1", "true", "yes", "on"}


# Pydantic models for request/response
class AskRequest(BaseModel):
    question: str
    history: Optional[list] = []

class GoogleLogin(BaseModel):
    credential: str

class AskResponse(BaseModel):
    answer: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    index_exists: bool
    index_updated_at: Optional[float] = None
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
    job_id: Optional[str] = None
    documents: list[str]

class ReindexResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    index_path: str

# Global agent instance - lazy loaded, and reloaded when the index file changes
_agent = None
_embedder = None
_index_mtime = None

# State for bucket-triggered indexing (one build at a time, extra changes coalesced)
_index_state_lock = threading.Lock()
_index_running = False
_index_dirty = False


def get_agent():
    """Lazy-load the RAG agent, reloading the vector store if store.pkl was rebuilt."""
    global _agent, _embedder, _index_mtime

    if not os.path.exists(INDEX_PATH):
        logger.warning(f"⚠️ Index not found at {INDEX_PATH}")
        raise FileNotFoundError("No index found. Run scripts/build_index.py first.")

    mtime = os.path.getmtime(INDEX_PATH)
    if _agent is not None and mtime == _index_mtime:
        return _agent

    try:
        from rag.agent import RAGAgent
        from rag.embedder import Embedder
        from rag.retriever import Retriever
        from rag.vector_store import VectorStore

        if _embedder is None:
            logger.info("📥 Loading embedder...")
            _embedder = Embedder()

        logger.info("📥 Loading vector store...")
        store = VectorStore.load(INDEX_PATH)
        _agent = RAGAgent(Retriever(_embedder, store))
        _index_mtime = mtime
        logger.info("✅ Agent ready (index reloaded)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}", exc_info=True)
        raise
    return _agent


def _wait_for_mount(name: str, deleted: bool, timeout: float = 20.0) -> bool:
    """The bucket mount can lag a few seconds behind the event. Wait until it shows the change."""
    path = os.path.join(os.path.dirname(RAW_DIR), name)  # name looks like "raw/file.pdf"
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path) != deleted:
            return True
        time.sleep(1)
    return False


def create_app() -> FastAPI:
    app = FastAPI(title="HunterAI RAG API", version="1.0.0")

    logger.info("🚀 Initializing FastAPI app...")

    # The UI is served from this same origin, so CORS is off unless other origins are listed explicitly.
    cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize auth DB if configured
    if settings.db_url:
        logger.info("📦 Initializing database...")
        try:
            from services.auth import init_db
            init_db(settings.db_url)
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.warning(f"⚠️ Database initialization failed (continuing): {e}")

    if not google_auth.enabled() and not settings.db_url:
        logger.warning("⚠️ Neither GOOGLE_CLIENT_ID nor RAG_DB_URL is set: the API is open to anyone.")

    def _api_token(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
        if authorization:
            for prefix in ("ApiKey ", "Bearer "):
                if authorization.startswith(prefix):
                    return authorization[len(prefix):].strip() or None
        return x_api_key

    def require_auth(
        request: Request,
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None),
    ):
        """Allow a valid API key or a signed-in Google user. Auth is off only when neither is configured."""
        token = _api_token(authorization, x_api_key)
        if token:
            if not settings.db_url:
                raise HTTPException(status_code=401, detail="API keys are not enabled on this server.")
            from services.auth import verify_api_key as verify_token
            ws = verify_token(token)
            if not ws:
                raise HTTPException(status_code=401, detail="Invalid API key")
            return ws

        user = google_auth.user_from_request(request)
        if user:
            return user

        if google_auth.enabled() or settings.db_url:
            raise HTTPException(status_code=401, detail="Sign in required")
        return None

    require_user = require_auth    # ask / list documents
    require_writer = require_auth  # upload / delete / reindex / jobs

    @app.get("/api/config")
    async def public_config():
        return {
            "googleClientId": google_auth.client_id(), 
            "loginRequired": google_auth.enabled()
        }

    @app.post("/api/auth/google")
    async def auth_google(body: GoogleLogin, response: Response):
        if not google_auth.enabled():
            raise HTTPException(status_code=400, detail="Google login is not configured.")
        try:
            user = await run_in_threadpool(google_auth.verify_google_credential, body.credential)
        except Exception as e:
            logger.warning(f"Google token rejected: {e}")
            raise HTTPException(status_code=401, detail="Google sign-in failed.")
        if not google_auth.is_allowed(user["email"]):
            raise HTTPException(status_code=403, detail=f"{user['email']} is not allowed to use this app.")
        google_auth.start_session(response, user)
        return {"user": user}

    @app.get("/api/auth/me")
    async def auth_me(request: Request):
        user = google_auth.user_from_request(request)
        if google_auth.enabled() and not user:
            raise HTTPException(status_code=401, detail="Not signed in")
        return {"user": user}

    @app.post("/api/auth/logout")
    async def auth_logout(response: Response):
        google_auth.end_session(response)
        return {"status": "ok"}

    @app.on_event("startup")
    async def startup_event():
        """Log startup completion - don't load agent here."""
        logger.info("🎉 FastAPI startup complete - agent will load on first request")

    # ---------- RAG & API Routes ----------
    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest, user=Depends(require_user)):
        question = request.question.strip()
        history = request.history or []
        print("question: ", question)
        if not question:
            raise HTTPException(status_code=400, detail="No question provided.")

        if not os.path.exists(INDEX_PATH):
            raise HTTPException(status_code=400, detail="No index found. Upload a document first.")

        try:
            answer = get_agent().answer(question, history=history)
            return AskResponse(answer=answer)
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"The assistant could not answer that question. {str(e)}")
        print("answer:" )

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            index_exists=os.path.exists(INDEX_PATH),
            index_updated_at=os.path.getmtime(INDEX_PATH) if os.path.exists(INDEX_PATH) else None,
            app_name=settings.app_name,
            app_version=settings.app_version,
            documents_directory=RAW_DIR,
        )

    @app.get("/api/documents", response_model=DocumentsResponse)
    async def documents(user=Depends(require_user)):
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
    async def upload_documents(files: list[UploadFile] = File(...), user=Depends(require_writer)):
        """Save uploaded files into the documents folder (the bucket) and index them."""
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

            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{original_name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            with open(os.path.join(RAW_DIR, original_name), "wb") as f:
                f.write(content)
            saved_names.append(original_name)

        if not saved_names:
            raise HTTPException(status_code=400, detail="No valid files were saved.")

        who = (user or {}).get("email", "api-key") if isinstance(user, dict) else "api-key"
        logger.info(f"📄 {who} uploaded {saved_names}")

        if not INDEX_ON_UPLOAD:
            return UploadResponse(status="saved", documents=saved_names)

        try:
            from scripts.build_index import build_index
            await run_in_threadpool(build_index)
        except Exception as e:
            logger.error(f"❌ Indexing after upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Files saved, but indexing failed. {str(e)}")
        return UploadResponse(status="indexed", documents=saved_names)

    @app.delete("/api/documents/{name}")
    async def delete_document(name: str, user=Depends(require_writer)):
        safe = secure_filename(name)
        path = os.path.join(RAW_DIR, safe)
        if not safe or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Document not found.")
        os.remove(path)
        if INDEX_ON_UPLOAD:
            try:
                from scripts.build_index import build_index
                await run_in_threadpool(build_index)
            except Exception as e:
                logger.error(f"❌ Reindex after delete failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"File deleted, but reindexing failed. {str(e)}")
        return {"status": "deleted", "name": safe}

    @app.post("/api/reindex", response_model=ReindexResponse)
    async def reindex(user=Depends(require_writer)):
        try:
            from scripts.build_index import build_index
            result = await run_in_threadpool(build_index)
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
    async def list_all_jobs(user=Depends(require_writer)):
        jobs = list_jobs()
        return {"jobs": jobs}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str, user=Depends(require_writer)):
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"job": job}

    # ---------- Automatic Reindex when the Bucket Changes ----------
    @app.api_route("/internal/gcs-event/{token}", methods=["GET", "POST", "OPTIONS"])
    async def gcs_event(
        token: str,
        request: Request,
        payload: Optional[dict] = Body(None),
        ce_subject: Optional[str] = Header(None),
        ce_type: Optional[str] = Header(None),
    ):
        if request.method == "OPTIONS":
            return {"status": "ok"}

        global _index_running, _index_dirty

        expected = os.getenv("EVENT_TOKEN", "")
        if not expected or not secrets.compare_digest(token, expected):
            raise HTTPException(status_code=403, detail="Forbidden")

        name = (payload or {}).get("name") or (ce_subject or "").removeprefix("objects/")
        ext = os.path.splitext(name)[1].lower()
        if not name.startswith("raw/") or ext not in ALLOWED_UPLOAD_EXTENSIONS:
            return {"status": "ignored", "object": name}

        logger.info(f"📬 Bucket event {ce_type or '?'} for {name}")

        with _index_state_lock:
            if _index_running:
                _index_dirty = True
                return {"status": "coalesced", "object": name}
            _index_running = True

        try:
            from scripts.build_index import build_index

            while True:
                with _index_state_lock:
                    _index_dirty = False
                result = await run_in_threadpool(build_index)
                logger.info(f"📚 Reindexed after change to {name}: {result}")
                with _index_state_lock:
                    if not _index_dirty:
                        break
        except Exception as e:
            logger.error(f"❌ Event-driven reindex failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Reindex failed")
        finally:
            with _index_state_lock:
                _index_running = False

        return {
            "status": "indexed",
            "documents": result.get("documents"),
            "chunks": result.get("chunks"),
        }

    # ---------- Static & Single Page Application (React) Fallback ----------
    if os.path.exists("static"):
        # Mount assets folder if present inside static/
        if os.path.exists("static/assets"):
            app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

        @app.get("/{full_path:path}")
        async def catch_all(full_path: str):
            # Do not intercept /api or /internal paths
            if full_path.startswith("api/") or full_path.startswith("internal/"):
                raise HTTPException(status_code=404, detail="API route not found")

            # Check if specific file exists in static folder (e.g., favicon.ico)
            target_file = os.path.join("static", full_path)
            if full_path and os.path.isfile(target_file):
                return FileResponse(target_file)

            # Default SPA fallback to index.html
            return FileResponse("static/index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)