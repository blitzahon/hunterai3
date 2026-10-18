import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Hunter AI RAG"
    app_version: str = "0.1.0"
    host: str = os.getenv("RAG_HOST", "127.0.0.1")
    port: int = int(os.getenv("RAG_PORT", "5000"))
    debug: bool = os.getenv("RAG_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    index_path: str = os.getenv("RAG_INDEX_PATH", "data/index/store.pkl")
    db_url: str | None = os.getenv("RAG_DB_URL")


settings = Settings()
