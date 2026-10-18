"""Authentication utilities: API keys and workspace model using SQLAlchemy.

This module connects to a DB URL (Postgres recommended) and provides a
minimal API key issuance and verification flow. For development where
no DB_URL is configured, authentication is a no-op (disabled).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

from werkzeug.security import generate_password_hash, check_password_hash

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("RAG_DB_URL")  # e.g. postgres://user:pass@host:5432/db

Base = declarative_base()
_engine = None
_SessionLocal = None


class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text("now()"))
    api_keys = relationship("ApiKey", back_populates="workspace")


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_id = Column(String(48), unique=True, nullable=False)
    secret_hash = Column(String(256), nullable=False)
    name = Column(String(200), nullable=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"), nullable=False)
    disabled = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, server_default=text("now()"))

    workspace = relationship("Workspace", back_populates="api_keys")


def init_db(url: Optional[str] = None):
    """Initialize DB connection and create tables if needed.

    If DATABASE_URL is not set, this function is a no-op and returns False.
    """
    global _engine, _SessionLocal
    db_url = url or DATABASE_URL
    if not db_url:
        return False

    # create engine
    _engine = create_engine(db_url, future=True)
    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(bind=_engine)
    return True


def _get_session():
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized. Call init_db() with a DATABASE_URL")
    return _SessionLocal()


def create_workspace(name: str) -> Workspace:
    """Create a workspace by name or return existing."""
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized")
    s = _get_session()
    try:
        ws = s.query(Workspace).filter(Workspace.name == name).first()
        if ws:
            return ws
        ws = Workspace(name=name)
        s.add(ws)
        s.commit()
        s.refresh(ws)
        return ws
    finally:
        s.close()


def create_api_key(workspace_name: str, key_name: str = "default") -> str:
    """Create an API key for a workspace and return the plaintext token.

    Token format: <key_id>.<secret>
    """
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized")
    s = _get_session()
    try:
        ws = s.query(Workspace).filter(Workspace.name == workspace_name).first()
        if not ws:
            ws = Workspace(name=workspace_name)
            s.add(ws)
            s.commit()
            s.refresh(ws)

        key_id = uuid.uuid4().hex[:24]
        secret = uuid.uuid4().hex + uuid.uuid4().hex
        secret_hash = generate_password_hash(secret)

        api_key = ApiKey(key_id=key_id, secret_hash=secret_hash, name=key_name, workspace_id=ws.id)
        s.add(api_key)
        s.commit()
        token = f"{key_id}.{secret}"
        return token
    finally:
        s.close()


def verify_api_key(token: str) -> Optional[Workspace]:
    """Verify a token and return the associated Workspace or None.

    Token must be in format key_id.secret
    """
    if not token or _SessionLocal is None:
        return None
    parts = token.split(".")
    if len(parts) < 2:
        return None
    key_id = parts[0]
    secret = ".".join(parts[1:])
    s = _get_session()
    try:
        api_key = s.query(ApiKey).filter(ApiKey.key_id == key_id, ApiKey.disabled == False).first()
        if not api_key:
            return None
        if not check_password_hash(api_key.secret_hash, secret):
            return None
        ws = s.query(Workspace).filter(Workspace.id == api_key.workspace_id).first()
        return ws
    finally:
        s.close()
