"""Simple background job queue for short-term async ingestion.

This is intentionally lightweight (file-backed JSON queue + worker thread)
so it works without external dependencies like Redis. It's suitable for a
single-process host (dev or simple container). For production use, swap
this out for RQ/Celery and Redis.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from scripts.build_index import build_index

JOBS_DIR = "data/jobs"
JOBS_FILE = os.path.join(JOBS_DIR, "jobs.json")
POLL_INTERVAL = 2.0

os.makedirs(JOBS_DIR, exist_ok=True)

_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _read_jobs() -> List[Dict[str, Any]]:
    if not os.path.exists(JOBS_FILE):
        return []
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_jobs(jobs: List[Dict[str, Any]]) -> None:
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


def _save_job(job: Dict[str, Any]) -> None:
    with _lock:
        jobs = _read_jobs()
        # replace existing or append
        for i, j in enumerate(jobs):
            if j.get("id") == job.get("id"):
                jobs[i] = job
                _write_jobs(jobs)
                return
        jobs.append(job)
        _write_jobs(jobs)


def enqueue_job(job_type: str, payload: Dict[str, Any]) -> str:
    """Create a new job and persist it to the job file."""
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": job_type,
        "payload": payload,
        "status": "pending",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "result": None,
        "error": None,
    }
    _save_job(job)
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    jobs = _read_jobs()
    for j in jobs:
        if j.get("id") == job_id:
            return j
    return None


def list_jobs() -> List[Dict[str, Any]]:
    return _read_jobs()


def _process_job(job: Dict[str, Any]) -> None:
    job["status"] = "running"
    job["updated_at"] = _now_iso()
    _save_job(job)

    try:
        if job["type"] == "index":
            # run the build_index script against default locations
            result = build_index()
            job["result"] = result
            job["status"] = "completed"
        else:
            job["error"] = f"Unknown job type: {job['type']}"
            job["status"] = "failed"
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "failed"
    finally:
        job["updated_at"] = _now_iso()
        _save_job(job)


def _worker_loop() -> None:
    while not _stop_event.is_set():
        # pick the oldest pending job
        jobs = _read_jobs()
        pending = [j for j in jobs if j.get("status") == "pending"]
        if pending:
            # FIFO
            job = pending[0]
            _process_job(job)
            # small pause to allow other requests
            time.sleep(0.1)
        else:
            time.sleep(POLL_INTERVAL)


def start_worker(detach: bool = True) -> threading.Thread:
    """Start the background worker thread (idempotent).

    Returns the Thread instance.
    """
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return _worker_thread

    _stop_event.clear()
    t = threading.Thread(target=_worker_loop, daemon=detach, name="job-worker")
    _worker_thread = t
    t.start()
    return t


def stop_worker() -> None:
    _stop_event.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=2.0)


# Start worker automatically when imported in app
start_worker()
