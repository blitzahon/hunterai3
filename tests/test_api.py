import pytest
from fastapi.testclient import TestClient

from api_fastapi import app
from services import google_auth

USER = {"email": "alice@example.com", "name": "Alice", "picture": None}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("api_fastapi.RAW_DIR", str(tmp_path))
    monkeypatch.setattr("scripts.build_index.build_index", lambda *a, **k: {"documents": 1, "chunks": 1})
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def login_on(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setattr(google_auth, "verify_google_credential", lambda cred: dict(USER))


def _upload(client):
    return client.post(
        "/api/documents/upload",
        files={"files": ("sample.txt", b"sample knowledge text", "text/plain")},
    )


def test_health_endpoint_reports_runtime_state(client):
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "index_exists" in payload


def test_document_upload_endpoint_accepts_files_when_login_is_off(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    response = _upload(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "indexed"
    assert payload["documents"] == ["sample.txt"]


def test_api_requires_sign_in_when_login_is_on(client, login_on):
    assert client.get("/api/documents").status_code == 401
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401
    assert _upload(client).status_code == 401
    assert client.delete("/api/documents/sample.txt").status_code == 401
    assert client.post("/api/reindex").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_google_sign_in_starts_a_session(client, login_on):
    response = client.post("/api/auth/google", json={"credential": "token"})
    assert response.status_code == 200
    assert response.json()["user"]["email"] == USER["email"]

    assert client.get("/api/auth/me").json()["user"]["email"] == USER["email"]
    assert client.get("/api/documents").status_code == 200
    assert _upload(client).status_code == 200

    client.post("/api/auth/logout")
    assert client.get("/api/documents").status_code == 401


def test_forged_session_cookie_is_rejected(client, login_on):
    client.cookies.set(google_auth.COOKIE_NAME, "alice@example.com|Alice")
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/documents").status_code == 401


def test_session_signed_with_another_secret_is_rejected(client, login_on, monkeypatch):
    client.post("/api/auth/google", json={"credential": "token"})
    monkeypatch.setenv("SESSION_SECRET", "rotated-secret")
    assert client.get("/api/documents").status_code == 401


def test_allowed_emails_is_checked_on_sign_in_and_on_every_request(client, login_on, monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAILS", "bob@example.com")
    assert client.post("/api/auth/google", json={"credential": "token"}).status_code == 403

    monkeypatch.setenv("ALLOWED_EMAILS", "@example.com")
    assert client.post("/api/auth/google", json={"credential": "token"}).status_code == 200
    assert client.get("/api/documents").status_code == 200

    monkeypatch.setenv("ALLOWED_EMAILS", "bob@example.com")
    assert client.get("/api/documents").status_code == 401


def test_api_key_header_is_rejected_when_api_keys_are_not_enabled(client, login_on):
    response = client.get("/api/documents", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 401


def test_cors_is_off_by_default(client):
    response = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_ask_returns_answer_with_token_usage(client, monkeypatch, tmp_path):
    usage = {"input_tokens": 120, "output_tokens": 30, "thinking_tokens": 10, "total_tokens": 160, "model_calls": 2}

    class FakeAgent:
        def answer_with_usage(self, *a, **k):
            return "hello", usage

    index = tmp_path / "store.pkl"
    index.write_bytes(b"")
    monkeypatch.setattr("api_fastapi.INDEX_PATH", str(index))
    monkeypatch.setattr("api_fastapi.get_agent", lambda: FakeAgent())

    response = client.post("/api/ask", json={"question": "hi"})
    assert response.status_code == 200
    assert response.json() == {"answer": "hello", "usage": usage}


def test_ask_returns_503_when_the_model_is_rate_limited(client, monkeypatch, tmp_path):
    from rag.agent import ModelBusyError

    class BusyAgent:
        def answer_with_usage(self, *a, **k):
            raise ModelBusyError("429 RESOURCE_EXHAUSTED")

    index = tmp_path / "store.pkl"
    index.write_bytes(b"")
    monkeypatch.setattr("api_fastapi.INDEX_PATH", str(index))
    monkeypatch.setattr("api_fastapi.get_agent", lambda: BusyAgent())

    response = client.post("/api/ask", json={"question": "hi"})
    assert response.status_code == 503
    assert "busy" in response.json()["detail"]
    assert response.headers["retry-after"] == "10"
