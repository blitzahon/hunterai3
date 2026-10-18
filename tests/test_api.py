from fastapi.testclient import TestClient

from api_fastapi import app


def test_health_endpoint_reports_runtime_state():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "index_exists" in payload


def test_document_upload_endpoint_accepts_files(tmp_path, monkeypatch):
    monkeypatch.setattr("api_fastapi.RAW_DIR", str(tmp_path))
    monkeypatch.setattr("api_fastapi.enqueue_job", lambda *_: "test-job")

    with TestClient(app) as client:
        response = client.post(
            "/api/documents/upload",
            files={"files": ("sample.txt", b"sample knowledge text", "text/plain")},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "queued"
    assert payload["documents"] == ["sample.txt"]
