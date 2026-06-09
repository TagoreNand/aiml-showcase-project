from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_config_and_health():
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/config").json()
    assert "vector_backend" in cfg and "extraction_backend" in cfg


def test_metrics_summary_reports_backend():
    body = client.get("/metrics").json()
    assert "vector_backend" in body and "tenants" in body


def test_drift_endpoint():
    r = client.get("/monitoring/drift").json()
    assert "psi" in r and "drift_detected" in r and "data_quality" in r


def test_prometheus_endpoint():
    assert client.get("/metrics/prometheus").status_code == 200


def test_file_ingest_text():
    files = {"file": ("note.txt", b"Resume: Jane Smith, ML engineer, jane@x.com", "text/plain")}
    r = client.post("/ingest/file?document_id=cv_1", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["ocr_source"] == "text" and body["chars_extracted"] > 0
