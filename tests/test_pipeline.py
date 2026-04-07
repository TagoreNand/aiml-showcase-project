from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze_endpoint():
    response = client.post(
        "/analyze",
        json={
            "document_id": "test_doc_001",
            "text": "Invoice INV-9001 from Acme Cloud for $999 due on 2026-04-30."
        }
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["label"] in {"invoice", "contract", "resume", "support_ticket"}
    assert "summary" in payload


def test_ingest_and_search():
    ingest = client.post(
        "/ingest",
        json={
            "document_id": "test_search_doc",
            "text": "Priority critical. Search latency spiked after deployment for enterprise users."
        }
    )
    assert ingest.status_code == 200

    search = client.post("/search", json={"query": "critical search latency", "top_k": 2})
    assert search.status_code == 200
    results = search.json()["results"]
    assert len(results) >= 1
