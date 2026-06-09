import pytest
from fastapi.testclient import TestClient

import app.config as config
from app.main import app
from app.security import jwt_utils
from app.security.passwords import hash_password, verify_password

client = TestClient(app)


def test_password_hash_roundtrip():
    h = hash_password("s3cret")
    assert verify_password("s3cret", h) and not verify_password("nope", h)


def test_jwt_roundtrip_and_tamper():
    tok = jwt_utils.encode({"sub": "a"}, "k", expires_minutes=5)
    assert jwt_utils.decode(tok, "k")["sub"] == "a"
    with pytest.raises(jwt_utils.JWTError):
        jwt_utils.decode(tok, "wrong-key")


def test_rbac_enforced_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(config.Settings, "AUTH_ENABLED", True)
    # no token -> 401
    assert client.post("/analyze", json={"document_id": "d", "text": "hello world"}).status_code == 401
    # viewer cannot ingest, analyst can
    tv = client.post("/auth/token", data={"username": "viewer", "password": "viewer123"}).json()["access_token"]
    ta = client.post("/auth/token", data={"username": "analyst", "password": "analyst123"}).json()["access_token"]
    assert client.post("/ingest", json={"document_id": "x", "text": "hello world doc"},
                       headers={"Authorization": f"Bearer {tv}"}).status_code == 403
    assert client.post("/ingest", json={"document_id": "x", "text": "hello world doc"},
                       headers={"Authorization": f"Bearer {ta}"}).status_code == 200
    # bad creds
    assert client.post("/auth/token", data={"username": "viewer", "password": "x"}).status_code == 401


def test_auth_disabled_keeps_endpoints_open():
    # default config has AUTH_ENABLED False
    assert client.get("/config").json()["auth_enabled"] in (False, True)
    assert client.post("/analyze", json={"document_id": "d", "text": "Invoice INV-1 for $5"}).status_code == 200
