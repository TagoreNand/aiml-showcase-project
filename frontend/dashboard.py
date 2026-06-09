from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuPilot AI Demo", page_icon="📄", layout="wide")
st.title("📄 DocuPilot AI")
st.caption("Document intelligence — classify · extract · retrieve · QA · monitor")

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.identity = None


def auth_headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


# --------------------------------------------------------------------------- #
# Sidebar: connection, auth, live config
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Connection")
    API_URL = st.text_input("API URL", value=API_URL)

    st.header("Authentication")
    st.caption("Only needed when AUTH_ENABLED=true. Demo: admin/analyst/viewer (e.g. analyst123).")
    with st.form("login"):
        u = st.text_input("Username", value="analyst")
        p = st.text_input("Password", value="analyst123", type="password")
        if st.form_submit_button("Log in"):
            try:
                r = requests.post(f"{API_URL}/auth/token", data={"username": u, "password": p}, timeout=15)
                if r.status_code == 200:
                    body = r.json()
                    st.session_state.token = body["access_token"]
                    st.session_state.identity = f"{u} ({body['role']} @ {body['tenant_id']})"
                    st.success(f"Logged in as {st.session_state.identity}")
                else:
                    st.error(f"Login failed ({r.status_code})")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Login error: {exc}")
    if st.session_state.identity:
        st.info(f"Signed in: {st.session_state.identity}")
        if st.button("Log out"):
            st.session_state.token = None
            st.session_state.identity = None

    st.header("Active config")
    try:
        cfg = requests.get(f"{API_URL}/config", timeout=10).json()
        st.json(cfg)
    except Exception:  # noqa: BLE001
        st.caption("API not reachable yet.")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Analyze", "Ingest", "Search", "Ask", "Monitoring"]
)

with tab1:
    st.subheader("Analyze a document")
    doc_id = st.text_input("Document ID", value="demo_doc_001", key="analyze_id")
    text = st.text_area(
        "Document text", height=200,
        value="Invoice INV-1001 from Acme Cloud for $1499 due on 2026-04-10. Please process payment.",
        key="analyze_text",
    )
    if st.button("Run analysis"):
        r = requests.post(f"{API_URL}/analyze", json={"document_id": doc_id, "text": text},
                          headers=auth_headers(), timeout=30)
        st.json(r.json())

with tab2:
    st.subheader("Ingest text")
    ingest_id = st.text_input("Document ID ", value="demo_doc_002", key="ingest_id")
    ingest_text = st.text_area(
        "Text to ingest", height=140,
        value="Support ticket priority high. Customer reports login failures after deployment.",
        key="ingest_text",
    )
    if st.button("Ingest document"):
        r = requests.post(f"{API_URL}/ingest", json={"document_id": ingest_id, "text": ingest_text},
                          headers=auth_headers(), timeout=30)
        st.json(r.json())

    st.divider()
    st.subheader("Ingest a file (PDF / image / text · OCR)")
    file_id = st.text_input("Document ID", value="demo_file_001", key="file_id")
    up = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg", "txt", "md", "csv"])
    if st.button("Ingest file") and up is not None:
        files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
        r = requests.post(f"{API_URL}/ingest/file", params={"document_id": file_id},
                          files=files, headers=auth_headers(), timeout=60)
        st.json(r.json())

with tab3:
    st.subheader("Search corpus")
    query = st.text_input("Search query", value="high priority login issue")
    top_k = st.slider("Top K results", 1, 10, 3)
    if st.button("Search"):
        r = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k},
                          headers=auth_headers(), timeout=30)
        st.json(r.json())

with tab4:
    st.subheader("Ask a question")
    question = st.text_input("Question", value="What issue was reported after deployment?")
    top_k_ask = st.slider("Top K evidence", 1, 10, 3, key="ask_top_k")
    if st.button("Ask"):
        r = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": top_k_ask},
                          headers=auth_headers(), timeout=30)
        st.json(r.json())

with tab5:
    st.subheader("Monitoring & MLOps")
    c1, c2 = st.columns(2)
    if c1.button("Service metrics"):
        c1.json(requests.get(f"{API_URL}/metrics", headers=auth_headers(), timeout=15).json())
    if c2.button("Drift report"):
        c2.json(requests.get(f"{API_URL}/monitoring/drift", headers=auth_headers(), timeout=15).json())
    st.caption("Prometheus metrics exposed at /metrics/prometheus · audit log at /audit (admin).")
