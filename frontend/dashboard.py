from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuPilot AI Demo", page_icon="📄", layout="wide")
st.title("📄 DocuPilot AI")
st.caption("Interview-ready document intelligence demo")

tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Ingest", "Search", "Ask"])

with tab1:
    st.subheader("Analyze a document")
    doc_id = st.text_input("Document ID", value="demo_doc_001", key="analyze_id")
    text = st.text_area(
        "Document text",
        height=220,
        value="Invoice INV-1001 from Acme Cloud for $1499 due on 2026-04-10. Please process payment.",
        key="analyze_text"
    )
    if st.button("Run analysis"):
        response = requests.post(f"{API_URL}/analyze", json={"document_id": doc_id, "text": text}, timeout=30)
        st.json(response.json())

with tab2:
    st.subheader("Ingest into searchable corpus")
    ingest_id = st.text_input("Document ID ", value="demo_doc_002", key="ingest_id")
    ingest_text = st.text_area(
        "Text to ingest",
        height=180,
        value="Support ticket priority high. Customer reports login failures after deployment.",
        key="ingest_text"
    )
    if st.button("Ingest document"):
        response = requests.post(f"{API_URL}/ingest", json={"document_id": ingest_id, "text": ingest_text}, timeout=30)
        st.json(response.json())

with tab3:
    st.subheader("Search corpus")
    query = st.text_input("Search query", value="high priority login issue")
    top_k = st.slider("Top K results", 1, 10, 3)
    if st.button("Search"):
        response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k}, timeout=30)
        st.json(response.json())

with tab4:
    st.subheader("Ask a question")
    question = st.text_input("Question", value="What issue was reported after deployment?")
    top_k_ask = st.slider("Top K evidence", 1, 10, 3, key="ask_top_k")
    if st.button("Ask"):
        response = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": top_k_ask}, timeout=30)
        st.json(response.json())
