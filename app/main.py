from __future__ import annotations

from fastapi import FastAPI

from app.config import Settings
from app.schemas import (
    AnalysisResponse,
    AskRequest,
    AskResponse,
    DocumentIn,
    FeedbackRequest,
    IngestResponse,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.classifier import DocumentClassifier
from app.services.feedback import FeedbackStore
from app.services.pipeline import analyze_document, answer_question
from app.services.retrieval import RetrievalEngine
from app.ml.train_classifier import train


Settings.ensure_dirs()
MODEL_PATH = Settings.MODEL_DIR / "document_classifier.joblib"

if not MODEL_PATH.exists():
    train()

classifier = DocumentClassifier(MODEL_PATH)
retrieval_engine = RetrievalEngine(Settings.CORPUS_PATH)
feedback_store = FeedbackStore(Settings.FEEDBACK_PATH)

app = FastAPI(
    title=Settings.APP_NAME,
    version="1.0.0",
    description="Document intelligence and workflow automation platform."
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": Settings.APP_ENV,
        "app": Settings.APP_NAME,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(payload: DocumentIn):
    retrieval_engine.add_document(payload.document_id, payload.text)
    return IngestResponse(message="Document ingested successfully", corpus_size=retrieval_engine.count())


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: DocumentIn):
    result = analyze_document(payload.text, classifier)
    return AnalysisResponse(
        document_id=payload.document_id,
        classification=result["classification"],
        entities=result["entities"],
        summary=result["summary"],
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    results = retrieval_engine.search(payload.query, payload.top_k)
    return SearchResponse(
        query=payload.query,
        results=[SearchResult(**row) for row in results],
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    result = answer_question(payload.question, retrieval_engine, payload.top_k)
    evidence = [SearchResult(**row) for row in result["evidence"]]
    return AskResponse(question=payload.question, answer=result["answer"], evidence=evidence)


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    feedback_store.save(payload.model_dump())
    return {"message": "Feedback stored successfully"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(
        documents_indexed=retrieval_engine.count(),
        feedback_records=feedback_store.count(),
        available_labels=classifier.labels(),
    )
