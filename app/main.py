from __future__ import annotations

import logging

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.security import OAuth2PasswordRequestForm

from app.config import Settings
from app.schemas import (
    AnalysisResponse,
    AskRequest,
    AskResponse,
    DocumentIn,
    FeedbackRequest,
    FileIngestResponse,
    IngestResponse,
    MeResponse,
    MetricsResponse,
    RetrainResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TokenResponse,
)
from app.security.auth import authenticate, create_access_token, get_current_principal, require_role
from app.security.users import Principal
from app.services.audit import AuditLogger
from app.services.classifier import DocumentClassifier
from app.services.feedback import FeedbackStore
from app.services.ocr import extract_text
from app.services.pipeline import analyze_document, answer_question
from app.services.retrieval import RetrievalEngine
from app.services.tasks import get_task_runner
from app.observability import metrics
from app.observability.monitoring import get_monitor
from app.ml.train_classifier import train

logger = logging.getLogger(__name__)

Settings.ensure_dirs()
MODEL_PATH = Settings.MODEL_DIR / "document_classifier.joblib"
if not MODEL_PATH.exists():
    train()

classifier = DocumentClassifier(MODEL_PATH)
retrieval_engine = RetrievalEngine(Settings.CORPUS_PATH)
feedback_store = FeedbackStore(Settings.FEEDBACK_PATH)
audit = AuditLogger(Settings.AUDIT_LOG_PATH)
monitor = get_monitor()
tasks = get_task_runner()

app = FastAPI(
    title=Settings.APP_NAME,
    version=Settings.APP_VERSION,
    description="Document intelligence and workflow automation platform.",
)


@app.middleware("http")
async def _observe(request, call_next):
    endpoint = request.url.path
    with metrics.observe_latency(endpoint):
        response = await call_next(request)
    metrics.record_request(endpoint, status=str(response.status_code))
    return response


def _post_analyze(label: str, confidence: float, text: str, actor: str, tenant: str) -> None:
    """Async side-effects: drift tracking, metrics, audit."""
    monitor.record_prediction(label, text)
    metrics.record_prediction(label, confidence)
    audit.record("analyze", actor=actor, tenant_id=tenant, details={"label": label})


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    principal = authenticate(form_data.username, form_data.password)
    if principal is None:
        audit.record("login", actor=form_data.username, status="denied")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    audit.record("login", actor=principal.username, tenant_id=principal.tenant_id)
    return TokenResponse(
        access_token=create_access_token(principal),
        role=principal.role,
        tenant_id=principal.tenant_id,
    )


@app.get("/auth/me", response_model=MeResponse, tags=["auth"])
def me(principal: Principal = Depends(get_current_principal)):
    return MeResponse(username=principal.username, role=principal.role, tenant_id=principal.tenant_id)


# --------------------------------------------------------------------------- #
# Health / config / metrics
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "environment": Settings.APP_ENV, "app": Settings.APP_NAME}


@app.get("/config")
def config():
    return Settings.as_dict()


@app.get("/metrics", response_model=MetricsResponse)
def metrics_summary(principal: Principal = Depends(require_role("viewer"))):
    metrics.set_corpus_size(retrieval_engine.count())
    return MetricsResponse(
        documents_indexed=retrieval_engine.count(),
        feedback_records=feedback_store.count(),
        available_labels=classifier.labels(),
        vector_backend=retrieval_engine.backend,
        extraction_backend=Settings.EXTRACTION_BACKEND,
        tenants=retrieval_engine.tenants(),
    )


@app.get("/metrics/prometheus")
def prometheus_metrics():
    metrics.set_corpus_size(retrieval_engine.count())
    body, content_type = metrics.render_latest()
    return Response(content=body, media_type=content_type)


@app.get("/monitoring/drift")
def drift(principal: Principal = Depends(require_role("viewer"))):
    return monitor.drift_report()


@app.get("/audit")
def audit_log(limit: int = 50, principal: Principal = Depends(require_role("admin"))):
    return {"entries": audit.tail(limit=limit)}


# --------------------------------------------------------------------------- #
# Core document intelligence
# --------------------------------------------------------------------------- #
@app.post("/ingest", response_model=IngestResponse)
def ingest_document(
    payload: DocumentIn,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(require_role("analyst")),
):
    retrieval_engine.add_document(payload.document_id, payload.text, tenant_id=principal.tenant_id)
    tasks.submit(background_tasks, audit.record, "ingest",
                 actor=principal.username, tenant_id=principal.tenant_id,
                 resource=payload.document_id)
    return IngestResponse(
        message="Document ingested successfully",
        corpus_size=retrieval_engine.count(principal.tenant_id),
        tenant_id=principal.tenant_id,
    )


@app.post("/ingest/file", response_model=FileIngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    document_id: str,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_role("analyst")),
):
    raw = await file.read()
    result = extract_text(raw, file.filename or document_id)
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"message": "No text could be extracted", "notes": result.notes},
        )
    retrieval_engine.add_document(
        document_id, result.text, tenant_id=principal.tenant_id,
        metadata={"source_file": file.filename, "ocr_source": result.source},
    )
    tasks.submit(background_tasks, audit.record, "ingest_file",
                 actor=principal.username, tenant_id=principal.tenant_id,
                 resource=document_id, details={"ocr_source": result.source})
    return FileIngestResponse(
        message="File ingested successfully",
        document_id=document_id,
        corpus_size=retrieval_engine.count(principal.tenant_id),
        tenant_id=principal.tenant_id,
        ocr_source=result.source,
        ocr_used=result.ocr_used,
        chars_extracted=len(result.text),
        notes=result.notes,
    )


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(
    payload: DocumentIn,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(require_role("analyst")),
):
    result = analyze_document(payload.text, classifier)
    cls = result["classification"]
    tasks.submit(background_tasks, _post_analyze, cls["label"], cls["confidence"],
                 payload.text, principal.username, principal.tenant_id)
    return AnalysisResponse(
        document_id=payload.document_id,
        classification=cls,
        entities=result["entities"],
        summary=result["summary"],
        extraction_methods=result["extraction_methods"],
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, principal: Principal = Depends(require_role("viewer"))):
    results = retrieval_engine.search(payload.query, payload.top_k, tenant_id=principal.tenant_id)
    return SearchResponse(
        query=payload.query,
        results=[SearchResult(**row) for row in results],
        tenant_id=principal.tenant_id,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, principal: Principal = Depends(require_role("viewer"))):
    result = answer_question(payload.question, retrieval_engine, payload.top_k,
                             tenant_id=principal.tenant_id)
    evidence = [SearchResult(**row) for row in result["evidence"]]
    return AskResponse(
        question=payload.question,
        answer=result["answer"],
        evidence=evidence,
        answer_backend=result.get("answer_backend", "extractive"),
    )


@app.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(require_role("analyst")),
):
    record = payload.model_dump()
    record["_actor"] = principal.username
    record["_tenant"] = principal.tenant_id
    feedback_store.save(record)
    tasks.submit(background_tasks, audit.record, "feedback",
                 actor=principal.username, tenant_id=principal.tenant_id,
                 resource=payload.document_id)
    return {"message": "Feedback stored successfully"}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, principal: Principal = Depends(require_role("admin"))):
    removed = retrieval_engine.delete_document(document_id, tenant_id=principal.tenant_id)
    audit.record("delete", actor=principal.username, tenant_id=principal.tenant_id,
                 resource=document_id, status="ok" if removed else "not_found")
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "document_id": document_id}


@app.post("/retrain", response_model=RetrainResponse)
def retrain(background_tasks: BackgroundTasks, principal: Principal = Depends(require_role("admin"))):
    def _retrain_and_reload():
        train()
        classifier.load()
        audit.record("retrain", actor=principal.username, tenant_id=principal.tenant_id)

    tasks.submit(background_tasks, _retrain_and_reload)
    return RetrainResponse(message="Retraining scheduled", status="accepted")
