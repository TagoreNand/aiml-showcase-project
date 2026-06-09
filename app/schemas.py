from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=5, description="Raw document text")


class IngestResponse(BaseModel):
    message: str
    corpus_size: int
    tenant_id: str = "public"


class FileIngestResponse(BaseModel):
    message: str
    document_id: str
    corpus_size: int
    tenant_id: str = "public"
    ocr_source: str
    ocr_used: bool
    chars_extracted: int
    notes: List[str] = []


class ClassificationResult(BaseModel):
    label: str
    confidence: float


class AnalysisResponse(BaseModel):
    document_id: str
    classification: ClassificationResult
    entities: Dict[str, Any]
    summary: str
    extraction_methods: List[str] = ["rules"]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    document_id: str
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    tenant_id: str = "public"


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    evidence: List[SearchResult]
    answer_backend: str = "extractive"


class FeedbackRequest(BaseModel):
    document_id: str
    predicted_label: Optional[str] = None
    corrected_label: Optional[str] = None
    corrected_entities: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class MetricsResponse(BaseModel):
    documents_indexed: int
    feedback_records: int
    available_labels: List[str]
    vector_backend: str = "memory"
    extraction_backend: str = "rules"
    tenants: List[str] = []


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: str


class MeResponse(BaseModel):
    username: str
    role: str
    tenant_id: str


class RetrainResponse(BaseModel):
    message: str
    status: str
