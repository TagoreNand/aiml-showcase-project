from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=5, description="Raw document text")


class IngestResponse(BaseModel):
    message: str
    corpus_size: int


class ClassificationResult(BaseModel):
    label: str
    confidence: float


class AnalysisResponse(BaseModel):
    document_id: str
    classification: ClassificationResult
    entities: Dict[str, Any]
    summary: str


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


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    evidence: List[SearchResult]


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
