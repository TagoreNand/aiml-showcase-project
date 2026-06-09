"""Central application configuration.

Backward compatible with the original ``Settings`` class (same attributes and
``ensure_dirs`` classmethod) while adding feature flags for the production
extensions: vector store backend, auth/RBAC, OCR, LLM/NER extraction, MLflow,
Prometheus and drift monitoring.

Everything is environment driven and *local-first*: every advanced integration
is optional and degrades gracefully to the original behaviour when the backing
service or dependency is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    # ---- core app ----------------------------------------------------------
    APP_NAME = os.getenv("APP_NAME", "DocuPilot AI")
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
    BASE_DIR = Path(__file__).resolve().parents[1]

    # ---- data paths (unchanged defaults) -----------------------------------
    MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "data/models")
    CORPUS_PATH = BASE_DIR / os.getenv("CORPUS_PATH", "data/processed/corpus.json")
    FEEDBACK_PATH = BASE_DIR / os.getenv("FEEDBACK_PATH", "data/processed/feedback.jsonl")
    TRAINING_DATA_PATH = BASE_DIR / os.getenv("TRAINING_DATA_PATH", "data/raw/training_documents.json")
    AUDIT_LOG_PATH = BASE_DIR / os.getenv("AUDIT_LOG_PATH", "data/processed/audit.jsonl")
    METRICS_DIR = BASE_DIR / os.getenv("METRICS_DIR", "data/processed/metrics")

    # ---- multi-tenancy -----------------------------------------------------
    DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "public")

    # ---- vector store ------------------------------------------------------
    # backend: "memory" (default, no external service) or "pgvector"
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "memory").strip().lower()
    # embedder: "tfidf" (default, pure sklearn) or "sentence-transformers"
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf").strip().lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 384)

    # Postgres / pgvector connection (only used when VECTOR_BACKEND=pgvector)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://docupilot:docupilot@localhost:5432/docupilot",
    )

    # ---- OCR ---------------------------------------------------------------
    OCR_ENABLED = _env_bool("OCR_ENABLED", True)  # auto-falls back if libs absent
    OCR_LANG = os.getenv("OCR_LANG", "eng")
    OCR_DPI = _env_int("OCR_DPI", 200)

    # ---- extraction (NER / LLM) -------------------------------------------
    # extraction strategy: "rules" (default), "ner", or "llm".
    # "ner"/"llm" fall back to "rules" when their deps/keys are missing.
    EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "rules").strip().lower()
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # LLM provider: "none" (default), "openai", "anthropic"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # ---- auth / RBAC -------------------------------------------------------
    AUTH_ENABLED = _env_bool("AUTH_ENABLED", False)  # off by default for the demo
    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = _env_int("JWT_EXPIRE_MINUTES", 60)
    # Built-in demo users (used when no external user store is configured).
    # Format: "username:password:role:tenant" comma separated.
    DEMO_USERS = os.getenv(
        "DEMO_USERS",
        "admin:admin123:admin:public,"
        "analyst:analyst123:analyst:public,"
        "viewer:viewer123:viewer:public",
    )

    # ---- MLflow ------------------------------------------------------------
    MLFLOW_ENABLED = _env_bool("MLFLOW_ENABLED", False)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "docupilot-classifier")

    # ---- observability -----------------------------------------------------
    PROMETHEUS_ENABLED = _env_bool("PROMETHEUS_ENABLED", True)  # no-op if lib absent
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))  # PSI alert level

    # ---- async workflows ---------------------------------------------------
    # "background" (FastAPI BackgroundTasks, default) or "celery"
    TASK_BACKEND = os.getenv("TASK_BACKEND", "background").strip().lower()
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def as_dict(cls) -> dict:
        """Non-secret config snapshot, useful for /health and debugging."""
        return {
            "app_name": cls.APP_NAME,
            "app_env": cls.APP_ENV,
            "app_version": cls.APP_VERSION,
            "vector_backend": cls.VECTOR_BACKEND,
            "embedding_backend": cls.EMBEDDING_BACKEND,
            "extraction_backend": cls.EXTRACTION_BACKEND,
            "llm_provider": cls.LLM_PROVIDER,
            "ocr_enabled": cls.OCR_ENABLED,
            "auth_enabled": cls.AUTH_ENABLED,
            "mlflow_enabled": cls.MLFLOW_ENABLED,
            "prometheus_enabled": cls.PROMETHEUS_ENABLED,
            "task_backend": cls.TASK_BACKEND,
        }


settings = Settings()
