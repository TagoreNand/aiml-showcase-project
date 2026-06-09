"""Vector-store factory with graceful backend fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.services.vectorstore.base import StoredDocument, VectorStore
from app.services.vectorstore.memory import InMemoryVectorStore

logger = logging.getLogger(__name__)

__all__ = ["VectorStore", "StoredDocument", "InMemoryVectorStore", "build_vector_store"]


def build_vector_store(corpus_path: Optional[Path] = None) -> VectorStore:
    """Return the configured vector store, falling back to in-memory on error."""
    corpus_path = corpus_path or Settings.CORPUS_PATH

    if Settings.VECTOR_BACKEND in {"pgvector", "postgres", "pg"}:
        try:
            from app.services.vectorstore.pgvector import PgVectorStore

            return PgVectorStore(Settings.DATABASE_URL)
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            logger.warning(
                "pgvector backend unavailable (%s); falling back to in-memory store",
                exc,
            )

    return InMemoryVectorStore(corpus_path)
