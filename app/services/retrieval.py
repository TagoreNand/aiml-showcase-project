"""Retrieval engine — a thin, backward-compatible facade over a VectorStore.

The original public API (``RetrievalEngine(corpus_path)`` with ``add_document``,
``search`` and ``count``) is preserved, so existing callers and tests keep
working. Under the hood it now delegates to a pluggable backend (in-memory
TF-IDF/dense, or Postgres + pgvector) selected by configuration, and adds
multi-tenant scoping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app.config import Settings
from app.services.vectorstore import VectorStore, build_vector_store


class RetrievalEngine:
    def __init__(self, corpus_path: Path, store: Optional[VectorStore] = None):
        self.corpus_path = corpus_path
        self.store: VectorStore = store or build_vector_store(corpus_path)

    @property
    def backend(self) -> str:
        return self.store.backend

    def add_document(
        self,
        document_id: str,
        text: str,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.store.add(
            document_id,
            text,
            tenant_id=tenant_id or Settings.DEFAULT_TENANT,
            metadata=metadata,
        )

    def delete_document(self, document_id: str, tenant_id: Optional[str] = None) -> bool:
        return self.store.delete(document_id, tenant_id or Settings.DEFAULT_TENANT)

    def search(
        self, query: str, top_k: int = 3, tenant_id: Optional[str] = None
    ) -> List[Dict[str, object]]:
        return self.store.search(
            query, top_k=top_k, tenant_id=tenant_id or Settings.DEFAULT_TENANT
        )

    def count(self, tenant_id: Optional[str] = None) -> int:
        # No tenant arg -> total across tenants (matches legacy global count).
        return self.store.count(tenant_id)

    def tenants(self) -> List[str]:
        return self.store.tenants()
