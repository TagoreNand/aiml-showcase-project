"""In-memory, JSON-persisted vector store (the local-first default).

Supports two scoring modes:

* ``tfidf`` (default) — re-fits a ``TfidfVectorizer`` over the tenant's corpus
  and ranks by cosine similarity. This reproduces the original DocuPilot
  retrieval behaviour exactly, so existing behaviour/tests are preserved.
* ``dense`` — uses the configured :class:`~app.services.embeddings.Embedder`
  (hashing or sentence-transformers) for semantic search.

Documents are namespaced by ``tenant_id`` to provide multi-tenant isolation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.services.embeddings import cosine_sim, get_embedder
from app.services.vectorstore.base import StoredDocument, VectorStore

logger = logging.getLogger(__name__)
EXCERPT_LEN = 220


class InMemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self, corpus_path: Path, mode: Optional[str] = None):
        self.corpus_path = Path(corpus_path)
        # "tfidf" keeps original behaviour; "dense" uses embeddings.
        self.mode = (mode or ("dense" if Settings.EMBEDDING_BACKEND != "tfidf" else "tfidf"))
        self._docs: List[StoredDocument] = []
        self._index_cache: Dict[str, Any] = {}
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.corpus_path.exists():
            try:
                raw = json.loads(self.corpus_path.read_text(encoding="utf-8") or "[]")
            except json.JSONDecodeError:
                raw = []
            for row in raw:
                self._docs.append(
                    StoredDocument(
                        document_id=row["document_id"],
                        text=row["text"],
                        tenant_id=row.get("tenant_id", Settings.DEFAULT_TENANT),
                        metadata=row.get("metadata", {}),
                    )
                )
        self._invalidate()

    def _persist(self) -> None:
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        records = [d.to_record() for d in self._docs]
        self.corpus_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _invalidate(self) -> None:
        self._index_cache = {}

    # ---- helpers -----------------------------------------------------------
    def _tenant_docs(self, tenant_id: str) -> List[StoredDocument]:
        return [d for d in self._docs if d.tenant_id == tenant_id]

    # ---- VectorStore API ---------------------------------------------------
    def add(
        self,
        document_id: str,
        text: str,
        tenant_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._docs = [
            d for d in self._docs
            if not (d.document_id == document_id and d.tenant_id == tenant_id)
        ]
        self._docs.append(
            StoredDocument(document_id, text, tenant_id, metadata or {})
        )
        self._persist()
        self._invalidate()

    def delete(self, document_id: str, tenant_id: str = "public") -> bool:
        before = len(self._docs)
        self._docs = [
            d for d in self._docs
            if not (d.document_id == document_id and d.tenant_id == tenant_id)
        ]
        removed = len(self._docs) != before
        if removed:
            self._persist()
            self._invalidate()
        return removed

    def count(self, tenant_id: Optional[str] = None) -> int:
        if tenant_id is None:
            return len(self._docs)
        return len(self._tenant_docs(tenant_id))

    def tenants(self) -> List[str]:
        return sorted({d.tenant_id for d in self._docs})

    def search(
        self, query: str, top_k: int = 3, tenant_id: str = "public"
    ) -> List[Dict[str, Any]]:
        docs = self._tenant_docs(tenant_id)
        if not docs:
            return []
        if self.mode == "dense":
            scores = self._dense_scores(tenant_id, docs, query)
        else:
            scores = self._tfidf_scores(tenant_id, docs, query)

        ranked = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked:
            doc = docs[int(idx)]
            results.append({
                "document_id": doc.document_id,
                "score": float(round(float(scores[int(idx)]), 4)),
                "excerpt": doc.text[:EXCERPT_LEN],
            })
        return results

    # ---- scoring -----------------------------------------------------------
    def _tfidf_scores(self, tenant_id, docs, query) -> np.ndarray:
        cache = self._index_cache.get(("tfidf", tenant_id))
        if cache is None:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vec.fit_transform([d.text for d in docs])
            cache = (vec, matrix)
            self._index_cache[("tfidf", tenant_id)] = cache
        vec, matrix = cache
        qv = vec.transform([query])
        return cosine_similarity(qv, matrix)[0]

    def _dense_scores(self, tenant_id, docs, query) -> np.ndarray:
        embedder = get_embedder()
        cache = self._index_cache.get(("dense", tenant_id))
        if cache is None:
            matrix = embedder.encode([d.text for d in docs])
            cache = matrix
            self._index_cache[("dense", tenant_id)] = cache
        matrix = cache
        qv = embedder.encode([query])[0]
        return cosine_sim(qv, matrix)
