"""Postgres + pgvector backend (optional, VECTOR_BACKEND=pgvector).

Uses dense embeddings so similarity search runs in the database via the
pgvector ``<=>`` (cosine distance) operator. Requires ``psycopg`` and
``pgvector`` plus a reachable Postgres with the ``vector`` extension.

If any of that is missing the constructor raises, and the factory in
``__init__`` transparently falls back to the in-memory store.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.services.embeddings import get_embedder
from app.services.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)
EXCERPT_LEN = 220


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, dsn: Optional[str] = None):
        # Imports are local so the module loads even without the optional deps.
        import psycopg  # type: ignore
        from pgvector.psycopg import register_vector  # type: ignore

        self._psycopg = psycopg
        self._register_vector = register_vector
        self.dsn = dsn or Settings.DATABASE_URL
        self.embedder = get_embedder()
        self.dim = self.embedder.dim
        self._conn = psycopg.connect(self.dsn, autocommit=True)
        self._init_schema()
        logger.info("PgVectorStore connected (dim=%d)", self.dim)

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id            BIGSERIAL PRIMARY KEY,
                    tenant_id     TEXT NOT NULL DEFAULT 'public',
                    document_id   TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding     vector({self.dim}),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (tenant_id, document_id)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS documents_tenant_idx "
                "ON documents (tenant_id);"
            )
        self._register_vector(self._conn)

    def add(
        self,
        document_id: str,
        text: str,
        tenant_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        import json as _json

        vector = self.embedder.encode([text])[0]
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (tenant_id, document_id, text, metadata, embedding)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (tenant_id, document_id)
                DO UPDATE SET text = EXCLUDED.text,
                              metadata = EXCLUDED.metadata,
                              embedding = EXCLUDED.embedding;
                """,
                (tenant_id, document_id, text, _json.dumps(metadata or {}), vector),
            )

    def delete(self, document_id: str, tenant_id: str = "public") -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE tenant_id = %s AND document_id = %s;",
                (tenant_id, document_id),
            )
            return cur.rowcount > 0

    def count(self, tenant_id: Optional[str] = None) -> int:
        with self._conn.cursor() as cur:
            if tenant_id is None:
                cur.execute("SELECT count(*) FROM documents;")
            else:
                cur.execute(
                    "SELECT count(*) FROM documents WHERE tenant_id = %s;",
                    (tenant_id,),
                )
            return int(cur.fetchone()[0])

    def tenants(self) -> List[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT tenant_id FROM documents ORDER BY tenant_id;")
            return [r[0] for r in cur.fetchall()]

    def search(
        self, query: str, top_k: int = 3, tenant_id: str = "public"
    ) -> List[Dict[str, Any]]:
        qv = self.embedder.encode([query])[0]
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, text, 1 - (embedding <=> %s) AS score
                FROM documents
                WHERE tenant_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (qv, tenant_id, qv, top_k),
            )
            rows = cur.fetchall()
        return [
            {
                "document_id": r[0],
                "score": float(round(float(r[2]), 4)),
                "excerpt": r[1][:EXCERPT_LEN],
            }
            for r in rows
        ]
