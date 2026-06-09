-- Enable pgvector and a starter schema for DocuPilot multi-tenant documents.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'public',
    document_id TEXT NOT NULL,
    text        TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, document_id)
);

CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id);
-- Approximate nearest-neighbour index for cosine distance.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
