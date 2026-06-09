"""Pluggable text embedding backends.

Two backends are provided:

* ``HashingEmbedder`` (default) — pure scikit-learn ``HashingVectorizer`` that
  maps text to a fixed-dimension, L2-normalised dense vector. Deterministic and
  dependency-free, so the pgvector path works without any model download.
* ``SentenceTransformerEmbedder`` — real dense semantic embeddings via
  ``sentence-transformers`` when installed and selected.

``get_embedder()`` chooses based on ``Settings`` and *always* returns a working
embedder, falling back to hashing if the heavy dependency is unavailable.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class Embedder:
    """Minimal embedder interface: fixed-dim, L2-normalised dense vectors."""

    name: str = "base"
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic hashing-trick embeddings (no model, no network)."""

    name = "hashing"

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vec.transform(list(texts))
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Dense semantic embeddings via sentence-transformers (optional dep)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


_EMBEDDER: Embedder | None = None


def get_embedder(force_reload: bool = False) -> Embedder:
    """Return a process-wide embedder instance, honouring config with fallback."""
    global _EMBEDDER
    if _EMBEDDER is not None and not force_reload:
        return _EMBEDDER

    backend = Settings.EMBEDDING_BACKEND
    if backend in {"sentence-transformers", "st", "sbert"}:
        try:
            _EMBEDDER = SentenceTransformerEmbedder(Settings.EMBEDDING_MODEL)
            logger.info("Using SentenceTransformer embedder (%s)", Settings.EMBEDDING_MODEL)
            return _EMBEDDER
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to hashing embedder",
                exc,
            )

    _EMBEDDER = HashingEmbedder(Settings.EMBEDDING_DIM)
    logger.info("Using hashing embedder (dim=%d)", _EMBEDDER.dim)
    return _EMBEDDER


def cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a matrix of row vectors."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query_vec.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    q = q / np.clip(q_norm, 1e-12, None)
    m = matrix / np.clip(m_norm, 1e-12, None)
    return (m @ q.T).ravel()
