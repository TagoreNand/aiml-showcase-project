"""Prometheus instrumentation with a no-op fallback.

If ``prometheus_client`` is installed and ``PROMETHEUS_ENABLED`` is true, real
metrics are collected and exposed at ``/metrics/prometheus``. Otherwise every
helper becomes a cheap no-op so application code is identical in both modes.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from app.config import Settings

logger = logging.getLogger(__name__)

try:  # optional dependency
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM = True
except Exception:  # noqa: BLE001
    _PROM = False
    CONTENT_TYPE_LATEST = "text/plain"

ENABLED = _PROM and Settings.PROMETHEUS_ENABLED

if ENABLED:
    REGISTRY = CollectorRegistry()
    REQUESTS = Counter(
        "docupilot_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY
    )
    LATENCY = Histogram(
        "docupilot_request_latency_seconds", "Request latency", ["endpoint"], registry=REGISTRY
    )
    PREDICTIONS = Counter(
        "docupilot_predictions_total", "Classifier predictions", ["label"], registry=REGISTRY
    )
    CONFIDENCE = Histogram(
        "docupilot_prediction_confidence", "Prediction confidence", registry=REGISTRY
    )
    CORPUS_SIZE = Gauge(
        "docupilot_corpus_documents", "Documents in corpus", registry=REGISTRY
    )
else:
    REGISTRY = None


def record_request(endpoint: str, status: str = "ok") -> None:
    if ENABLED:
        REQUESTS.labels(endpoint=endpoint, status=status).inc()


def record_prediction(label: str, confidence: float) -> None:
    if ENABLED:
        PREDICTIONS.labels(label=label).inc()
        CONFIDENCE.observe(float(confidence))


def set_corpus_size(n: int) -> None:
    if ENABLED:
        CORPUS_SIZE.set(n)


@contextmanager
def observe_latency(endpoint: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        if ENABLED:
            LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)


def render_latest() -> tuple[bytes, str]:
    if not ENABLED:
        return (
            b"# prometheus_client not installed or disabled\n",
            "text/plain",
        )
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
