"""Train + register the document classifier.

Adds, on top of the original trainer:
* a held-out evaluation (accuracy + macro-F1),
* a drift *baseline* capture (label distribution + text stats),
* optional MLflow experiment tracking & model logging (graceful fallback to a
  local ``metrics.json`` / ``model_card.json`` when MLflow is unavailable).

``train()`` still returns the model path, so existing callers are unaffected.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.observability.monitoring import get_monitor

logger = logging.getLogger(__name__)


def load_training_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, labels


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def _evaluate(texts, labels) -> dict:
    """Held-out metrics; falls back to in-sample if a class is too small."""
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = texts, texts, labels, labels
    pipe = _build_pipeline().fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return {
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "macro_f1": round(float(f1_score(y_te, preds, average="macro")), 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


def _log_mlflow(params: dict, metrics: dict, model, model_path: Path) -> bool:
    if not Settings.MLFLOW_ENABLED:
        return False
    try:
        import mlflow  # type: ignore
        import mlflow.sklearn  # type: ignore

        # The model registry needs a tracking *server*; a local file store
        # (the default when no URI is set) doesn't support it, so only register
        # when a real tracking URI is configured.
        has_server = bool(Settings.MLFLOW_TRACKING_URI)
        if has_server:
            mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Settings.MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="docupilot-classifier" if has_server else None,
            )
        logger.info("Logged run to MLflow experiment '%s'", Settings.MLFLOW_EXPERIMENT)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging skipped (%s)", exc)
        return False


def train() -> Path:
    Settings.ensure_dirs()
    texts, labels = load_training_data(Settings.TRAINING_DATA_PATH)

    params = {
        "model": "LogisticRegression",
        "vectorizer": "tfidf(1,2)",
        "max_iter": 1000,
        "n_samples": len(texts),
        "n_classes": len(set(labels)),
    }

    metrics = _evaluate(texts, labels)

    pipeline = _build_pipeline().fit(texts, labels)
    model_path = Settings.MODEL_DIR / "document_classifier.joblib"
    joblib.dump(pipeline, model_path)

    # Capture drift baseline from the training distribution.
    get_monitor().write_baseline(labels, texts)

    # MLflow (optional) + always-on local model card.
    mlflow_logged = _log_mlflow(params, metrics, pipeline, model_path)
    card = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "params": params,
        "metrics": metrics,
        "labels": sorted(set(labels)),
        "mlflow_logged": mlflow_logged,
    }
    (Settings.METRICS_DIR / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )

    print(f"Saved classifier to: {model_path}")
    print(f"Eval: accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']}")
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
