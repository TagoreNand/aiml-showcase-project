from __future__ import annotations

import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import Settings


def load_training_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, labels


def train() -> Path:
    Settings.ensure_dirs()
    texts, labels = load_training_data(Settings.TRAINING_DATA_PATH)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(texts, labels)

    model_path = Settings.MODEL_DIR / "document_classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved classifier to: {model_path}")
    return model_path


if __name__ == "__main__":
    train()
