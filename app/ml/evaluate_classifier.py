from __future__ import annotations

import json
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.config import Settings


def load_training_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, labels


def evaluate() -> None:
    texts, labels = load_training_data(Settings.TRAINING_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    evaluate()
