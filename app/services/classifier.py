from pathlib import Path
from typing import Dict, List
import joblib


class DocumentClassifier:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

    def predict(self, text: str) -> Dict[str, float | str]:
        if self.pipeline is None:
            self.load()

        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = probs.argmax()

        return {
            "label": str(classes[best_idx]),
            "confidence": float(round(probs[best_idx], 4)),
        }

    def labels(self) -> List[str]:
        if self.pipeline is None:
            self.load()
        return [str(label) for label in self.pipeline.classes_]
