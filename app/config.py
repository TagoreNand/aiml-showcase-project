from pathlib import Path
import os


class Settings:
    APP_NAME = os.getenv("APP_NAME", "DocuPilot AI")
    APP_ENV = os.getenv("APP_ENV", "development")
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "data/models")
    CORPUS_PATH = BASE_DIR / os.getenv("CORPUS_PATH", "data/processed/corpus.json")
    FEEDBACK_PATH = BASE_DIR / os.getenv("FEEDBACK_PATH", "data/processed/feedback.jsonl")
    TRAINING_DATA_PATH = BASE_DIR / os.getenv("TRAINING_DATA_PATH", "data/raw/training_documents.json")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
