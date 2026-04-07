from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class FeedbackStore:
    def __init__(self, feedback_path: Path):
        self.feedback_path = feedback_path
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_path.exists():
            self.feedback_path.touch()

    def save(self, payload: Dict[str, Any]) -> None:
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def count(self) -> int:
        if not self.feedback_path.exists():
            return 0
        with open(self.feedback_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
