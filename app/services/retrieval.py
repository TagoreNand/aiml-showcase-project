from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalEngine:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.documents: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_vectors = None
        self._load()

    def _load(self) -> None:
        if self.corpus_path.exists():
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            self.documents = []
        self._reindex()

    def _persist(self) -> None:
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    def _reindex(self) -> None:
        if not self.documents:
            self.doc_vectors = None
            return
        texts = [doc["text"] for doc in self.documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def add_document(self, document_id: str, text: str) -> None:
        self.documents = [d for d in self.documents if d["document_id"] != document_id]
        self.documents.append({"document_id": document_id, "text": text})
        self._persist()
        self._reindex()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str | float]]:
        if not self.documents or self.doc_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append({
                "document_id": doc["document_id"],
                "score": float(round(similarities[idx], 4)),
                "excerpt": doc["text"][:220],
            })
        return results

    def count(self) -> int:
        return len(self.documents)
