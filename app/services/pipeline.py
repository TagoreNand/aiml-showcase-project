from __future__ import annotations

from typing import Dict, Any

from app.services.extractor import extract_entities


def generate_summary(text: str, label: str, entities: Dict[str, Any]) -> str:
    preview = text.replace("\n", " ").strip()[:160]
    entity_bits = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "no key entities"
    return f"Detected as '{label}'. Summary: {preview}. Extracted: {entity_bits}."


def analyze_document(text: str, classifier) -> Dict[str, Any]:
    classification = classifier.predict(text)
    label = classification["label"]
    entities = extract_entities(text, label=label)
    summary = generate_summary(text=text, label=label, entities=entities)
    return {
        "classification": classification,
        "entities": entities,
        "summary": summary,
    }


def answer_question(question: str, retrieval_engine, top_k: int = 3) -> Dict[str, Any]:
    evidence = retrieval_engine.search(question, top_k=top_k)
    if not evidence:
        return {
            "answer": "No evidence found in the current corpus. Ingest documents first.",
            "evidence": [],
        }

    best = evidence[0]
    answer = (
        f"Based on the most relevant document ({best['document_id']}), the strongest matching evidence is: "
        f"{best['excerpt']}"
    )
    return {"answer": answer, "evidence": evidence}
