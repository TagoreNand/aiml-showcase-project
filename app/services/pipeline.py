from __future__ import annotations

from typing import Any, Dict

from app.config import Settings
from app.services.extraction import run_extraction
from app.services import llm


def generate_summary(text: str, label: str, entities: Dict[str, Any]) -> str:
    preview = text.replace("\n", " ").strip()[:160]
    entity_bits = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "no key entities"
    return f"Detected as '{label}'. Summary: {preview}. Extracted: {entity_bits}."


def analyze_document(text: str, classifier) -> Dict[str, Any]:
    classification = classifier.predict(text)
    label = classification["label"]
    entities, methods = run_extraction(text, label=label)
    summary = generate_summary(text=text, label=label, entities=entities)
    return {
        "classification": classification,
        "entities": entities,
        "summary": summary,
        "extraction_methods": methods,
    }


def answer_question(question: str, retrieval_engine, top_k: int = 3, tenant_id: str | None = None) -> Dict[str, Any]:
    evidence = retrieval_engine.search(
        question, top_k=top_k, tenant_id=tenant_id or Settings.DEFAULT_TENANT
    )
    if not evidence:
        return {
            "answer": "No evidence found in the current corpus. Ingest documents first.",
            "evidence": [],
            "answer_backend": "none",
        }

    # Prefer an LLM-synthesised, grounded answer when configured & available.
    context = "\n\n".join(f"[{e['document_id']}] {e['excerpt']}" for e in evidence)
    llm_answer = llm.answer(question, context)
    if llm_answer:
        return {"answer": llm_answer.strip(), "evidence": evidence, "answer_backend": "llm"}

    best = evidence[0]
    answer = (
        f"Based on the most relevant document ({best['document_id']}), the strongest matching evidence is: "
        f"{best['excerpt']}"
    )
    return {"answer": answer, "evidence": evidence, "answer_backend": "extractive"}
