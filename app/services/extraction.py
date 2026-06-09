"""Extraction orchestrator: rules -> NER -> LLM (config driven, graceful).

The deterministic regex rules always run as a reliable base. Depending on
``EXTRACTION_BACKEND`` the result is then augmented:

* ``rules`` (default) — regex/heuristics only.
* ``ner``             — add spaCy entities (orgs, people, dates, ...).
* ``llm``             — add LLM-extracted structured fields (highest priority),
                        still backed by rules+NER for robustness.

Each augmentation is best-effort: if its dependency/credential is missing the
orchestrator silently keeps the lower tier, and reports which tiers actually
contributed via ``methods``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.config import Settings
from app.services.extractor import extract_entities
from app.services import ner, llm


def _merge(base: Dict[str, Any], extra: Dict[str, Any], overwrite: bool) -> None:
    for key, value in extra.items():
        if not value:
            continue
        if overwrite or key not in base:
            base[key] = value


def run_extraction(text: str, label: str | None = None) -> Tuple[Dict[str, Any], List[str]]:
    """Return (entities, methods_used)."""
    backend = Settings.EXTRACTION_BACKEND
    methods: List[str] = ["rules"]

    entities: Dict[str, Any] = extract_entities(text, label=label)

    if backend in {"ner", "hybrid", "llm"}:
        ner_found = ner.ner_entities(text)
        if ner_found:
            _merge(entities, ner_found, overwrite=False)
            methods.append("ner")

    if backend == "llm":
        llm_found = llm.extract(text, label=label)
        if llm_found:
            # LLM is the most capable extractor -> it wins on conflicts.
            _merge(entities, llm_found, overwrite=True)
            methods.append("llm")

    return entities, methods
