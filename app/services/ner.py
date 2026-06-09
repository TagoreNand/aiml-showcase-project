"""spaCy-backed Named Entity Recognition (optional).

Provides statistical entity extraction (ORG, PERSON, MONEY, DATE, GPE, ...)
that complements the deterministic regex rules. If spaCy or its model is not
installed, ``ner_entities`` returns an empty dict and the caller falls back to
rules — so this never breaks the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.config import Settings

logger = logging.getLogger(__name__)

_NLP = None
_LOAD_FAILED = False

# Map spaCy entity labels onto DocuPilot's field names.
_LABEL_MAP = {
    "ORG": "organizations",
    "PERSON": "people",
    "MONEY": "money_mentions",
    "DATE": "dates",
    "GPE": "locations",
    "LOC": "locations",
}


def _load():
    global _NLP, _LOAD_FAILED
    if _NLP is not None or _LOAD_FAILED:
        return _NLP
    try:
        import spacy  # type: ignore

        try:
            _NLP = spacy.load(Settings.SPACY_MODEL, disable=["lemmatizer"])
        except Exception:  # model not downloaded -> blank English w/ NER absent
            logger.warning(
                "spaCy model '%s' not found; NER backend disabled. "
                "Run: python -m spacy download %s",
                Settings.SPACY_MODEL,
                Settings.SPACY_MODEL,
            )
            _LOAD_FAILED = True
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy unavailable (%s); NER backend disabled", exc)
        _LOAD_FAILED = True
        return None
    return _NLP


def available() -> bool:
    return _load() is not None


def ner_entities(text: str) -> Dict[str, Any]:
    nlp = _load()
    if nlp is None:
        return {}
    doc = nlp(text)
    grouped: Dict[str, list] = {}
    for ent in doc.ents:
        field = _LABEL_MAP.get(ent.label_)
        if not field:
            continue
        grouped.setdefault(field, [])
        value = ent.text.strip()
        if value and value not in grouped[field]:
            grouped[field].append(value)
    return grouped
