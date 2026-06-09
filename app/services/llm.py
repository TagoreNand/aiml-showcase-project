"""LLM-backed structured extraction and answer synthesis (optional).

Supports OpenAI and Anthropic providers. When ``LLM_PROVIDER=none`` (default),
the SDK is missing, or no API key is set, every entry point reports
"unavailable" and callers fall back to deterministic logic. This keeps the
project fully runnable offline while exposing a real LLM path for production.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from app.config import Settings

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = (
    "You are an information extraction engine for enterprise documents. "
    "Given a document and its type, return ONLY a compact JSON object of the "
    "salient structured fields (e.g. invoice_number, amount, vendor, due_date, "
    "candidate_name, email, priority, contract_parties, term_months). "
    "Omit fields that are absent. Do not add commentary."
)


def available() -> bool:
    provider = Settings.LLM_PROVIDER
    if provider == "openai":
        return bool(Settings.OPENAI_API_KEY)
    if provider == "anthropic":
        return bool(Settings.ANTHROPIC_API_KEY)
    return False


def _complete(system: str, user: str, max_tokens: int = 512) -> Optional[str]:
    """Provider-agnostic completion. Returns text or None on any failure."""
    provider = Settings.LLM_PROVIDER
    try:
        if provider == "openai":
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=Settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=Settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        if provider == "anthropic":
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=Settings.ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=Settings.LLM_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(getattr(b, "text", "") for b in resp.content)
    except Exception as exc:  # noqa: BLE001 - graceful fallback
        logger.warning("LLM call failed (%s); falling back", exc)
    return None


def _parse_json(blob: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract(text: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """LLM structured extraction. Returns dict, or None if unavailable/failed."""
    if not available():
        return None
    user = f"Document type: {label or 'unknown'}\n\nDocument:\n{text}"
    out = _complete(_EXTRACTION_SYSTEM, user)
    if out is None:
        return None
    return _parse_json(out)


def answer(question: str, context: str) -> Optional[str]:
    """Grounded QA over retrieved context. Returns answer, or None if unavailable."""
    if not available():
        return None
    system = (
        "Answer the question using ONLY the provided context. "
        "If the answer is not present, say so. Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return _complete(system, user, max_tokens=300)
