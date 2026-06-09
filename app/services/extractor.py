"""Deterministic, rule-based entity extraction.

Hardened regex + heuristics that form the reliable base layer of the extraction
stack (NER and LLM augment this; see ``app.services.extraction``). Output keys
are unchanged for backward compatibility.
"""

import re
from typing import Any, Dict, Optional

# Words that should terminate a greedily-captured organisation/vendor name.
_VENDOR_STOPWORDS = re.compile(
    r"\s+(?:for|due|invoice|dated|on|to|regarding|re|priority|amount|total)\b",
    re.IGNORECASE,
)


def _search(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _clean_org(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    value = _VENDOR_STOPWORDS.split(value)[0]
    return value.strip(" .,-")


def _extract_amount(text: str) -> Optional[str]:
    # 1) Prefer an explicit currency-prefixed amount.
    m = re.search(r"\$\s?([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", text)
    if m:
        return m.group(1).replace(",", "")
    # 2) Otherwise an amount/total/sum labelled number.
    m = re.search(
        r"(?:amount|total|sum|balance)(?:\s+due)?[:\s]+\$?\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    return None


def extract_entities(text: str, label: str | None = None) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}

    email = _search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    amount = _extract_amount(text)
    invoice_number = _search(r"(INV[-\s]?\d{3,6})", text, re.IGNORECASE)
    priority = _search(r"priority[:\s]+(low|medium|high|critical)", text, re.IGNORECASE)
    vendor = _clean_org(_search(r"(?:from|issued by|vendor[:\s]+)\s*([A-Z][A-Za-z0-9 &.,-]+)", text))
    candidate_name = _search(
        r"candidate[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)", text, re.IGNORECASE
    )
    term_months = _search(r"(\d+)\s+months?", text, re.IGNORECASE)
    due_date = _search(r"due(?:\s+on)?[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)

    contract_match = re.search(
        r"between\s+([A-Z][A-Za-z0-9 &.-]+?)\s+and\s+([A-Z][A-Za-z0-9 &.-]+?)(?:\s+for|\s+regarding|\.|,|$)",
        text,
    )
    if contract_match:
        entities["contract_parties"] = [
            contract_match.group(1).strip(),
            contract_match.group(2).strip(),
        ]

    if email:
        entities["email"] = email
    if amount:
        entities["amount"] = amount
    if invoice_number:
        entities["invoice_number"] = invoice_number.upper().replace(" ", "-")
    if priority:
        entities["priority"] = priority.lower()
    if vendor:
        entities["vendor"] = vendor
    if candidate_name:
        entities["candidate_name"] = candidate_name
    if term_months:
        entities["term_months"] = int(term_months)
    if due_date:
        entities["due_date"] = due_date

    if label == "resume" and "candidate_name" not in entities:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line.split()) >= 2 and all(
                token[:1].isupper() for token in first_line.split()[:2]
            ):
                entities["candidate_name"] = first_line

    return entities
