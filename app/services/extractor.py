import re
from typing import Any, Dict


def _search(pattern: str, text: str, flags: int = 0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def extract_entities(text: str, label: str | None = None) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}

    email = _search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    amount = _search(r'\$?\s?([0-9]+(?:\.[0-9]{2})?)', text)
    invoice_number = _search(r'(INV[-\s]?\d{3,6})', text, re.IGNORECASE)
    priority = _search(r'priority[:\s]+(low|medium|high|critical)', text, re.IGNORECASE)
    vendor = _search(r'from\s+([A-Z][A-Za-z0-9 &.-]+)', text)
    candidate_name = _search(r'candidate[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)', text, re.IGNORECASE)
    term_months = _search(r'(\d+)\s+months?', text, re.IGNORECASE)
    due_date = _search(r'due(?:\s+on)?[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})', text, re.IGNORECASE)

    contract_match = re.search(r'between\s+([A-Z][A-Za-z0-9 &.-]+)\s+and\s+([A-Z][A-Za-z0-9 &.-]+)', text)
    if contract_match:
        entities["contract_parties"] = [contract_match.group(1).strip(), contract_match.group(2).strip()]

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
            if len(first_line.split()) >= 2 and all(token[:1].isupper() for token in first_line.split()[:2]):
                entities["candidate_name"] = first_line

    return entities
