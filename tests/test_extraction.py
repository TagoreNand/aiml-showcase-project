from app.services.extractor import extract_entities
from app.services.extraction import run_extraction


def test_invoice_amount_not_confused_with_invoice_number():
    text = "Invoice INV-2002 from Globex Corp for $1,234.50 due on 2026-05-01. priority high"
    ents = extract_entities(text, label="invoice")
    assert ents["amount"] == "1234.50"
    assert ents["invoice_number"] == "INV-2002"
    assert ents["vendor"] == "Globex Corp"
    assert ents["due_date"] == "2026-05-01"


def test_plain_dollar_amount_without_commas():
    ents = extract_entities("Invoice INV-1001 from Acme Cloud for $1499 due on 2026-04-10.", label="invoice")
    assert ents["amount"] == "1499"


def test_contract_parties_clean():
    ents = extract_entities("Service agreement between Helios Retail and Quantum Ops for 12 months.")
    assert ents["contract_parties"] == ["Helios Retail", "Quantum Ops"]
    assert ents["term_months"] == 12


def test_orchestrator_reports_methods_with_rules_default():
    entities, methods = run_extraction("Support ticket priority critical after deploy.", label="support_ticket")
    assert "rules" in methods
    assert entities.get("priority") == "critical"
