from app.config import Settings
from app.services.retrieval import RetrievalEngine


SAMPLE_DOCS = [
    {
        "document_id": "seed_invoice_001",
        "text": "Invoice INV-3001 from Atlas Compute for $1899 due on 2026-04-20 for cloud infrastructure usage."
    },
    {
        "document_id": "seed_contract_001",
        "text": "Service agreement between Helios Retail and Quantum Ops for 12 months focused on supply chain analytics."
    },
    {
        "document_id": "seed_ticket_001",
        "text": "Support ticket priority high. Checkout failures increased after the weekend deployment."
    },
]


if __name__ == "__main__":
    Settings.ensure_dirs()
    engine = RetrievalEngine(Settings.CORPUS_PATH)
    for doc in SAMPLE_DOCS:
        engine.add_document(doc["document_id"], doc["text"])
    print(f"Seeded {len(SAMPLE_DOCS)} sample documents.")
