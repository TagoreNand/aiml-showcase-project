from app.services.ocr import extract_text


def test_plain_text_no_ocr_needed():
    r = extract_text(b"Invoice INV-1 from Acme for $10", "doc.txt")
    assert r.ok and r.source == "text" and r.ocr_used is False


def test_unextractable_image_degrades_gracefully():
    # Not a real image; OCR stack likely absent -> must not raise.
    r = extract_text(b"\x89PNG-not-real", "scan.png")
    assert r.ok is False
    assert r.source == "none"
