"""OCR / text extraction for uploaded files (scanned PDFs, images, text).

Strategy (local-first, graceful degradation):

1. Plain text / markdown / csv  -> decode directly (no OCR needed).
2. PDF with a text layer        -> extract with ``pypdf`` (fast, no OCR).
3. Scanned PDF or image         -> rasterise + ``pytesseract`` OCR.

If the optional OCR stack (``pytesseract``/``pdf2image``/``Pillow`` and the
system ``tesseract``/``poppler`` binaries) is unavailable, the function returns
an empty-but-explained result instead of raising, so the API stays up.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List

from app.config import Settings

logger = logging.getLogger(__name__)

TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class OcrResult:
    text: str
    source: str = "unknown"          # text | pdf-text | pdf-ocr | image-ocr | none
    ocr_used: bool = False
    pages: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def _ext(filename: str) -> str:
    i = filename.rfind(".")
    return filename[i:].lower() if i >= 0 else ""


def _pdf_text_layer(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:  # noqa: BLE001
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf text extraction failed: %s", exc)
        return ""


def _ocr_image_bytes(data: bytes) -> str:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img, lang=Settings.OCR_LANG).strip()


def _ocr_pdf(data: bytes) -> tuple[str, int]:
    from pdf2image import convert_from_bytes  # type: ignore
    import pytesseract  # type: ignore

    images = convert_from_bytes(data, dpi=Settings.OCR_DPI)
    chunks = [
        pytesseract.image_to_string(img, lang=Settings.OCR_LANG) for img in images
    ]
    return "\n".join(chunks).strip(), len(images)


def extract_text(data: bytes, filename: str) -> OcrResult:
    """Extract text from raw bytes, choosing the cheapest viable method."""
    ext = _ext(filename)

    # 1) Plain text-like files.
    if ext in TEXT_EXTS:
        try:
            return OcrResult(text=data.decode("utf-8", errors="replace"), source="text")
        except Exception as exc:  # noqa: BLE001
            return OcrResult(text="", source="none", notes=[f"decode error: {exc}"])

    # 2) PDFs: prefer text layer, then OCR.
    if ext == ".pdf":
        layer = _pdf_text_layer(data)
        if layer:
            return OcrResult(text=layer, source="pdf-text", pages=layer.count("\f") + 1)
        if not Settings.OCR_ENABLED:
            return OcrResult(text="", source="none", notes=["OCR disabled; PDF has no text layer"])
        try:
            text, pages = _ocr_pdf(data)
            return OcrResult(text=text, source="pdf-ocr", ocr_used=True, pages=pages)
        except Exception as exc:  # noqa: BLE001
            return OcrResult(
                text="",
                source="none",
                notes=[f"OCR unavailable ({exc}); install pytesseract+pdf2image+poppler"],
            )

    # 3) Images.
    if ext in IMAGE_EXTS:
        if not Settings.OCR_ENABLED:
            return OcrResult(text="", source="none", notes=["OCR disabled"])
        try:
            return OcrResult(text=_ocr_image_bytes(data), source="image-ocr", ocr_used=True, pages=1)
        except Exception as exc:  # noqa: BLE001
            return OcrResult(
                text="",
                source="none",
                notes=[f"OCR unavailable ({exc}); install pytesseract+Pillow+tesseract"],
            )

    # 4) Last resort: best-effort decode.
    try:
        return OcrResult(text=data.decode("utf-8", errors="replace"), source="text")
    except Exception as exc:  # noqa: BLE001
        return OcrResult(text="", source="none", notes=[f"unsupported file type '{ext}': {exc}"])
