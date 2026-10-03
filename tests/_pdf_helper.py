"""Shared test helper: build text-based PDFs containing real Devanagari text.

PyMuPDF's ``Page.insert_text`` with its default (Latin-only) fonts cannot
round-trip Devanagari — glyphs come back as U+00B7 dots — so fixture PDFs
are generated with reportlab using a Devanagari-capable font instead.
Production extraction (``app.services.text_extraction``) is unchanged and
reads these PDFs via PyMuPDF.
"""

from __future__ import annotations

import io
import os

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

_FONT_NAME = "DevanagariTest"
_registered = False

# (path, subfontIndex or None). Nirmala.ttc ships with Windows; Noto paths
# cover common Linux runners.
_FONT_CANDIDATES: tuple[tuple[str, int | None], ...] = (
    ("C:/Windows/Fonts/Nirmala.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", None),
    ("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf", None),
)


def _ensure_font() -> str:
    global _registered
    if _registered:
        return _FONT_NAME
    for path, index in _FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            if index is None:
                pdfmetrics.registerFont(TTFont(_FONT_NAME, path))
            else:
                pdfmetrics.registerFont(
                    TTFont(_FONT_NAME, path, subfontIndex=index)
                )
            _registered = True
            return _FONT_NAME
        except Exception:
            continue
    raise RuntimeError(
        "No Devanagari-capable font found. Install Noto Sans Devanagari "
        "or run tests on Windows (Nirmala.ttc)."
    )


def make_pdf_bytes(paragraphs: str | list[str]) -> bytes:
    """Render paragraphs into a text-based PDF and return its bytes."""
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    font = _ensure_font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFont(font, 11)
    y = 750.0
    for para in paragraphs:
        if y < 80:
            c.showPage()
            c.setFont(font, 11)
            y = 750.0
        c.drawString(72, y, para)
        y -= 20.0
    c.save()
    return buf.getvalue()
