"""Document text-extraction service.

Reusable, route-independent logic for:
- pasted (typed/copied) text normalization
- .txt byte extraction
- text-based .pdf extraction via PyMuPDF
- .docx extraction via python-docx (paragraphs + tables)

Security notes:
- Callers pass raw bytes + an untrusted filename; only the lower-cased
  suffix is used for type detection. The full filename is never used as a
  path, never executed, and never trusted for content sniffing alone.
- Uploaded bytes are written to a randomly-named temp file (correct suffix)
  and always removed in a ``finally`` block.
"""

from __future__ import annotations

import os
import re
import tempfile

import pymupdf  # PyMuPDF

ALLOWED_EXTENSIONS = frozenset({".txt", ".pdf", ".docx"})
PDF_MAGIC = b"%PDF-"
DOCX_MAGIC = b"PK\x03\x04"  # .docx is a ZIP archive
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB

# Collapse 3+ consecutive newlines down to a paragraph break.
_BLANK_RUN_RE = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """Base class for all extraction failures."""


class UnsupportedFileTypeError(DocumentExtractionError):
    pass


class EmptyFileError(DocumentExtractionError):
    pass


class FileTooLargeError(DocumentExtractionError):
    pass


class EmptyExtractedTextError(DocumentExtractionError):
    pass


class EncryptedPDFError(DocumentExtractionError):
    pass


class CorruptDocumentError(DocumentExtractionError):
    pass


def normalize_text(text: str) -> str:
    """Normalize pasted/extracted text while preserving Devanagari.

    - Normalises CRLF/CR to LF.
    - Strips trailing whitespace per line (keeps leading spaces).
    - Collapses 3+ newlines to a single blank line (paragraph boundary).
    - Strips leading/trailing blank lines.
    - Never touches non-ASCII characters.
    """
    if not isinstance(text, str):
        raise EmptyExtractedTextError("Input text must be a string.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = _BLANK_RUN_RE.sub("\n\n", normalized)
    normalized = normalized.strip("\n ")
    if not normalized.strip():
        raise EmptyExtractedTextError("Pasted text is empty after normalization.")
    return normalized


def normalize_pasted_text(text: str) -> str:
    """Entry point for pasted-text input (requirement 1)."""
    if text is None or (isinstance(text, str) and not text.strip()):
        raise EmptyExtractedTextError("Pasted text must not be empty.")
    return normalize_text(text)


def _decode_txt_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    raise CorruptDocumentError(
        "Could not decode .txt file as UTF-8 (or UTF-16). "
        "Please upload a UTF-8 encoded text file."
    )


def _extract_pdf_text_from_path(path: str) -> str:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise CorruptDocumentError(f"Could not read PDF file: {exc}") from exc
    try:
        if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
            raise EncryptedPDFError(
                "PDF is password-protected. Please upload an unprotected PDF."
            )
        try:
            page_count = doc.page_count
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read PDF structure: {exc}") from exc
        if page_count == 0:
            raise EmptyExtractedTextError("PDF contains no pages.")
        pages: list[str] = []
        try:
            for page in doc:
                try:
                    pages.append(page.get_text("text"))
                except Exception as exc:
                    raise CorruptDocumentError(
                        f"Could not extract text from a PDF page: {exc}"
                    ) from exc
        except CorruptDocumentError:
            raise
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read PDF pages: {exc}") from exc
        combined = "\n\n".join(p for p in pages if p and p.strip())
        if not combined.strip():
            raise EmptyExtractedTextError(
                "No extractable text found in PDF. "
                "Scanned/image-only PDFs are not supported."
            )
        return normalize_text(combined)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _extract_docx_text_from_path(path: str) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise CorruptDocumentError(
            "Word document support is unavailable (python-docx not installed)."
        ) from exc
    try:
        doc = Document(path)
    except Exception as exc:
        raise CorruptDocumentError(f"Could not read Word file: {exc}") from exc
    parts: list[str] = []
    try:
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        parts.append(cell.text)
    except Exception as exc:
        raise CorruptDocumentError(
            f"Could not extract text from Word file: {exc}"
        ) from exc
    combined = "\n\n".join(parts)
    if not combined.strip():
        raise EmptyExtractedTextError("No extractable text found in Word file.")
    return normalize_text(combined)


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    # Never trust directories in the uploaded name; suffix only.
    base = os.path.basename(filename)
    _, dot, suffix = base.rpartition(".")
    if not dot:
        return ""
    return f".{suffix.lower()}"


def extract_from_upload(
    filename: str | None,
    data: bytes | None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> str:
    """Extract normalized text from uploaded ``.txt`` / ``.pdf`` / ``.docx`` bytes.

    Writes bytes to a secure temp file, extracts, and always cleans up.
    Raises a ``DocumentExtractionError`` subclass with a user-facing message.
    """
    if data is None or len(data) == 0:
        raise EmptyFileError("Uploaded file is empty (0 bytes).")
    if len(data) > max_size_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds the "
            f"{max_size_bytes} bytes limit."
        )

    suffix = _safe_suffix(filename)
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            "Only .txt, .pdf and .docx files are supported."
        )

    if suffix == ".pdf" and not data.lstrip().startswith(PDF_MAGIC):
        raise CorruptDocumentError("File is not a valid PDF (missing %PDF- header).")
    if suffix == ".docx" and not data.lstrip().startswith(DOCX_MAGIC):
        raise CorruptDocumentError("File is not a valid Word file (missing ZIP header).")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="nlp_doc_", delete=False
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
        if suffix == ".txt":
            with open(tmp_path, "rb") as fh:
                raw = fh.read()
            return normalize_text(_decode_txt_bytes(raw))
        if suffix == ".docx":
            return _extract_docx_text_from_path(tmp_path)
        return _extract_pdf_text_from_path(tmp_path)
    finally:
        if tmp_path is not None:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass


def detect_source_type(filename: str | None) -> str:
    suffix = _safe_suffix(filename)
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    return "txt"
