"""Service-level tests for app.services.text_extraction."""

import glob
import os
import tempfile

import pymupdf
import pytest

from app.services import text_extraction as svc

from ._pdf_helper import make_pdf_bytes

MARATHI_1 = "मुंबई उच्च न्यायालयाने मालमत्ता वादात महत्त्वाचा निर्णय दिला."
MARATHI_2 = "करारानुसार देय रक्कम ₹५,००,००० इतकी निश्चित करण्यात आली."


def make_encrypted_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), MARATHI_1)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        doc.save(
            tmp_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw="secret",
            owner_pw="owner",
        )
        doc.close()
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def leftover_temp_docs():
    return glob.glob(os.path.join(tempfile.gettempdir(), "nlp_doc_*"))


def test_pasted_marathi_preserved():
    text = f"{MARATHI_1}\n\n{MARATHI_2}"
    out = svc.normalize_pasted_text(text)
    assert MARATHI_1 in out
    assert MARATHI_2 in out
    assert "\n\n" in out  # paragraph boundary preserved


def test_pasted_text_empty_fails():
    with pytest.raises(svc.EmptyExtractedTextError):
        svc.normalize_pasted_text("   \n  ")


def test_txt_extraction_marathi():
    data = f"{MARATHI_1}\n\n{MARATHI_2}\n".encode("utf-8")
    before = set(leftover_temp_docs())
    out = svc.extract_from_upload(" karar.txt", data)
    assert MARATHI_1 in out
    assert MARATHI_2 in out
    assert set(leftover_temp_docs()) == before  # temp file cleaned up


def test_pdf_extraction_marathi():
    data = make_pdf_bytes([MARATHI_1, MARATHI_2])
    before = set(leftover_temp_docs())
    out = svc.extract_from_upload("nirnay.pdf", data)
    assert MARATHI_1 in out
    assert MARATHI_2 in out
    assert set(leftover_temp_docs()) == before


def test_unsupported_format_fails():
    with pytest.raises(svc.UnsupportedFileTypeError):
        svc.extract_from_upload("malicious.exe", b"MZ fake binary")


def test_no_extension_fails():
    with pytest.raises(svc.UnsupportedFileTypeError):
        svc.extract_from_upload("noextension", b"hello")


def test_empty_file_fails():
    with pytest.raises(svc.EmptyFileError):
        svc.extract_from_upload("empty.txt", b"")


def test_oversize_fails():
    with pytest.raises(svc.FileTooLargeError):
        svc.extract_from_upload("big.txt", b"a" * 100, max_size_bytes=10)


def test_whitespace_txt_empty_extraction_fails():
    with pytest.raises(svc.EmptyExtractedTextError):
        svc.extract_from_upload("blank.txt", b"   \n\n  ")


def test_blank_pdf_empty_extraction_fails():
    doc = pymupdf.open()
    doc.new_page()  # no text inserted
    data = bytes(doc.tobytes())
    doc.close()
    with pytest.raises(svc.EmptyExtractedTextError):
        svc.extract_from_upload("blank.pdf", data)


def test_malformed_pdf_fails():
    with pytest.raises(svc.CorruptDocumentError):
        svc.extract_from_upload("broken.pdf", b"definitely not a pdf")


def test_encrypted_pdf_fails_with_useful_error():
    data = make_encrypted_pdf_bytes()
    with pytest.raises(svc.EncryptedPDFError, match="password-protected"):
        svc.extract_from_upload("locked.pdf", data)


def test_temp_cleanup_on_failure():
    before = set(leftover_temp_docs())
    with pytest.raises(svc.DocumentExtractionError):
        svc.extract_from_upload("evil.pdf", b"not a pdf")
    assert set(leftover_temp_docs()) == before


def test_untrusted_filename_traversal_safe():
    data = "साधा मजकूर".encode("utf-8")
    out = svc.extract_from_upload("..\\..\\evil.txt", data)
    assert "साधा मजकूर" in out
