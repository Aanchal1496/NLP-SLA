"""API tests for document upload / pasted-text endpoints."""

from fastapi.testclient import TestClient

from app.main import app

from ._pdf_helper import make_pdf_bytes

client = TestClient(app)

MARATHI = "मुंबई उच्च न्यायालयाने मालमत्ता वादात महत्त्वाचा निर्णय दिला."


def test_health_ok():
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_pasted_text_success_marathi():
    resp = client.post("/api/v1/documents/text", json={"text": MARATHI})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert MARATHI in body["text"]
    assert body["source_type"] == "pasted_text"
    assert body["char_count"] == len(body["text"])


def test_pasted_text_empty_fails_gracefully():
    resp = client.post("/api/v1/documents/text", json={"text": "   "})
    assert resp.status_code == 422


def test_upload_txt_success():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("karar.txt", MARATHI.encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    assert MARATHI in resp.json()["text"]


def test_upload_pdf_success():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("nirnay.pdf", make_pdf_bytes(MARATHI), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    assert MARATHI in resp.json()["text"]


def test_upload_unsupported_fails():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("run.exe", b"MZ...", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_empty_fails():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_malformed_pdf_fails():
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )
    assert resp.status_code == 422
