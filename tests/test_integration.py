# -*- coding: utf-8 -*-
"""Integration tests against the real FastAPI app (TestClient + live boot)."""

import csv
import io
import json
import subprocess
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)

PARA = (
    "मुंबई उच्च न्यायालयाने दिनांक १२/०५/२०२४ रोजी कलम ३०२ अंतर्गत "
    "सी.आर. नं. १२३/२०२४ मधील आरोपीस ₹५,००,००० दंड ठोठावला।"
)

REQUIRED = [
    ("GET", "/health"),
    ("POST", "/api/v1/analyze"),
    ("POST", "/api/v1/analyze-file"),
    ("POST", "/api/v1/preprocess"),
    ("POST", "/api/v1/predict-intent"),
    ("POST", "/api/v1/extract-entities"),
    ("GET", "/api/v1/dataset/stats"),
    ("GET", "/api/v1/dataset/records"),
    ("GET", "/api/v1/model/metrics"),
    ("POST", "/api/v1/export"),
]


def test_all_required_endpoints_registered():
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    for method, path in REQUIRED:
        assert path in paths, path
        assert method.lower() in paths[path], (method, path)
    assert client.get("/docs").status_code == 200


def test_analyze_combined_contract():
    resp = client.post("/api/v1/analyze", json={"text": PARA})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in (
        "original_text", "processed_text", "tokens", "sentences",
        "statistics", "intent", "entities", "entity_counts", "warnings",
    ):
        assert key in body, key
    assert body["original_text"] == PARA
    assert body["statistics"]["word_count"] == len(body["tokens"])
    assert body["statistics"]["sentence_count"] == len(body["sentences"])
    assert body["statistics"]["char_count"] == len(body["processed_text"])
    assert sum(body["entity_counts"].values()) == len(body["entities"])
    assert body["intent"] is not None  # starter model trained in this workspace
    assert body["intent"]["intent"] in body["intent"]["labels"]
    assert "intent-unavailable" not in body["warnings"]
    assert body["entities"]  # baseline NER always runs
    for ent in body["entities"]:
        assert PARA[ent["start"] : ent["end"]] == ent["text"]


def test_analyze_blank_rejected():
    assert client.post("/api/v1/analyze", json={"text": "  "}).status_code == 422


def test_analyze_file_txt_and_pdf():
    txt = client.post(
        "/api/v1/analyze-file",
        files={"file": ("a.txt", PARA.encode("utf-8"), "text/plain")},
    )
    assert txt.status_code == 200, txt.text
    assert txt.json()["entities"]

    from ._pdf_helper import make_pdf_bytes

    pdf = client.post(
        "/api/v1/analyze-file",
        files={"file": ("a.pdf", make_pdf_bytes(PARA), "application/pdf")},
    )
    assert pdf.status_code == 200, pdf.text
    assert "intent-unavailable" not in pdf.json()["warnings"]


def test_analyze_file_rejects_bad_uploads():
    bad = client.post(
        "/api/v1/analyze-file",
        files={"file": ("run.exe", b"MZ...", "application/octet-stream")},
    )
    assert bad.status_code == 400
    empty = client.post(
        "/api/v1/analyze-file",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert empty.status_code == 400


def test_flat_prediction_paths_match_primary():
    for flat, primary in (
        ("/api/v1/predict-intent", "/api/v1/predict/intent"),
        ("/api/v1/extract-entities", "/api/v1/predict/entities"),
    ):
        first = client.post(flat, json={"text": PARA})
        second = client.post(primary, json={"text": PARA})
        assert first.status_code == second.status_code
        assert first.json() == second.json()
    # Model available: predictions served, not a crash.
    assert client.post("/api/v1/predict-intent", json={"text": PARA}).status_code == 200
    assert client.post("/api/v1/predict-intent", json={"text": "  "}).status_code == 422


def test_dataset_and_metrics_do_not_fabricate():
    stats = client.get("/api/v1/dataset/stats")
    assert stats.status_code == 200
    assert stats.json()["available"] is True  # starter dataset connected
    assert stats.json()["records"]["used"] >= 64
    assert len(stats.json()["intent_distribution"]["labels"]) == 8

    records = client.get("/api/v1/dataset/records")
    assert records.status_code == 200
    assert records.json()["available"] is True
    assert records.json()["returned_records"] > 0

    metrics = client.get("/api/v1/model/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["available"] is True  # starter model trained
    assert metrics.json()["selected_model"] in metrics.json()["results"]


def test_records_pagination_validated():
    assert client.get("/api/v1/dataset/records?page=0").status_code == 422
    assert client.get("/api/v1/dataset/records?page_size=101").status_code == 422


def test_export_json_and_csv():
    as_json = client.post("/api/v1/export", json={"format": "json", "text": PARA})
    assert as_json.status_code == 200, as_json.text
    payload = as_json.json()
    assert payload["format"] == "json"
    assert payload["filename"] == "analysis.json"
    assert payload["mime_type"] == "application/json"
    parsed = json.loads(payload["content"])
    assert parsed["entity_counts"] and parsed["entities"]

    as_csv = client.post("/api/v1/export", json={"format": "csv", "text": PARA})
    assert as_csv.status_code == 200, as_csv.text
    payload = as_csv.json()
    assert payload["mime_type"] == "text/csv"
    rows = list(csv.DictReader(io.StringIO(payload["content"])))
    assert rows  # header + at least one clause row
    assert set(rows[0]) == {
        "section", "clause_index", "original", "clean", "intent",
        "intent_confidence", "entities", "english_translation",
        "translation_provider", "translation_model", "notes",
    }
    assert any(r["section"] == "clause" for r in rows)
    assert any("DATE" in r["entities"] for r in rows)

    import app.api.v1.analyze as analyze_api
    from app.schemas.analysis import TranslationPart

    real_resolve = analyze_api._resolve_translation
    analyze_api._resolve_translation = lambda text: TranslationPart(
        available=True, text="Mock English.", provider="groq",
        model="mock-model")
    try:
        tr_csv = client.post(
            "/api/v1/export",
            json={"format": "csv", "text": PARA, "include_translation": True})
        assert tr_csv.status_code == 200, tr_csv.text
        tr_rows = list(csv.DictReader(io.StringIO(tr_csv.json()["content"])))
        tr_row = next(
            r for r in tr_rows if r["section"] == "english_translation")
        assert tr_row["english_translation"] == "Mock English."
        assert tr_row["translation_model"] == "mock-model"

        tr_json = client.post(
            "/api/v1/export",
            json={"format": "json", "text": PARA, "include_translation": True})
        assert tr_json.status_code == 200, tr_json.text
        tr_parsed = json.loads(tr_json.json()["content"])
        assert tr_parsed["translation"]["available"] is True
        assert tr_parsed["translation"]["text"] == "Mock English."

        plain_json = json.loads(as_json.json()["content"])
        assert plain_json["translation"] is None  # opt-in only
    finally:
        analyze_api._resolve_translation = real_resolve

    assert client.post("/api/v1/export", json={"format": "xml", "text": PARA}).status_code == 422
    assert client.post("/api/v1/export", json={"format": "json", "text": "  "}).status_code == 422


def test_export_never_exposes_server_paths():
    resp = client.post("/api/v1/export", json={"format": "json", "text": PARA})
    content = resp.json()["content"]
    assert "Traceback" not in content
    assert "D:\\" not in content and "C:\\" not in content


def test_cors_allows_listed_origins_without_wildcard():
    assert "*" not in settings.frontend_origins
    resp = client.get(
        "/health", headers={"Origin": "http://localhost:3000"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_app_boots_and_serves_health():
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "app.main:app", "--port", "8123"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="D:\\HARSH\\projects\\Timepass\\NLP",
    )
    try:
        body = None
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=2) as res:
                    body = res.read().decode("utf-8")
                break
            except Exception:
                continue
        assert body is not None, "uvicorn did not serve /health in time"
        assert json.loads(body) == {"status": "ok"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
