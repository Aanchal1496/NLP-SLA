# -*- coding: utf-8 -*-
"""Frontend↔backend integration contract tests (no browser required).

Verifies the static dashboard honestly integrates with the FastAPI backend:
no fabricated endpoints/tokens/metrics, every DOM id the integration script
needs exists, every API path it calls is a real backend route, and the app
serves the page + script.
"""

import re

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HTML = "D:\\HARSH\\projects\\Timepass\\NLP\\frontend\\index.html"
JS = "D:\\HARSH\\projects\\Timepass\\NLP\\frontend\\nlp-integration.js"


def _html():
    with open(HTML, encoding="utf-8") as fh:
        return fh.read()


def _js():
    with open(JS, encoding="utf-8") as fh:
        return fh.read()


def test_frontend_files_exist():
    import os

    assert os.path.exists(HTML)
    assert os.path.exists(JS)


def test_no_fabricated_endpoints_or_metrics():
    src = _html()
    for banned in (
        "api.marathinlp.ai",
        "Bearer",
        "dev_token",
        "91.4%",
        "98.4%",
        "1,420",
        "naive_bayes",
        "devanagari_stemming",
        "document_type",
        "Mock Action Feedback",
    ):
        assert banned not in src, banned


def test_no_mock_fetch_responses_in_js():
    src = _js()
    assert "fetch(" in src
    for banned in ("Mock", "mockData", "fakeData", "TODO"):
        assert banned not in src, banned


def test_js_dom_ids_exist_in_html():
    src = _html()
    needed = set(re.findall(r"\$\('([A-Za-z][\w\-]*)'\)", _js()))
    needed |= set(re.findall(r"getElementById\('([A-Za-z][\w\-]*)'\)", _js()))
    created = {"nlp-analyze-status", "live-dataset-banner"}
    html_ids = set(re.findall(r'id="([A-Za-z][\w\-]*)"', src))
    missing = {i for i in needed if i not in html_ids and i not in created}
    assert not missing, missing


def test_js_api_paths_are_real_backend_routes():
    from app.main import app as live_app

    called = set(re.findall(r"""['"](/api/v1/[A-Za-z\-/]+)['"]""", _js()))
    assert called, "integration JS must call backend routes"
    routes = {
        r.path
        for r in live_app.routes
        if hasattr(r, "path") and r.path.startswith("/api/")
    }
    unknown = {p for p in called if p not in routes}
    assert not unknown, unknown


def test_backend_serves_frontend_and_script():
    page = client.get("/app/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "nlp-integration.js" in page.text
    script = client.get("/app/nlp-integration.js")
    assert script.status_code == 200
    assert "fetch(" in script.text


def test_analyze_button_and_upload_wired():
    src = _html()
    assert 'id="analyze-action-btn"' in src
    assert 'id="file-uploader"' in src
    assert 'id="marathi-input-area"' in src
    js = _js()
    assert "/api/v1/analyze-file" in js
    assert "FormData" in js
    assert "multipart" in js.lower() or "FormData" in js


def test_export_uses_real_endpoint():
    js = _js()
    assert "/api/v1/export" in js
    assert "Blob" in js


def test_nav_routes_resolve_to_router_pages():
    src = _html()
    js = _js()
    assert "initRouter" in js and "showPage" in js
    for route in ("#/home", "#/analyzer", "#/results", "#/dataset"):
        assert 'href="%s"' % route in src, route
    for page in ("home", "analyzer", "results", "dataset"):
        assert page in js


def test_disclaimer_and_status_hooks_present():
    js = _js()
    assert "does not provide legal advice" in js
    assert "/health" in js
    assert "data-backend-status" in _html()
