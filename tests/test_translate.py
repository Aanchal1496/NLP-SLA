# -*- coding: utf-8 -*-
"""Tests for clause-wise analysis + GROQ translation (mocked, no network)."""
import json

from fastapi.testclient import TestClient

from app.main import app
from app.services import analysis as analysis_svc
from app.services import translate as translate_svc

client = TestClient(app)


def test_split_paragraphs():
    assert analysis_svc.split_paragraphs("a\n\nb\n\n\nc") == ["a", "b", "c"]
    assert analysis_svc.split_paragraphs("single") == ["single"]
    assert analysis_svc.split_paragraphs("  \n\n  ") == []


def test_split_single_newline_pdf_text_into_clauses():
    # PyMuPDF-style single-newline page text still gets clause-wise output.
    assert analysis_svc.split_paragraphs("line1\nline2\nline3") == [
        "line1", "line2", "line3"]


def test_version_endpoint_reports_build():
    from app.main import BUILD
    body = client.get("/version").json()
    assert body["build"] == BUILD
    assert "groq-translate" in body["features"]


def test_analyze_returns_clauses_and_clean_text():
    text = "First para here.\n\nSecond para here."
    body = client.post("/api/v1/analyze", json={"text": text}).json()
    assert len(body["clauses"]) == 2
    assert body["clauses"][0]["index"] == 0
    assert body["clauses"][0]["original"] == "First para here."
    assert isinstance(body["clean_text"], str) and body["clean_text"]
    assert body["translation"] is None  # separate /translate call


def test_clause_offsets_relative_to_clause():
    body = client.post(
        "/api/v1/analyze",
        json={"text": "Hello world.\n\nDate 12/05/2024 here."}).json()
    assert len(body["clauses"]) == 2
    for clause in body["clauses"]:
        for ent in clause["entities"]:
            assert clause["original"][ent["start"]:ent["end"]] == ent["text"]


def test_translate_no_key_returns_unavailable(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "")
    body = client.post(
        "/api/v1/translate", json={"text": "hello world"}).json()
    assert body["available"] is False
    assert "API key" in body["reason"]


def test_translate_blank_rejected():
    resp = client.post("/api/v1/translate", json={"text": "   "})
    assert resp.status_code == 422


def test_translate_success_mocked(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(settings, "groq_model", "test-model")

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Hello."}}]}

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    body = client.post(
        "/api/v1/translate", json={"text": "test text"}).json()
    assert body == {"available": True, "text": "Hello.", "provider": "groq",
                    "model": "test-model", "reason": ""}


def test_translate_api_error_becomes_unavailable(monkeypatch):
    import time

    from app.core.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    class FakeResp:
        status_code = 429
        headers = {}

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    body = client.post(
        "/api/v1/translate", json={"text": "test text"}).json()
    assert body["available"] is False
    assert "rate-limited" in body["reason"]


def test_translate_retries_429_then_succeeds(monkeypatch):
    import time

    from app.core.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(settings, "groq_model", "test-model")
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    calls = {"n": 0}

    class FlakyResp:
        status_code = 429
        headers = {}

    class GoodResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Hello."}}]}

    import httpx

    def fake_post(*a, **k):
        calls["n"] += 1
        return FlakyResp() if calls["n"] == 1 else GoodResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    body = client.post(
        "/api/v1/translate", json={"text": "test text"}).json()
    assert body["available"] is True
    assert body["text"] == "Hello."
    assert calls["n"] == 2


def test_early_stop_splits_and_stitches(monkeypatch):
    import time

    from app.core.config import settings
    from app.services import translate as svc
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    bodies = []

    class Resp:
        def __init__(self, text, finish="stop"):
            self.status_code = 200
            self._text = text
            self._finish = finish

        def json(self):
            return {"choices": [{"message": {"content": self._text},
                                 "finish_reason": self._finish}]}

    import httpx

    def fake_post(*a, **k):
        bodies.append(k["json"])
        content = k["json"]["messages"][-1]["content"]
        if len(content) > 700:
            return Resp("Short title only.")  # stubborn on big pieces
        return Resp("FULL translation of <" + content[:20] + ">")

    monkeypatch.setattr(httpx, "post", fake_post)
    chunk = "word " * 200  # ~1000 chars, trips the short-output guard
    out = svc._translate_chunk(chunk, "gsk_test", "m", 5.0)
    # Split path: NO "continue/stopped early" follow-up, halves translated.
    assert len(bodies) == 3  # whole + 2 halves
    assert all("stopped early" not in m["content"]
               for b in bodies for m in b["messages"]
               if m["role"] == "user")
    assert out.count("FULL translation") == 2
    assert "Short title only." not in out


def test_length_finish_retries_with_bigger_budget(monkeypatch):
    import time

    from app.core.config import settings
    from app.services import translate as svc
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    budgets = []

    class Resp:
        def __init__(self, text, finish):
            self.status_code = 200
            self._text = text
            self._finish = finish

        def json(self):
            return {"choices": [{"message": {"content": self._text},
                                 "finish_reason": self._finish}]}

    import httpx

    def fake_post(*a, **k):
        budgets.append(k["json"]["max_tokens"])
        if len(budgets) == 1:
            return Resp("cut off mid senten", "length")
        return Resp("cut off mid sentence, now complete and long " + "y" * 300,
                    "stop")

    monkeypatch.setattr(httpx, "post", fake_post)
    out = svc._translate_chunk("x" * 900, "gsk_test", "m", 5.0)
    assert budgets[1] > budgets[0]
    assert "now complete" in out


def test_long_text_is_chunked(monkeypatch):
    import time

    from app.core.config import settings
    from app.services import translate as svc
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(
        svc, "_translate_chunk",
        lambda chunk, *a, **k: seen.append(chunk) or ("EN:" + chunk[:10]))
    long_text = "\n\n".join("paragraph %d with some marathi words here" % i
                            for i in range(60))
    assert len(long_text) > 2500
    result = svc.translate_to_english(long_text)
    assert result.available is True
    assert len(seen) > 1
    assert result.chunks == len(seen)
