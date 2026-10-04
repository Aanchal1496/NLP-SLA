# -*- coding: utf-8 -*-
"""Tests for the SILVER-trained spaCy NER (opt-in; baseline stays default).

Silver model: trained on synthetic template data (data/silver/*), NOT human
gold. Assertions check plumbing (offsets, labels, honest statuses), never
claim human-gold quality.
"""
import os

from fastapi.testclient import TestClient

from app.main import app
from app.services import ner_trained as svc

client = TestClient(app)

MODEL_OK = os.path.isdir(os.path.join("models", "ner", "model"))


def test_trained_status_shape():
    body = client.get("/api/v1/predict/ner-status").json()
    assert set(body) >= {"available"}
    if MODEL_OK:
        assert body["available"] is True
        assert body["method"] == "spacy_ner"
    else:
        assert body["available"] is False


def test_trained_metrics_shape():
    body = client.get("/api/v1/model/ner-metrics").json()
    assert set(body) >= {"available"}
    if MODEL_OK:
        assert body["available"] is True
        assert body["data_kind"].startswith("SILVER")
        assert body["silver_heldout"]["f1"] >= 0.0


def test_baseline_default_unchanged():
    # Default endpoint serves baseline spans; when IndicBERT NER is trained it
    # merges them (type "hybrid_baseline+indicbert_ner") instead of replacing.
    para = "दिनांक 12/05/2024 रोजी सुनावणी होईल।"
    body = client.post("/api/v1/predict/entities", json={"text": para}).json()
    assert body["model"]["type"] in ("hybrid_baseline", "hybrid_baseline+indicbert_ner")


def test_trained_predict_offsets_when_model_present():
    if not MODEL_OK:
        resp = client.post(
            "/api/v1/predict/entities-trained", json={"text": "test text"})
        assert resp.status_code == 503
        return
    # Template-style sentence (in-distribution for silver model).
    import csv
    import json as js
    with open("data/silver/ner_silver.csv", encoding="utf-8") as fh:
        row = next(iter(__import__("csv").DictReader(fh)))
    text = row["text"]
    resp = client.post(
        "/api/v1/predict/entities-trained", json={"text": text})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"]["type"] == "spacy_silver_ner"
    assert body["count"] == len(body["entities"])
    for ent in body["entities"]:
        assert text[ent["start"]:ent["end"]] == ent["text"]
        assert ent["method"] == "spacy_ner"
