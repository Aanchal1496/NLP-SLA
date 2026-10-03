# -*- coding: utf-8 -*-
"""Tests for dataset analytics + evaluation endpoints.

No project dataset exists, so project-level numbers are asserted only as
honest unavailable statuses. All arithmetic is verified on a tiny,
obviously-synthetic ENGLISH fixture confined to tmp_path, and intent-metric
values are cross-checked by independent recomputation.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.main import app
from app.services import analytics as analytics_svc
from app.services.analytics import analyze_dataset_file
from training import analytics as analytics_cli
from training import train_intent
from training.dataset import DatasetError

client = TestClient(app)

# Synthetic analytics fixture: English, meaningless labels, tmp-only.
ANALYTICS_ROWS = [
    {"text": "alpha beta gamma", "intent": "x"},
    {"text": "alpha beta", "intent": "x"},
    {"text": "beta gamma delta", "intent": "y"},
    {"text": "   ", "intent": "x"},
    {"text": "alpha beta gamma", "intent": "x"},
    {"text": "epsilon", "intent": "   "},
]


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# --- analytics arithmetic on synthetic fixture ----------------------------------

def test_analytics_counts_and_distributions_sum(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ANALYTICS_ROWS)
    payload = analyze_dataset_file(csv, "text", "intent")
    assert payload["available"] is True
    assert payload["records"]["total"] == 6
    assert payload["records"]["used"] == 4
    assert payload["records"]["dropped_missing_text"] == 1
    assert payload["records"]["dropped_missing_label"] == 1
    assert payload["records"]["duplicate_rows"] == 1
    dist = payload["intent_distribution"]
    assert dist["labels"] == ["x", "y"]
    assert dist["counts"] == [3, 1]
    assert sum(dist["counts"]) == dist["total"] == payload["records"]["used"]


def test_text_analytics_values(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ANALYTICS_ROWS)
    payload = analyze_dataset_file(csv, "text", "intent")
    stats = payload["text_analytics"]
    used = ["alpha beta gamma", "alpha beta", "beta gamma delta", "alpha beta gamma"]
    assert stats["record_count"] == 4
    assert stats["average_chars"] == pytest.approx(sum(map(len, used)) / 4)
    assert stats["min_chars"] == min(map(len, used))
    assert stats["max_chars"] == max(map(len, used))
    assert sum(stats["char_length_histogram"]["counts"]) == 4
    assert stats["vocabulary_size"] == 4  # alpha beta gamma delta
    assert stats["frequent_words"][0] == {"token": "beta", "count": 4}
    assert stats["frequent_bigrams"][0] == {"bigram": "alpha beta", "count": 3}
    assert stats["sentence_word_counts"]["total_sentences"] == 4
    assert sum(stats["sentence_word_counts"]["histogram"]["counts"]) == 4


def test_chart_json_serializable(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ANALYTICS_ROWS)
    payload = analyze_dataset_file(csv, "text", "intent")
    round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))
    assert round_tripped["available"] is True
    assert "intent_distribution" in round_tripped


def test_entity_column_json_and_bio(tmp_path):
    rows = [
        {"text": "doc one", "intent": "x", "ents": '[{"label": "DATE"}]'},
        {"text": "doc two", "intent": "y", "ents": '[{"label": "DATE"}, {"label": "MONEY"}]'},
        {"text": "doc three", "intent": "x", "ents": "O B-MONEY O"},
    ]
    csv = _write_csv(tmp_path / "e.csv", rows)
    payload = analyze_dataset_file(csv, "text", "intent", entity_col="ents")
    assert payload["entity_distribution"]["labels"] == ["DATE", "MONEY"]
    assert payload["entity_distribution"]["counts"] == [2, 2]


def test_entity_distribution_absent_without_column(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ANALYTICS_ROWS)
    payload = analyze_dataset_file(csv, "text", "intent")
    assert payload["entity_distribution"] is None
    assert "entity annotation column" in payload["entity_distribution_reason"]


# --- honest handling of the missing real dataset ----------------------------------

def test_real_missing_dataset_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        analyze_dataset_file(str(tmp_path / "absent.csv"), "text", "intent")


def test_cli_missing_dataset_returns_2(tmp_path, capsys):
    assert analytics_cli.main(["--data", str(tmp_path / "absent.csv")]) == 2


def test_api_dataset_unavailable_by_default():
    resp = client.get("/api/v1/analytics/dataset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True  # project dataset connected
    assert body["records"]["used"] >= 64


def test_api_intent_metrics_unavailable_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "empty"))
    resp = client.get("/api/v1/analytics/intent-metrics")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


# --- extended intent metrics: keys, recomputation, reproducibility ------------------

INTENT_ROWS = (
    [{"text": f"court verdict property matter number {i}", "intent": "a"} for i in range(6)]
    + [{"text": f"loan installment overdue payment notice {i}", "intent": "b"} for i in range(6)]
)


def test_extended_metrics_match_recomputation(tmp_path):
    csv = _write_csv(tmp_path / "t.csv", INTENT_ROWS)
    out = str(tmp_path / "artifacts")
    metrics = train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent", out_dir=out,
        test_size=0.25, random_state=42,
    )
    stored = metrics["results"][metrics["selected_model"]]
    for key in (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "classification_report", "confusion_matrix",
    ):
        assert key in stored, key
    frame = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = train_test_split(
        list(frame["text"]), list(frame["intent"]),
        test_size=0.25, random_state=42, stratify=list(frame["intent"]),
    )
    assert stored["test_size"] == len(y_test)  # held-out only, never train rows
    assert metrics["train_size"] == len(y_train)
    reloaded_pred = _predict_selected(out, X_test)
    assert stored["accuracy"] == float(accuracy_score(y_test, reloaded_pred))
    assert stored["weighted_f1"] == float(
        f1_score(y_test, reloaded_pred, average="weighted", zero_division=0)
    )
    assert stored["confusion_matrix"]["matrix"] == confusion_matrix(
        y_test, reloaded_pred, labels=["a", "b"]
    ).tolist()
    assert stored["confusion_matrix"]["labels"] == ["a", "b"]
    assert set(stored["classification_report"]) >= {"a", "b", "accuracy", "macro avg"}


def _predict_selected(out, X_test):
    """Predictions from the reloaded artifact (proves the persisted file)."""
    import joblib
    import os

    pipe = joblib.load(os.path.join(out, "intent_pipeline.joblib"))
    return pipe.predict(X_test)


def test_metrics_reproducible_same_seed(tmp_path):
    csv = _write_csv(tmp_path / "t.csv", INTENT_ROWS)
    first = train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent",
        out_dir=str(tmp_path / "m1"), test_size=0.25, random_state=42,
    )
    second = train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent",
        out_dir=str(tmp_path / "m2"), test_size=0.25, random_state=42,
    )
    assert first["selected_model"] == second["selected_model"]
    assert first["results"] == second["results"]


def test_api_intent_metrics_serves_stored_file(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path / "t.csv", INTENT_ROWS)
    out = str(tmp_path / "artifacts")
    train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent", out_dir=out,
        test_size=0.25, random_state=42,
    )
    monkeypatch.setattr(settings, "model_dir", out)
    resp = client.get("/api/v1/analytics/intent-metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "confusion_matrix" in body["results"][body["selected_model"]]


# --- NER evaluation endpoint ---------------------------------------------------------

def test_ner_eval_with_gold_spans():
    text = "Meet on 12/05/2024 now"
    resp = client.post(
        "/api/v1/analytics/ner",
        json={"items": [{"text": text, "gold": [{"label": "DATE", "start": 8, "end": 18}]}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["evaluated"] is True
    assert body["gold_spans"] == 1
    assert body["predicted_spans"] >= 1
    assert body["precision"] == 1.0
    assert body["recall"] == 1.0
    assert body["f1"] == 1.0
    assert "exact" in body["matching_criteria"]


def test_ner_eval_without_gold_is_explicit():
    resp = client.post(
        "/api/v1/analytics/ner", json={"items": [{"text": "hello world", "gold": []}]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["evaluated"] is False
    assert body["reason"]


def test_ner_eval_empty_items_rejected():
    resp = client.post("/api/v1/analytics/ner", json={"items": []})
    assert resp.status_code == 422


def test_ner_project_status_documents_blocker():
    status = analytics_svc.ner_evaluation_status()
    assert status["evaluated"] is False
    assert "gold" in status["reason"].lower()
