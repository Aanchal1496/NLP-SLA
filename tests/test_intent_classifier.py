# -*- coding: utf-8 -*-
"""Tests for intent classification: validation, pipeline plumbing, API.

No project dataset exists (see data/README.md), so NO test trains on or
claims project data. The only fitted models here use an obviously-synthetic
ENGLISH fixture confined to tmp_path, solely to verify save/load mechanics
and that metrics equal independently recomputed values. Nothing synthetic is
written to models/ or reported as a project result.
"""

import json
import os

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.main import app
from app.services import intent_classifier as svc
from app.services.intent_classifier import (
    ModelUnavailableError,
    build_pipelines,
    load_model_bundle,
    predict_intent,
)
from training import train_intent
from training.dataset import (
    DatasetError,
    UnsuitableDatasetError,
    decide_stratify,
    load_dataset,
    validate_dataset,
)

client = TestClient(app)

# Synthetic plumbing fixture: English, meaningless labels, tmp-only.
SYNTH_TEXTS_A = [
    "the court issued a property verdict today",
    "property dispute verdict announced by the court",
    "court verdict on the land property case",
    "judge announced verdict in property matter",
    "property case verdict delivered this morning",
]
SYNTH_TEXTS_B = [
    "loan installment payment overdue notice",
    "bank sent overdue loan payment reminder",
    "installment of the loan remains overdue",
    "overdue notice for missed loan installment",
    "payment reminder loan installment overdue",
]


@pytest.fixture(autouse=True)
def _clear_cache():
    svc.clear_bundle_cache()
    yield
    svc.clear_bundle_cache()


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# --- dataset validation -------------------------------------------------------

def test_missing_file_is_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(str(tmp_path / "nope.csv"), "text", "intent")


def test_missing_columns_list_available(tmp_path):
    csv = _write_csv(tmp_path / "d.csv", [{"text": "hello"}])
    with pytest.raises(DatasetError, match="Available columns"):
        load_dataset(csv, "text", "intent")


def test_blank_rows_dropped_and_counted():
    frame = pd.DataFrame(
        [
            {"text": "doc one", "intent": "a"},
            {"text": "   ", "intent": "a"},
            {"text": "doc two", "intent": "   "},
            {"text": "doc three", "intent": "b"},
            {"text": "doc four", "intent": "b"},
            {"text": "doc five", "intent": "b"},
            {"text": "doc six", "intent": "a"},
            {"text": "doc seven", "intent": "b"},
        ]
    )
    report = validate_dataset(frame, "inline", "text", "intent")
    assert report.rows_total == 8
    assert report.rows_dropped_missing_text == 1
    assert report.rows_dropped_missing_label == 1
    assert report.rows_used == 6
    assert report.class_frequencies == {"a": 2, "b": 4}


def test_insufficient_classes_flagged_and_stratify_off():
    frame = pd.DataFrame(
        [{"text": f"doc {i}", "intent": "a"} for i in range(6)]
        + [{"text": "lone doc", "intent": "b"}]
    )
    report = validate_dataset(frame, "inline", "text", "intent")
    assert report.insufficient_classes == ["b"]
    assert report.stratify_supported is False


def test_stratify_decision_rule():
    assert decide_stratify({"a": 2, "b": 2}) is True
    assert decide_stratify({"a": 5, "b": 1}) is False
    assert decide_stratify({}) is False


def test_single_label_dataset_refused():
    frame = pd.DataFrame([{"text": f"doc {i}", "intent": "only"} for i in range(8)])
    with pytest.raises(UnsuitableDatasetError, match="at least 2 distinct labels"):
        validate_dataset(frame, "inline", "text", "intent")


def test_tiny_dataset_refused():
    frame = pd.DataFrame(
        [{"text": "one", "intent": "a"}, {"text": "two", "intent": "b"}]
    )
    with pytest.raises(UnsuitableDatasetError, match="usable rows"):
        validate_dataset(frame, "inline", "text", "intent")


def test_cli_missing_dataset_returns_2(tmp_path, capsys):
    code = train_intent.main(
        ["--data", str(tmp_path / "absent.csv"), "--out-dir", str(tmp_path / "m")]
    )
    assert code == 2


def test_cli_single_class_returns_3(tmp_path):
    csv = _write_csv(
        tmp_path / "d.csv",
        [{"text": f"doc {i}", "intent": "only"} for i in range(8)],
    )
    code = train_intent.main(["--data", csv, "--out-dir", str(tmp_path / "m")])
    assert code == 3
    assert not os.path.exists(os.path.join(str(tmp_path / "m"), "intent_pipeline.joblib"))


# --- pipeline plumbing on synthetic fixture (tmp-only) ---------------------------

def _synthetic_frame():
    rows = [{"text": t, "intent": "label_a"} for t in SYNTH_TEXTS_A]
    rows += [{"text": t, "intent": "label_b"} for t in SYNTH_TEXTS_B]
    return pd.DataFrame(rows)


def test_pipelines_fit_predict_and_roundtrip(tmp_path):
    frame = _synthetic_frame()
    pipes = build_pipelines(random_state=42)
    assert set(pipes) == {"tfidf_logreg", "tfidf_linearsvc"}
    for name, pipe in pipes.items():
        pipe.fit(list(frame["text"]), list(frame["intent"]))
        pred = pipe.predict(["court verdict on property"])
        assert pred[0] in ("label_a", "label_b")
        path = str(tmp_path / f"{name}.joblib")
        joblib.dump(pipe, path)
        reloaded = joblib.load(path)
        assert list(reloaded.predict(["court verdict on property"])) == list(pred)


def test_train_persist_compare_and_metrics_honest(tmp_path):
    csv = _write_csv(
        tmp_path / "synth.csv",
        [{"text": t, "intent": "label_a"} for t in SYNTH_TEXTS_A]
        + [{"text": t, "intent": "label_b"} for t in SYNTH_TEXTS_B],
    )
    out = str(tmp_path / "artifacts")
    metrics = train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent", out_dir=out,
        test_size=0.2, random_state=42,
    )
    assert os.path.exists(os.path.join(out, "intent_pipeline.joblib"))
    with open(os.path.join(out, "intent_metrics.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    # Independently recompute from the same reproducible split.
    frame = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = train_test_split(
        list(frame["text"]), list(frame["intent"]),
        test_size=0.2, random_state=42, stratify=list(frame["intent"]),
    )
    pipes = build_pipelines(random_state=42)
    pipes[stored["selected_model"]].fit(X_train, y_train)
    pred = pipes[stored["selected_model"]].predict(X_test)
    assert stored["results"][stored["selected_model"]]["accuracy"] == (
        float(accuracy_score(y_test, pred))
    )
    assert stored["results"][stored["selected_model"]]["macro_f1"] == (
        float(f1_score(y_test, pred, average="macro", zero_division=0))
    )
    assert metrics["selected_model"] == stored["selected_model"]
    assert stored["stratify_used"] is True
    # End-to-end prediction from the persisted artifacts.
    got = predict_intent("court verdict on property", model_dir=out)
    assert got.intent in ("label_a", "label_b")
    assert got.labels == ["label_a", "label_b"]
    if got.confidence_type == "probability":
        assert 0.0 <= got.confidence <= 1.0


def test_linearsvc_confidence_labelled_decision_score(tmp_path):
    pipes = build_pipelines(random_state=42)
    pipe = pipes["tfidf_linearsvc"]
    frame = _synthetic_frame()
    pipe.fit(list(frame["text"]), list(frame["intent"]))
    out = str(tmp_path / "artifacts")
    os.makedirs(out)
    joblib.dump(pipe, os.path.join(out, "intent_pipeline.joblib"))
    with open(os.path.join(out, "intent_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(["label_a", "label_b"], fh)
    with open(os.path.join(out, "intent_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"selected_model": "tfidf_linearsvc"}, fh)
    got = predict_intent("loan overdue installment", model_dir=out)
    assert got.confidence_type == "decision_score"
    assert isinstance(got.confidence, float)


# --- unavailable / invalid handling -------------------------------------------------

def test_predict_without_artifacts_raises_helpful_error(tmp_path):
    with pytest.raises(ModelUnavailableError, match="train_intent"):
        predict_intent("some text", model_dir=str(tmp_path / "empty"))


def test_predict_empty_text_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        predict_intent("   ", model_dir=str(tmp_path))
    with pytest.raises(TypeError):
        predict_intent(None, model_dir=str(tmp_path))


def test_api_unavailable_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "empty"))
    resp = client.post("/api/v1/predict/intent", json={"text": "some text"})
    assert resp.status_code == 503


def test_api_empty_text_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "empty"))
    resp = client.post("/api/v1/predict/intent", json={"text": "   "})
    assert resp.status_code == 422


def test_api_success_from_persisted_artifacts(tmp_path, monkeypatch):
    csv = _write_csv(
        tmp_path / "synth.csv",
        [{"text": t, "intent": "label_a"} for t in SYNTH_TEXTS_A]
        + [{"text": t, "intent": "label_b"} for t in SYNTH_TEXTS_B],
    )
    out = str(tmp_path / "artifacts")
    train_intent.train_and_persist(
        data=csv, text_col="text", label_col="intent", out_dir=out,
        test_size=0.2, random_state=42,
    )
    monkeypatch.setattr(settings, "model_dir", out)
    resp = client.post(
        "/api/v1/predict/intent", json={"text": "court verdict on property"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] in ("label_a", "label_b")
    assert body["model"]["labels"] == ["label_a", "label_b"]
    assert body["confidence_type"] in ("probability", "decision_score", "none")
