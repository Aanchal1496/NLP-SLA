"""IndicBERT intent inference (optional primary, TF-IDF stays as fallback).

Artifacts expected in <model_dir>/indicbert_intent/ (produced by
training/train_intent_indicbert.py):
    config.json + model weights + tokenizer + intent_labels.json + intent_metrics.json

- predict_indicbert(): softmax probabilities -> confidence_type="probability".
- predict_intent_auto(): tries IndicBERT first, falls back to the existing
  TF-IDF pipeline so /predict/intent never breaks when IndicBERT is absent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

SUBDIR = "indicbert_intent"
LABELS_FILE = "intent_labels.json"
METRICS_FILE = "intent_metrics.json"


class ModelUnavailableError(Exception):
    """Raised when no IndicBERT intent artifacts can be loaded."""


@dataclass
class IndicBertIntentPrediction:
    intent: str
    confidence_type: str = "probability"
    confidence: float | None = None
    low_confidence: bool = False
    model_name: str = "indicbert"
    labels: list[str] = field(default_factory=list)


_cache: dict[str, Any] = {}


def clear_indicbert_cache() -> None:
    _cache.clear()


def model_path(model_dir: str | None = None) -> str:
    return os.path.join(model_dir or settings.model_dir, SUBDIR)


def _paths(base: str | None = None) -> dict[str, str]:
    root = model_path(base)
    return {
        "root": root,
        "labels": os.path.join(root, LABELS_FILE),
        "metrics": os.path.join(root, METRICS_FILE),
    }


def load_indicbert_intent(model_dir: str | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    key = model_dir or settings.model_dir
    if key in _cache:
        return _cache[key]
    paths = _paths(model_dir)
    if not os.path.isdir(paths["root"]):
        raise ModelUnavailableError(
            f"No IndicBERT intent model at '{paths['root']}'. Train one with: "
            "python -m training.train_intent_indicbert --data <csv> "
            "--out-dir models/indicbert_intent"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(paths["root"])
        model = AutoModelForSequenceClassification.from_pretrained(paths["root"])
        model.eval()
    except Exception as exc:
        raise ModelUnavailableError(
            f"IndicBERT intent model at '{paths['root']}' could not be loaded ({exc})."
        ) from exc
    try:
        with open(paths["labels"], encoding="utf-8") as fh:
            labels = list(json.load(fh))
    except Exception:
        labels = [str(v) for v in getattr(model.config, "id2label", {}).values()]
    try:
        with open(paths["metrics"], encoding="utf-8") as fh:
            metrics = dict(json.load(fh))
    except Exception:
        metrics = {}
    bundle = {"model": model, "tokenizer": tokenizer, "labels": labels, "metrics": metrics}
    _cache[key] = bundle
    return bundle


def predict_indicbert(text: str, model_dir: str | None = None) -> IndicBertIntentPrediction:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    import torch

    bundle = load_indicbert_intent(model_dir)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    labels: list[str] = list(bundle["labels"])
    metrics: dict = dict(bundle["metrics"])
    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = logits.softmax(-1).cpu().tolist()
    best = max(range(len(probs)), key=lambda i: probs[i])
    id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}
    intent = str(id2label.get(best, labels[best] if best < len(labels) else best))
    conf = float(probs[best])
    base = str(metrics.get("base_model", "ai4bharat/indic-bert"))
    return IndicBertIntentPrediction(
        intent=intent,
        confidence_type="probability",
        confidence=conf,
        low_confidence=bool(conf < 0.5),
        model_name=f"indicbert:{base}",
        labels=labels or sorted(set(id2label.values())),
    )


def predict_intent_auto(text: str, model_dir: str | None = None):
    """Prefer IndicBERT; fall back to TF-IDF when IndicBERT is unavailable."""
    try:
        return predict_indicbert(text, model_dir)
    except ModelUnavailableError:
        from app.services import intent_classifier as tfidf_svc

        return tfidf_svc.predict_intent(text, model_dir)


def indicbert_status(model_dir: str | None = None) -> dict:
    paths = _paths(model_dir)
    if not os.path.isdir(paths["root"]):
        return {"available": False,
                "reason": f"No IndicBERT intent model at '{paths['root']}'. TF-IDF fallback still serves /predict/intent.",
                "path": paths["root"]}
    labels: list = []
    try:
        with open(paths["labels"], encoding="utf-8") as fh:
            labels = json.load(fh)
    except Exception:
        pass
    return {"available": True, "path": paths["root"], "labels": labels,
            "backend": "indicbert_sequence_classification"}
