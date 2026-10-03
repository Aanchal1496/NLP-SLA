"""Trained spaCy NER wrapper (SILVER-data model, optional).

The baseline (app/services/ner.py) stays the default for
POST /predict/entities so existing contracts never break.
This module exposes the SILVER-trained spaCy pipeline at
<model_dir>/ner/model with method="spacy_ner":

- load_trained_ner(): cached load, raises ModelUnavailableError if missing.
- predict_trained(text): exact char offsets (text[start:end] == span text).
- trained_status(): honest availability dict for APIs.
- Silver caveat: trained on synthetic template data
  (data/silver/ner_silver.csv); template-held-out scores are inflated.
  Do not present as human-gold evaluation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from app.core.config import settings

MODEL_SUBDIR = os.path.join("ner", "model")
LABELS_FILE = os.path.join("ner", "ner_labels.json")
METRICS_FILE = os.path.join("ner", "ner_metrics.json")


class ModelUnavailableError(Exception):
    """Raised when no trained spaCy NER model can be loaded."""


@dataclass
class TrainedMention:
    text: str
    label: str
    start: int
    end: int
    method: str = "spacy_ner"


@dataclass
class TrainedResult:
    original_text: str
    entities: list[TrainedMention] = field(default_factory=list)


_cache: dict[str, object] = {}


def clear_trained_cache() -> None:
    _cache.clear()


def model_path(model_dir: str | None = None) -> str:
    base = model_dir or settings.model_dir
    return os.path.join(base, MODEL_SUBDIR)


def load_trained_ner(model_dir: str | None = None):
    key = model_dir or settings.model_dir
    if key in _cache:
        return _cache[key]
    path = model_path(model_dir)
    if not os.path.isdir(path):
        raise ModelUnavailableError(
            f"No trained NER model at '{path}'. Train one with: "
            "python -m training.make_ner_silver && "
            "python -m training.train_ner --data data/silver/ner_silver.csv "
            "--out-dir models/ner"
        )
    try:
        import spacy

        nlp = spacy.load(path)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Stored NER model at '{path}' could not be loaded ({exc})."
        ) from exc
    _cache[key] = nlp
    return nlp


def predict_trained(text: str, model_dir: str | None = None) -> TrainedResult:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    nlp = load_trained_ner(model_dir)
    doc = nlp(text)
    mentions: list[TrainedMention] = []
    for ent in doc.ents:
        s, e = int(ent.start_char), int(ent.end_char)
        if not (0 <= s < e <= len(text)):
            continue
        mentions.append(TrainedMention(
            text=text[s:e], label=str(ent.label_), start=s, end=e,
        ))
    mentions.sort(key=lambda m: (m.start, m.end))
    return TrainedResult(original_text=text, entities=mentions)


def trained_status(model_dir: str | None = None) -> dict:
    base = model_dir or settings.model_dir
    path = model_path(base)
    if not os.path.isdir(path):
        return {"available": False,
                "reason": f"No trained NER model at '{path}'. Baseline regex+gazetteer still serves /predict/entities."}
    labels: list = []
    try:
        with open(os.path.join(base, LABELS_FILE), encoding="utf-8") as fh:
            labels = json.load(fh)
    except Exception:
        labels = []
    return {"available": True, "path": path, "labels": labels,
            "data_kind": "SILVER synthetic -- NOT human gold",
            "method": "spacy_ner"}
