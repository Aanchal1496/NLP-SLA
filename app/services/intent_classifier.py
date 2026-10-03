"""Intent-classification prediction service (route-independent).

The train/inference contract: training persists ONE sklearn ``Pipeline``
(TF-IDF vectorizer + classifier) to ``intent_pipeline.joblib``. Inference
loads that same object, so vectorization is identical by construction --
there is no separate inference-time vectorizer to drift.

Status: NO trained model exists yet (no annotated project dataset has been
supplied -- see ``data/README.md``). All entry points raise
:class:`ModelUnavailableError` with a remediation message until
``training/train_intent.py`` has produced artifacts. Nothing here fabricates
predictions: without artifacts, prediction refuses.

Confidence honesty:
- LogisticRegression pipelines expose ``predict_proba`` -> returned as
  ``confidence_type="probability"``. These are the model's own probabilities;
  NO post-hoc (Platt/isotonic) calibration is applied, and we do not claim
  otherwise.
- LinearSVC pipelines expose only ``decision_function`` -> returned as
  ``confidence_type="decision_score"`` (raw margin, NOT a probability).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import joblib

from app.core.config import settings
from app.services.marathi_preprocess import PreprocessOptions, clean_text

PIPELINE_FILENAME = "intent_pipeline.joblib"
LABELS_FILENAME = "intent_labels.json"
METRICS_FILENAME = "intent_metrics.json"

_TFIDF_CLEAN_OPTIONS = PreprocessOptions()

# Word-ish tokens of length >= 2, incl. the Devanagari block (letters AND
# combining marks, which bare ``\\w`` would split apart).
TOKEN_PATTERN = r"(?u)[A-Za-z0-9_ऀ-ॿ]{2,}"


class ModelUnavailableError(Exception):
    """Raised when no trained intent model can be loaded."""


def tfidf_preprocessor(text: str) -> str:
    """Per-document cleaning shared by training and inference.

    Top-level function (not a lambda) so pipelines referencing it survive a
    joblib save/load round-trip. Lowercasing affects Latin script only;
    Devanagari has no case and is unchanged.
    """
    return clean_text(text, _TFIDF_CLEAN_OPTIONS).lower()


def build_pipelines(random_state: int = 42) -> dict[str, Any]:
    """Construct the candidate TF-IDF + classifier pipelines (unfitted)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    vectorizer_kwargs = {
        "preprocessor": tfidf_preprocessor,
        "token_pattern": TOKEN_PATTERN,
        "ngram_range": (1, 2),
        "min_df": 1,
        "sublinear_tf": True,
    }
    return {
        "tfidf_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "tfidf_linearsvc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
    }


@dataclass
class ModelBundle:
    pipeline: Any
    labels: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_dir: str = ""


_bundle_cache: dict[str, ModelBundle] = {}


def clear_bundle_cache() -> None:
    _bundle_cache.clear()


def _artifact_paths(model_dir: str) -> dict[str, str]:
    return {
        "pipeline": os.path.join(model_dir, PIPELINE_FILENAME),
        "labels": os.path.join(model_dir, LABELS_FILENAME),
        "metrics": os.path.join(model_dir, METRICS_FILENAME),
    }


def load_model_bundle(model_dir: str | None = None) -> ModelBundle:
    """Load (and cache) the persisted pipeline + labels + metrics."""
    resolved = model_dir or settings.model_dir
    if resolved in _bundle_cache:
        return _bundle_cache[resolved]
    paths = _artifact_paths(resolved)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise ModelUnavailableError(
            "No trained intent model found "
            f"(missing in '{resolved}': {', '.join(missing)}). "
            "Train one with: "
            "python -m training.train_intent --data <dataset.csv> "
            "--text-col text --label-col intent"
        )
    try:
        pipeline = joblib.load(paths["pipeline"])
    except Exception as exc:
        raise ModelUnavailableError(
            f"Stored intent pipeline at '{paths['pipeline']}' could not be "
            f"loaded ({exc}). Retrain with training/train_intent.py."
        ) from exc
    try:
        with open(paths["labels"], encoding="utf-8") as fh:
            labels = json.load(fh)
        with open(paths["metrics"], encoding="utf-8") as fh:
            metrics = json.load(fh)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Intent model metadata in '{resolved}' is unreadable ({exc})."
        ) from exc
    bundle = ModelBundle(
        pipeline=pipeline, labels=list(labels), metrics=dict(metrics),
        model_dir=resolved,
    )
    _bundle_cache[resolved] = bundle
    return bundle


@dataclass
class IntentPrediction:
    intent: str
    confidence_type: str  # "probability" | "decision_score" | "none"
    confidence: float | None
    low_confidence: bool
    model_name: str
    labels: list[str] = field(default_factory=list)


def predict_intent(text: str, model_dir: str | None = None) -> IntentPrediction:
    """Predict the intent of one document string using persisted artifacts."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    bundle = load_model_bundle(model_dir)
    try:
        predicted = bundle.pipeline.predict([text])[0]
    except Exception as exc:
        raise ModelUnavailableError(
            f"Loaded intent pipeline failed at predict time ({exc})."
        ) from exc
    intent = str(predicted)
    clf_step = bundle.pipeline.named_steps.get("clf", bundle.pipeline)
    confidence_type = "none"
    confidence: float | None = None
    if hasattr(clf_step, "predict_proba"):
        try:
            proba = bundle.pipeline.predict_proba([text])[0]
            confidence_type = "probability"
            confidence = float(max(proba))
        except Exception:
            confidence_type = "none"
            confidence = None
    elif hasattr(bundle.pipeline, "decision_function"):
        try:
            scores = bundle.pipeline.decision_function([text])
            confidence_type = "decision_score"
            confidence = float(max(scores.ravel()))
        except Exception:
            confidence_type = "none"
            confidence = None
    low_confidence = bool(
        confidence_type == "probability" and (confidence or 0.0) < 0.5
    )
    model_name = str(bundle.metrics.get("selected_model", type(clf_step).__name__))
    return IntentPrediction(
        intent=intent,
        confidence_type=confidence_type,
        confidence=confidence,
        low_confidence=low_confidence,
        model_name=model_name,
        labels=list(bundle.labels),
    )
