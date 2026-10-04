"""Combined document analysis (route-independent).

Runs preprocessing, intent classification, and NER over one input and
assembles the single response used by ``POST /api/v1/analyze``,
``POST /api/v1/analyze-file``, and ``POST /api/v1/export``.

Graceful degradation: NER is always available (baseline needs no model);
the intent model is usually absent (no training data yet), in which case
``intent`` is ``None`` and a warning is recorded instead of failing the
whole analysis. NER offsets always refer to the ORIGINAL text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.services import ner as ner_svc
from app.services import ner_indicbert as ner_ib
from app.services.intent_classifier import IntentPrediction, ModelUnavailableError
from app.services.intent_indicbert import IndicBertIntentPrediction, predict_intent_auto
from app.services.intent_indicbert import ModelUnavailableError as IndicBertUnavailable
from app.services.marathi_preprocess import (
    PreprocessOptions,
    PreprocessResult,
    preprocess_marathi_text,
)

WARNING_INTENT_UNAVAILABLE = "intent-unavailable"
WARNING_LOW_CONFIDENCE = "low-confidence-intent"
WARNING_NO_ENTITIES = "no-entities-detected"


@dataclass
class ClauseAnalysis:
    """One paragraph clause: offsets are relative to the clause text."""

    index: int
    original: str
    clean: str
    intent: Any = None
    entities: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CombinedAnalysis:
    original_text: str = ""
    preprocessed: PreprocessResult = field(default_factory=PreprocessResult)
    intent: Any = None
    entities: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clauses: list[ClauseAnalysis] = field(default_factory=list)

    @property
    def entity_counts(self) -> dict[str, int]:
        return dict(Counter(m.label for m in self.entities))

    @property
    def clean_text(self) -> str:
        return self.preprocessed.processed_text


def split_paragraphs(text: str) -> list[str]:
    """Split normalized text into paragraph clauses (blank-line split).

    Scanned-layout PDF extraction often yields single newlines with no blank
    lines; in that case each non-empty line becomes its own clause so uploads
    still get clause-wise output instead of one giant block.
    """
    parts = [p.strip() for p in text.split("\n\n")]
    paras = [p for p in parts if p]
    if len(paras) <= 1:
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) > 1:
            return lines
    return paras


def analyze_text(
    text: str, options: PreprocessOptions | None = None
) -> CombinedAnalysis:
    """Run the full pipeline on one document string."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    preprocessed = preprocess_marathi_text(text, options or PreprocessOptions())
    warnings: list[str] = []
    intent: IntentPrediction | IndicBertIntentPrediction | None = None
    try:
        intent = predict_intent_auto(text)
        if intent.low_confidence:
            warnings.append(WARNING_LOW_CONFIDENCE)
    except (ModelUnavailableError, IndicBertUnavailable):
        warnings.append(WARNING_INTENT_UNAVAILABLE)
    entities = ner_ib.predict_merged(text)
    if not entities:
        warnings.append(WARNING_NO_ENTITIES)
    clauses: list[ClauseAnalysis] = []
    for idx, para in enumerate(split_paragraphs(text)):
        clause_warnings: list[str] = []
        try:
            clause_pre = preprocess_marathi_text(
                para, options or PreprocessOptions())
            clause_clean = clause_pre.processed_text
        except Exception:
            clause_clean = para
        clause_intent: IntentPrediction | IndicBertIntentPrediction | None = None
        try:
            clause_intent = predict_intent_auto(para)
            if clause_intent.low_confidence:
                clause_warnings.append(WARNING_LOW_CONFIDENCE)
        except (ModelUnavailableError, IndicBertUnavailable):
            clause_warnings.append(WARNING_INTENT_UNAVAILABLE)
        except Exception:
            clause_warnings.append(WARNING_INTENT_UNAVAILABLE)
        clause_entities = ner_ib.predict_merged(para)
        if not clause_entities:
            clause_warnings.append(WARNING_NO_ENTITIES)
        clauses.append(ClauseAnalysis(
            index=idx, original=para, clean=clause_clean,
            intent=clause_intent, entities=clause_entities,
            warnings=clause_warnings,
        ))
    return CombinedAnalysis(
        original_text=text,
        preprocessed=preprocessed,
        intent=intent,
        entities=entities,
        warnings=warnings,
        clauses=clauses,
    )
