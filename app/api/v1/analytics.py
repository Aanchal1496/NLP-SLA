from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.services import analytics as analytics_svc
from app.services import ner as ner_svc
from training.dataset import DatasetError

router = APIRouter()
flat_router = APIRouter()


class NerGoldSpan(BaseModel):
    label: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @field_validator("end")
    @classmethod
    def _ordered(cls, value: int, info) -> int:
        if info.data.get("start") is not None and value < info.data["start"]:
            raise ValueError("end must be >= start.")
        return value


class NerEvalItem(BaseModel):
    text: str
    gold: list[NerGoldSpan] = []

    @field_validator("text")
    @classmethod
    def _is_str(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("text must be a string.")
        return value


class NerEvalRequest(BaseModel):
    items: list[NerEvalItem]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, value: list[NerEvalItem]) -> list[NerEvalItem]:
        if not value:
            raise ValueError("items must not be empty.")
        return value


@router.get("/dataset", response_model=dict)
def dataset_analytics() -> dict:
    """Chart-ready stats for the configured project dataset file."""
    return _dataset_stats()


@flat_router.get("/dataset/stats", response_model=dict)
def dataset_stats_flat() -> dict:
    """Required integration path (alias of ``/analytics/dataset``)."""
    return _dataset_stats()


def _dataset_stats() -> dict:
    try:
        return analytics_svc.analyze_dataset_file(
            settings.dataset_path,
            settings.dataset_text_col,
            settings.dataset_label_col,
            settings.dataset_entity_col,
        )
    except DatasetError as exc:
        return {"available": False, "reason": str(exc)}


@flat_router.get("/dataset/records", response_model=dict)
def dataset_records(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    """Paginated raw dataset records (honest unavailable status if missing)."""
    try:
        return analytics_svc.read_dataset_records(
            settings.dataset_path,
            settings.dataset_text_col,
            settings.dataset_label_col,
            page,
            page_size,
        )
    except DatasetError as exc:
        return {"available": False, "reason": str(exc)}


@router.get("/intent-metrics", response_model=dict)
def intent_metrics() -> dict:
    """Persisted held-out intent metrics, or an honest unavailable status."""
    return _model_metrics()


@flat_router.get("/model/metrics", response_model=dict)
def model_metrics_flat() -> dict:
    """Required integration path (alias of ``/analytics/intent-metrics``)."""
    return _model_metrics()


def _model_metrics() -> dict:
    path = os.path.join(settings.model_dir, "intent_metrics.json")
    if not os.path.exists(path):
        return {
            "available": False,
            "reason": (
                "No intent evaluation has been performed: no trained model "
                "exists. Train with: python -m training.train_intent "
                "--data <dataset.csv> --text-col text --label-col intent"
            ),
        }
    try:
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stored intent metrics unreadable: {exc}",
        )
    return {"available": True, **stored}


@router.post("/ner", response_model=dict)
def ner_evaluation(payload: NerEvalRequest) -> dict:
    """Entity-level P/R/F1 for caller-supplied gold spans (exact matching)."""
    gold_total = sum(len(item.gold) for item in payload.items)
    if gold_total == 0:
        return {
            "evaluated": False,
            "reason": "No gold entity spans supplied; nothing to compare.",
            "matching_criteria": (
                "exact match on (start, end, label) against the original text"
            ),
        }
    predicted: list[ner_svc.EntityMention] = []
    gold: list[ner_svc.EntityMention] = []
    for item in payload.items:
        for mention in ner_svc.extract_entities(item.text).entities:
            predicted.append(mention)
        for span in item.gold:
            gold.append(
                ner_svc.EntityMention(
                    text=item.text[span.start : span.end],
                    label=span.label,
                    start=span.start,
                    end=span.end,
                    method="gold",
                )
            )
    scores = ner_svc.entity_prf(predicted, gold)
    return {
        "evaluated": True,
        "matching_criteria": (
            "exact match on (start, end, label) against the original text"
        ),
        "predicted_spans": len(predicted),
        "gold_spans": len(gold),
        **scores,
    }
