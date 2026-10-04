from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator

from app.services import ner as svc
from app.services import ner_indicbert as ib_svc
from app.services import ner_trained as trained_svc

router = APIRouter()
flat_router = APIRouter()
trained_router = APIRouter()


class EntitiesRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value


class EntityResult(BaseModel):
    text: str
    label: str
    start: int
    end: int
    method: str


class NerModelInfo(BaseModel):
    name: str
    type: str


class EntitiesResponse(BaseModel):
    entities: list[EntityResult]
    count: int
    model: NerModelInfo


@router.post("/entities", response_model=EntitiesResponse)
def predict_entities(payload: EntitiesRequest) -> EntitiesResponse:
    return _respond(payload)


@flat_router.post("/extract-entities", response_model=EntitiesResponse)
def extract_entities_flat(payload: EntitiesRequest) -> EntitiesResponse:
    """Required integration path (alias of ``/predict/entities``)."""
    return _respond(payload)


def _respond(payload: EntitiesRequest) -> EntitiesResponse:
    try:
        mentions = ib_svc.predict_merged(payload.text)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    model_type = "hybrid_baseline+indicbert_ner" if any(
        m.method == "indicbert_ner" for m in mentions
    ) else "hybrid_baseline"
    return EntitiesResponse(
        entities=[
            EntityResult(
                text=m.text, label=m.label, start=m.start, end=m.end,
                method=m.method,
            )
            for m in mentions
        ],
        count=len(mentions),
        model=NerModelInfo(name="marathi_legal_baseline_v1", type=model_type),
    )


@trained_router.post("/entities-trained", response_model=EntitiesResponse)
def predict_entities_trained(payload: EntitiesRequest) -> EntitiesResponse:
    """SILVER-trained spaCy NER (503 when untrained; baseline stays default)."""
    try:
        result = trained_svc.predict_trained(payload.text)
    except trained_svc.ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return EntitiesResponse(
        entities=[
            EntityResult(
                text=m.text, label=m.label, start=m.start, end=m.end,
                method=m.method,
            )
            for m in result.entities
        ],
        count=len(result.entities),
        model=NerModelInfo(name="marathi_silver_spacy_ner_v1", type="spacy_silver_ner"),
    )


@trained_router.get("/ner-status", response_model=dict)
def ner_status() -> dict:
    return trained_svc.trained_status()


@trained_router.get("/ner-metrics", response_model=dict)
def ner_metrics() -> dict:
    import json
    import os

    from app.core.config import settings

    path = os.path.join(settings.model_dir, "ner", "ner_metrics.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "No trained NER metrics. Train with training/train_ner.py."}
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    return {"available": True, **stored}


@trained_router.get("/ner-indicbert-status", response_model=dict)
def ner_indicbert_status() -> dict:
    return ib_svc.indicbert_ner_status()


@trained_router.get("/ner-indicbert-metrics", response_model=dict)
def ner_indicbert_metrics() -> dict:
    import json
    import os

    from app.core.config import settings

    path = os.path.join(settings.model_dir, "ner_indicbert", "ner_metrics.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "No IndicBERT NER metrics. Train with training/train_ner_indicbert.py."}
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    return {"available": True, **stored}
