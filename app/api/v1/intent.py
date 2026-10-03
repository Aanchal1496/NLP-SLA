from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator

from app.services import intent_classifier as svc

router = APIRouter()
flat_router = APIRouter()


class IntentRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value


class IntentModelInfo(BaseModel):
    name: str
    labels: list[str]


class IntentResponse(BaseModel):
    intent: str
    confidence_type: str
    confidence: float | None
    low_confidence: bool
    model: IntentModelInfo


@router.post("/intent", response_model=IntentResponse)
def predict_intent_endpoint(payload: IntentRequest) -> IntentResponse:
    return _respond(payload)


@flat_router.post("/predict-intent", response_model=IntentResponse)
def predict_intent_flat(payload: IntentRequest) -> IntentResponse:
    """Required integration path (alias of ``/predict/intent``)."""
    return _respond(payload)


def _respond(payload: IntentRequest) -> IntentResponse:
    try:
        result = svc.predict_intent(payload.text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except svc.ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    return IntentResponse(
        intent=result.intent,
        confidence_type=result.confidence_type,
        confidence=result.confidence,
        low_confidence=result.low_confidence,
        model=IntentModelInfo(name=result.model_name, labels=result.labels),
    )
