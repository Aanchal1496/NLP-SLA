from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.analysis import TranslateRequest, TranslateResponse
from app.services import translate as svc

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse)
def translate_text(payload: TranslateRequest) -> TranslateResponse:
    try:
        result = svc.translate_to_english(payload.text)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return TranslateResponse(
        available=result.available,
        text=result.text,
        provider=result.provider,
        model=result.model,
        reason=result.reason,
    )
