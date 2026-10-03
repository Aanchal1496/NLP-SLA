from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.preprocess import (
    PreprocessRequest,
    PreprocessResponse,
    TextStatisticsSchema,
)
from app.services import marathi_preprocess as svc

router = APIRouter()


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_document(payload: PreprocessRequest) -> PreprocessResponse:
    if not isinstance(payload.text, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text must be a string.",
        )
    try:
        options = svc.PreprocessOptions(
            unicode_form=payload.options.unicode_form,
            remove_urls=payload.options.remove_urls,
            remove_html=payload.options.remove_html,
            remove_stopwords=payload.options.remove_stopwords,
            custom_stopwords=payload.options.custom_stopwords,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    try:
        result = svc.preprocess_marathi_text(payload.text, options)
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    return PreprocessResponse(
        original_text=result.original_text,
        processed_text=result.processed_text,
        sentences=result.sentences,
        tokens=result.tokens,
        stats=TextStatisticsSchema(
            char_count=result.stats.char_count,
            word_count=result.stats.word_count,
            sentence_count=result.stats.sentence_count,
            vocab_size=result.stats.vocab_size,
        ),
    )
