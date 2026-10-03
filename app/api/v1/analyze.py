from __future__ import annotations

import csv
import io
import json
import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisStatistics,
    ClausePart,
    EntityPart,
    ExportRequest,
    ExportResponse,
    IntentPart,
)
from app.services import analysis as analysis_svc
from app.services import text_extraction as extraction_svc
from app.services.analysis import CombinedAnalysis
from app.services.marathi_preprocess import PreprocessOptions

router = APIRouter()


def _intent_part(intent) -> IntentPart | None:
    if intent is None:
        return None
    return IntentPart(
        intent=intent.intent,
        confidence_type=intent.confidence_type,
        confidence=intent.confidence,
        low_confidence=intent.low_confidence,
        model_name=intent.model_name,
        labels=intent.labels,
    )


def _to_response(result: CombinedAnalysis) -> AnalyzeResponse:
    stats = result.preprocessed.stats
    return AnalyzeResponse(
        original_text=result.original_text,
        processed_text=result.preprocessed.processed_text,
        tokens=result.preprocessed.tokens,
        sentences=result.preprocessed.sentences,
        statistics=AnalysisStatistics(
            char_count=stats.char_count,
            word_count=stats.word_count,
            sentence_count=stats.sentence_count,
            vocab_size=stats.vocab_size,
        ),
        intent=_intent_part(result.intent),
        entities=[
            EntityPart(
                text=m.text, label=m.label, start=m.start, end=m.end,
                method=m.method,
            )
            for m in result.entities
        ],
        entity_counts=result.entity_counts,
        warnings=result.warnings,
        clauses=[
            ClausePart(
                index=c.index,
                original=c.original,
                clean=c.clean,
                intent=_intent_part(c.intent),
                entities=[
                    EntityPart(
                        text=m.text, label=m.label, start=m.start,
                        end=m.end, method=m.method,
                    )
                    for m in c.entities
                ],
                warnings=c.warnings,
            )
            for c in result.clauses
        ],
        clean_text=result.clean_text,
        translation=None,
    )


def _service_options(schema_options) -> PreprocessOptions:
    return PreprocessOptions(
        unicode_form=schema_options.unicode_form,
        remove_urls=schema_options.remove_urls,
        remove_html=schema_options.remove_html,
        remove_stopwords=schema_options.remove_stopwords,
        custom_stopwords=schema_options.custom_stopwords,
    )


def _extraction_http_error(exc: extraction_svc.DocumentExtractionError) -> HTTPException:
    if isinstance(exc, extraction_svc.FileTooLargeError):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )
    if isinstance(exc, extraction_svc.EmptyExtractedTextError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if isinstance(exc, (extraction_svc.CorruptDocumentError, extraction_svc.EncryptedPDFError)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_document(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analysis_svc.analyze_text(payload.text, _service_options(payload.options))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return _to_response(result)


@router.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(file: UploadFile = File(...)) -> AnalyzeResponse:
    raw_name = file.filename or ""
    safe_name = os.path.basename(raw_name)
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )
    finally:
        try:
            await file.close()
        except Exception:
            pass
    try:
        text = extraction_svc.extract_from_upload(
            safe_name, data, max_size_bytes=settings.max_file_size_bytes
        )
    except extraction_svc.DocumentExtractionError as exc:
        raise _extraction_http_error(exc)
    try:
        result = analysis_svc.analyze_text(text)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return _to_response(result)


def _resolve_translation(text: str):
    """Translate for export; never raises (unavailable status instead)."""
    from app.schemas.analysis import TranslationPart
    from app.services import translate as translate_svc
    try:
        result = translate_svc.translate_to_english(text)
    except Exception as exc:
        return TranslationPart(available=False, reason=str(exc)[:300])
    return TranslationPart(
        available=result.available, text=result.text, provider=result.provider,
        model=result.model, reason=result.reason,
    )


def _analysis_to_csv(payload: AnalyzeResponse) -> str:
    """Clause-wise export: one row per clause + document + translation rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["section", "clause_index", "original", "clean", "intent",
         "intent_confidence", "entities", "english_translation",
         "translation_provider", "translation_model", "notes"]
    )
    for clause in payload.clauses:
        ents = " | ".join(f"{e.label}:{e.text}" for e in clause.entities)
        conf = ("" if clause.intent is None or clause.intent.confidence is None
                else clause.intent.confidence)
        writer.writerow(
            ["clause", clause.index, clause.original, clause.clean,
             clause.intent.intent if clause.intent else "", conf, ents,
             "", "", "", ";".join(clause.warnings)]
        )
    if not payload.clauses:
        intent_name = payload.intent.intent if payload.intent else ""
        conf = ("" if payload.intent is None or payload.intent.confidence is None
                else payload.intent.confidence)
        ents = " | ".join(f"{e.label}:{e.text}" for e in payload.entities)
        writer.writerow(
            ["document", "", payload.original_text, payload.processed_text,
             intent_name, conf, ents, "", "", "",
             ";".join(payload.warnings)]
        )
    if payload.translation is not None:
        tr = payload.translation
        writer.writerow(
            ["english_translation", "", "", "", "", "",
             "", tr.text if tr.available else "",
             tr.provider, tr.model,
             "" if tr.available else tr.reason]
        )
    return buf.getvalue()


@router.post("/export", response_model=ExportResponse)
def export_analysis(payload: ExportRequest) -> ExportResponse:
    try:
        result = analysis_svc.analyze_text(payload.text, _service_options(payload.options))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    response = _to_response(result)
    if payload.include_translation:
        response.translation = _resolve_translation(payload.text)
    if payload.format == "csv":
        return ExportResponse(
            format="csv",
            filename="analysis.csv",
            mime_type="text/csv",
            content=_analysis_to_csv(response),
        )
    return ExportResponse(
        format="json",
        filename="analysis.json",
        mime_type="application/json",
        content=json.dumps(response.model_dump(), ensure_ascii=False),
    )
