from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.document import PastedTextRequest, TextExtractionResponse
from app.services import text_extraction as svc

router = APIRouter()


def _http_error(exc: svc.DocumentExtractionError) -> HTTPException:
    if isinstance(exc, svc.FileTooLargeError):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )
    if isinstance(exc, svc.EmptyExtractedTextError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if isinstance(exc, (svc.CorruptDocumentError, svc.EncryptedPDFError)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/text", response_model=TextExtractionResponse)
def extract_pasted_text(payload: PastedTextRequest) -> TextExtractionResponse:
    try:
        text = svc.normalize_pasted_text(payload.text)
    except svc.DocumentExtractionError as exc:
        raise _http_error(exc)
    return TextExtractionResponse(
        text=text,
        char_count=len(text),
        source_type="pasted_text",
        filename=None,
    )


@router.post("/upload", response_model=TextExtractionResponse)
async def extract_uploaded_file(file: UploadFile = File(...)) -> TextExtractionResponse:
    # Never trust the uploaded filename as a path; keep only the base name
    # for the response echo. Type detection uses the suffix only.
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
        text = svc.extract_from_upload(
            safe_name,
            data,
            max_size_bytes=settings.max_file_size_bytes,
        )
    except svc.DocumentExtractionError as exc:
        raise _http_error(exc)
    source = svc.detect_source_type(safe_name)
    return TextExtractionResponse(
        text=text,
        char_count=len(text),
        source_type=source,  # type: ignore[arg-type]
        filename=safe_name or None,
    )
