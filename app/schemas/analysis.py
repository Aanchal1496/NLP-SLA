from typing import Literal

from pydantic import BaseModel, field_validator

from app.schemas.preprocess import PreprocessOptionsSchema


class AnalyzeRequest(BaseModel):
    text: str
    options: PreprocessOptionsSchema = PreprocessOptionsSchema()

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value


class AnalysisStatistics(BaseModel):
    char_count: int
    word_count: int
    sentence_count: int
    vocab_size: int


class IntentPart(BaseModel):
    intent: str
    confidence_type: str
    confidence: float | None
    low_confidence: bool
    model_name: str
    labels: list[str]


class EntityPart(BaseModel):
    text: str
    label: str
    start: int
    end: int
    method: str


class ClausePart(BaseModel):
    index: int
    original: str
    clean: str
    intent: IntentPart | None = None
    entities: list[EntityPart] = []
    warnings: list[str] = []


class TranslationPart(BaseModel):
    available: bool
    text: str = ""
    provider: str = "groq"
    model: str = ""
    reason: str = ""


class AnalyzeResponse(BaseModel):
    original_text: str
    processed_text: str
    tokens: list[str]
    sentences: list[str]
    statistics: AnalysisStatistics
    intent: IntentPart | None
    entities: list[EntityPart]
    entity_counts: dict[str, int]
    warnings: list[str]
    clauses: list[ClausePart] = []
    clean_text: str = ""
    translation: TranslationPart | None = None


class TranslateRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value


class TranslateResponse(BaseModel):
    available: bool
    text: str = ""
    provider: str = "groq"
    model: str = ""
    reason: str = ""


class ExportRequest(BaseModel):
    format: Literal["json", "csv"]
    text: str
    options: PreprocessOptionsSchema = PreprocessOptionsSchema()
    include_translation: bool = False

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value


class ExportResponse(BaseModel):
    format: str
    filename: str
    mime_type: str
    content: str
