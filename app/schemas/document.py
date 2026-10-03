from typing import Literal

from pydantic import BaseModel, field_validator


class PastedTextRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Pasted text must not be empty.")
        return value


class TextExtractionResponse(BaseModel):
    text: str
    char_count: int
    source_type: Literal["pasted_text", "txt", "pdf", "docx"]
    filename: str | None = None
