from pydantic import BaseModel, Field


class PreprocessOptionsSchema(BaseModel):
    unicode_form: str = Field(default="NFC", pattern="^(NFC|NFKC|NFD|NONE)$")
    remove_urls: bool = False
    remove_html: bool = False
    remove_stopwords: bool = False
    custom_stopwords: list[str] | None = None


class PreprocessRequest(BaseModel):
    # Empty string is allowed -> returns a safe empty result (200).
    text: str
    options: PreprocessOptionsSchema = PreprocessOptionsSchema()


class TextStatisticsSchema(BaseModel):
    char_count: int
    word_count: int
    sentence_count: int
    vocab_size: int


class PreprocessResponse(BaseModel):
    original_text: str
    processed_text: str
    sentences: list[str]
    tokens: list[str]
    stats: TextStatisticsSchema
