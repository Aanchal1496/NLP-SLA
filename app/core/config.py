from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NLP_", env_file=".env",
                                        env_file_encoding="utf-8")

    app_name: str = "Marathi Legal & Financial NLP"
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MiB
    allowed_extensions: frozenset[str] = frozenset({".txt", ".pdf", ".docx"})
    model_dir: str = "models"  # trained artifacts (intent_pipeline.joblib …)
    dataset_path: str = "data/raw/intents.csv"  # project dataset (when supplied)
    dataset_text_col: str = "text"
    dataset_label_col: str = "intent"
    dataset_entity_col: str | None = None
    frontend_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    # GROQ (OpenAI-compatible) translation: Marathi -> English.
    # Set NLP_GROQ_API_KEY in .env (never commit .env). Empty => /translate
    # returns {available: false} and analysis still works.
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    groq_timeout_s: float = 30.0


settings = Settings()
