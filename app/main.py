from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.core.config import settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Bump when shipping user-visible backend features. The dashboard compares
# this against its own EXPECTED_BUILD and tells the user to restart/refresh
# on mismatch instead of failing silently with stale code.
BUILD = "1.4"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Marathi Legal & Financial NLP",
        version="1.0.0",
        description=(
            "Document extraction, Marathi preprocessing, intent "
            "classification, NER, analytics, and export for Marathi legal "
            "and financial documents."
        ),
    )
    # Explicit frontend origins only: never "*" together with credentials.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(v1_router)

    # Serve the bundled static dashboard (if present) at /app. API routes
    # registered above take precedence; missing dir never breaks the API.
    if FRONTEND_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", tags=["health"])
    def root() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version", tags=["health"])
    def version() -> dict[str, object]:
        return {"build": BUILD,
                "features": ["clauses", "clean-text", "groq-translate",
                             "spacy-silver-ner", "light-theme"]}

    return app


app = create_app()
