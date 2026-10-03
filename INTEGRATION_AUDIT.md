# INTEGRATION_AUDIT — Marathi Legal & Financial NLP (frontend + backend)

Date (UTC): 2026-10-02. Every finding verified against actual files listed below.

## 1. Detected technologies

| Side | Finding (evidence) |
|---|---|
| Frontend | Single static file `frontend/index.html` (~185 KB, 2424 lines). NO framework (no React/Vue/Angular), NO `package.json`, NO lockfile, NO bundler config, NO `.env*`. Styling: Tailwind via CDN + Google Fonts. Logic: 4 unique inline vanilla-JS blocks (~13 KB). Backend entry: `app/main.py` (`create_app()`), FastAPI 0.115.12 + uvicorn 0.34.3 (pinned in `requirements.txt`). |
| NLP services | `app/services/`: `text_extraction.py` (PyMuPDF), `marathi_preprocess.py`, `intent_classifier.py` (TF-IDF sklearn Pipeline, NO trained artifact), `ner.py` (hybrid baseline), `analytics.py`, `analysis.py`. |
| Training | `training/train_intent.py`, `training/dataset.py`, `training/analytics.py`. |
| Dataset / models | `data/raw/` EMPTY (only `.gitkeep`); `models/` EMPTY (only `.gitkeep`). No annotations, no CCE-2 guidelines in workspace. |
| Tests | Backend pytest suite `tests/` (111 tests, all passing). No frontend tests (nothing to run: no package manager). |

## 2. Existing API endpoints (from `app/api/v1/router.py` + live `/openapi.json`)

`GET /health`, `GET /`; `POST /api/v1/analyze`, `/analyze-file`, `/export`;
`POST /api/v1/preprocess`; `POST /api/v1/predict-intent` (+ alias `/predict/intent`);
`POST /api/v1/extract-entities` (+ alias `/predict/entities`);
`GET /api/v1/dataset/stats` (+ `/analytics/dataset`), `/dataset/records`,
`/model/metrics` (+ `/analytics/intent-metrics`); `POST /api/v1/analytics/ner`;
legacy `/api/v1/documents/*`. Plus `/docs`, `/openapi.json`.

## 3. Current frontend→backend connections

**NONE.** `frontend/index.html` contains zero `fetch`/`axios`/`XMLHttpRequest`
calls and zero localhost references (verified by scan). All UI is static:

- `analyze-action-btn` is an `<a href="#results">` anchor — scrolls only.
- Results panes/tokens/statistics/metrics/dataset table are hardcoded sample text.
- Export buttons run a mock animation (`"Export Buttons Mock Action Feedback"` in source) and download nothing.
- Payload preview builds a fictional schema (`document_type`, `model_architecture`, `devanagari_stemming`) matching no backend contract.
- `model-selector` offers `naive_bayes` and `logistic_regression` — backend has no model choice (auto-selects; currently untrained).
- Fabricated claims: "F1 91.4%", "Devanagari OCR 98.4%", "Corpus 1,420 docs", "FastAPI: Connected (v1.2)" (always-on pill), "Lemmatized Tokens", "Tokenizer: IndicNLP + Spacy-Devanagari", fake endpoint `https://api.marathinlp.ai/v1/dataset-statistics` with `Bearer dev_token`.

## 4. Missing / broken connections (to fix)

1. Pasted-text analysis → `POST /api/v1/analyze`.
2. PDF/TXT upload → `POST /api/v1/analyze-file` (currently staged client-side only).
3. Preprocessing view → `processed_text/tokens/stats` from `/analyze`.
4. Intent display → `intent` object (null + warning while untrained).
5. NER display → `entities` with offsets.
6. Dataset stats/records → `GET /dataset/stats`, `/dataset/records` (unavailable states).
7. Metrics → `GET /model/metrics` (unavailable state).
8. Export → `POST /export` real downloads (JSON + CSV); PDF-report button has no backend support → rewire to CSV.
9. API base URL config, connection status, loading/error/empty states.
10. Single-command serving: mount `frontend/` at `/app` in FastAPI.

## 5. Dependency / configuration issues

- Frontend needs no install (static file). Backend pins all verified (13/13 exact).
- CORS already allows `localhost:3000/5173` (+127.0.0.1) with credentials, no wildcard — correct for separate serving; same-origin `/app` mount needs no CORS at all.
- `model-selector` values do not exist server-side (fix in UI).
- `dev_token` bearer in UI implies auth that does not exist (remove).

## 6. Planned integration fixes

1. Backend: `StaticFiles` mount of `frontend/` at `/app` (skip gracefully if missing); test asserting 200 + HTML.
2. Frontend: one `NLP_APP` integration script — `API_BASE` (same-origin-aware), live `/health` pill, analyze wiring with loading/error/duplicate-submit guard, live render into the four tab panes + intent/entities regions, real export downloads, live dataset/metrics with unavailable states, honest payload preview, corrected claims/labels/selector/endpoint card, legal disclaimer.
3. Tests: `tests/test_frontend_integration.py` (static HTML/JS contract checks: no fake URLs, IDs referenced exist, API paths match backend routes); keep 111 green.
4. Docs: README startup (one-command + two-terminal), `start-dev.ps1`.

## 7. Current blockers

- No annotated dataset → intent stays 503/unavailable; dataset/metrics sections show honest empty states (NOT blocked for integration: states are testable).
- No CCE-2 guidelines → NER labels remain baseline-defined (documented).
- No browser automation in this environment → real-browser console/responsive checks cannot run; verify via served-HTML assertions + backend contract tests instead (marked below).

## 8. Fixes applied (verified)

- Removed mock export handlers; exports now download real `/export` JSON/CSV.
- `updatePayloadPreview` shows the real request schema; static sample payload replaced.
- `model-selector` reduced to honest single "Auto" option; `resetForm()` implemented (was undefined).
- Dead nav links (`#models`, `#ask`, `#about`) removed; working anchors added (`#analyzer`, `#results`, `#dataset`, `#pipeline`, `#search`, `#top`).
- Fabricated numbers neutralized (`91.4%`, `98.4%`, `1,420`, `850/570`, `320ms`, `0.942 F1`, fake `api.marathinlp.ai` + bearer token); connection pill is live via `/health`.
- Results panes, intent card, dataset widgets/table, and stat cards render live backend data or explicit unavailable states; session analyses prepend to Recent Analyses.
- Backend serves the dashboard at `/app` (single-command demo); `start-dev.ps1` verified booting (health + `/app` over HTTP).
- Nav rebuilt as hash-routed pages (`#/home`, `#/analyzer`, `#/results`, `#/dataset`): the 4 concatenated document copies collapsed to one visible sidebar/header, one focused view at a time, back-button support, auto-jump to Results after analysis. Router boot verified by executing it against a stub DOM in Node (`tests/router_harness.js`).
- Disclaimer added (UI + README): experimental tool, no legal advice.
- Tests: `tests/test_frontend_integration.py` (10 contract tests) + `tests/test_frontend_js.py` (syntax + harness). Full suite: 123 passed.
