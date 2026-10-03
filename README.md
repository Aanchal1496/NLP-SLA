# Marathi Legal & Financial NLP — FastAPI Backend

**Purpose.** A self-contained backend for Marathi legal & financial
document analysis: Devanagari-preserving `.txt`/`.pdf` intake, configurable
Marathi preprocessing, TF-IDF intent classification (once annotated data is
supplied), a transparent hybrid NER baseline, dataset analytics, held-out
evaluation reporting, combined analysis, and JSON/CSV export — served over a
documented REST API with an interactive `/docs` UI.

**Current data status (verified, not assumed).** No annotated project
dataset, annotation guidelines, or trained model exist in this workspace
yet. Endpoints that need them return explicit unavailable statuses; nothing
is fabricated. See `data/README.md` for the expected schema.

## Backend architecture

```text
app/
├── main.py               # app factory, CORS (explicit origins), /health, /docs
├── core/config.py        # NLP_*-prefixed settings (paths, limits, origins)
├── api/v1/
│   ├── documents.py      # pasted-text + file extraction endpoints
│   ├── preprocess.py     # /preprocess
│   ├── intent.py         # /predict/intent + /predict-intent alias
│   ├── entities.py       # /predict/entities + /extract-entities alias
│   ├── analytics.py      # dataset stats/records, metrics, NER eval (+ aliases)
│   └── analyze.py        # /analyze, /analyze-file, /export
├── schemas/              # Pydantic request/response contracts
└── services/             # route-independent logic (extraction, preprocessing,
                          # intent + NER, analytics, combined analysis)
training/                 # dataset validation, train_intent CLI, analytics CLI
tests/                    # unit + API + integration tests (111)
data/raw/                 # project CSV goes here (empty; see data/README.md)
models/                   # trained artifacts go here (empty until training)
```

No database or external service is used — the API is stateless.

## Prerequisites

- Python 3.12 (verified on 3.12.0) with `pip`.
- Test fixture PDFs need a Devanagari font: Windows
  `C:\Windows\Fonts\Nirmala.ttc` or Linux Noto Sans Devanagari.

## Installation

```powershell
cd D:\HARSH\projects\Timepass\NLP
pip install -r requirements.txt
Copy-Item .env.example .env   # optional; defaults work out of the box
```

## Dataset setup

```powershell
# Place the annotated UTF-8 CSV here (never commit private data):
#   columns: text , intent  (+ optional entity-span/BIO column)
Copy-Item <your-annotated-file>.csv data\raw\intents.csv
python -m training.analytics --data data/raw/intents.csv --text-col text --label-col intent
```

Column names are configurable (`--text-col`, `--label-col`); every distinct
label value becomes a trained intent. Validation refuses missing files,
wrong schemas, and unsuitable data (exit codes 2/3) instead of guessing.

## Model training

```powershell
python -m training.train_intent --data data/raw/intents.csv --text-col text --label-col intent --out-dir models
```

Compares TF-IDF + LogisticRegression vs TF-IDF + LinearSVC on a
reproducible held-out split (seed 42; stratification only if supported),
persists `intent_pipeline.joblib` + labels + full metrics to `models/`.

## Server startup

One command runs the whole website (API + dashboard — the backend serves
`frontend/` at `/app/`):

```powershell
powershell -ExecutionPolicy Bypass -File start-dev.ps1
```

- Website: `http://127.0.0.1:8000/app/`
- API docs: `http://127.0.0.1:8000/docs`

Manual equivalent: `uvicorn app.main:app --reload` (run from this folder),
then open `http://127.0.0.1:8000/app/`. Two-terminal alternative: backend as
above + `python -m http.server 3000 --directory frontend` (allowed CORS
origin), opening `http://localhost:3000`.

Health: `GET /health` → `{"status": "ok"}`. Interactive docs: `/docs`;
machine schema: `/openapi.json`. No database or external service required.

## Frontend (static dashboard)

No installation step and no framework: `frontend/index.html` +
`frontend/nlp-integration.js` (vanilla JS, Tailwind via CDN, syntax-checked
with `node --check`). The dashboard is a multi-view app: the sidebar
switches focused pages (Dashboard, Analyzer, Results, Dataset) via hash
routes (`#/home`, `#/analyzer`, `#/results`, `#/dataset`) with working
browser-back support; analyzing a document jumps straight to Results.
calls the real API (`/health`, `/analyze`, `/analyze-file`, `/export`,
`/dataset/stats`, `/dataset/records`, `/model/metrics`); API base defaults
to the serving origin on port 8000, else `http://127.0.0.1:8000`
(override: `localStorage.nlp_api_base`). All predictions, entities,
statistics, and downloads are live backend responses; unavailable data
shows explicit empty states, never fabricated values. File uploads use
`FormData` (browser-set multipart boundary); the Analyze button disables
while a request is in flight.

## Endpoint map

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness |
| `POST /api/v1/analyze` | combined analysis (stats, text, tokens, intent, entities, counts, warnings) |
| `POST /api/v1/analyze-file` | combined analysis for `.txt`/`.pdf` uploads |
| `POST /api/v1/preprocess` | Marathi preprocessing + stats |
| `POST /api/v1/predict-intent` | intent prediction (503 while untrained) |
| `POST /api/v1/extract-entities` | baseline NER spans |
| `GET /api/v1/dataset/stats` | chart-ready dataset stats (honest unavailable status) |
| `GET /api/v1/dataset/records?page=&page_size=` | paginated raw records (1–100 per page) |
| `GET /api/v1/model/metrics` | held-out intent metrics or unavailable status |
| `POST /api/v1/export` | analysis as downloadable JSON/CSV (`{format, filename, mime_type, content}`) |

Legacy aliases kept for backward compatibility: `/api/v1/predict/intent`,
`/api/v1/predict/entities`, `/api/v1/analytics/dataset`,
`/api/v1/analytics/intent-metrics`, `/api/v1/analytics/ner`.

## Frontend integration

Base URL `http://127.0.0.1:8000`. CORS allows `http://localhost:3000`,
`http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`
(override via `NLP_FRONTEND_ORIGINS`) with credentials — no wildcard.
`fetch` with `credentials: "include"` works from those origins.

```javascript
// Combined analysis
const res = await fetch("http://127.0.0.1:8000/api/v1/analyze", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "कलम ३०२ अंतर्गत सुनावणी होईल।"}),
});
const analysis = await res.json(); // tokens, intent (may be null + warnings), entities

// File upload
const form = new FormData();
form.append("file", fileInput.files[0]); // .txt or text-based .pdf
const up = await fetch("http://127.0.0.1:8000/api/v1/analyze-file", {method: "POST", body: form});

// Export + browser download
const exp = await fetch("http://127.0.0.1:8000/api/v1/export", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({format: "csv", text: analysis.original_text}),
});
const {filename, mime_type, content} = (await exp.json());
const blob = new Blob([content], {type: mime_type});
const a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = filename;
a.click();

// Paginated records + honest statuses
const page = await (await fetch("http://127.0.0.1:8000/api/v1/dataset/records?page=1&page_size=20")).json();
if (page.available === false) { /* show page.reason: no dataset yet */ }
```

## Endpoints

### Pasted text — `POST /api/v1/documents/text`

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/documents/text `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "मुंबई उच्च न्यायालयाने मालमत्ता वादात निर्णय दिला."}'
```

### File upload — `POST /api/v1/documents/upload`

Only `.txt` and text-based `.pdf` (max 10 MiB, configured via
`NLP_MAX_FILE_SIZE_BYTES`). Scanned/image-only PDFs are rejected with a
clear error since they contain no extractable text.

```powershell
# TXT
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/documents/upload `
  -Method Post -Form @{ file = Get-Item .\karar.txt }

# PDF
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/documents/upload `
  -Method Post -Form @{ file = Get-Item .\nirnay.pdf }
```

Both endpoints return `{ text, char_count, source_type, filename }`.

## Text preprocessing — `POST /api/v1/preprocess`

Reusable service (`app/services/marathi_preprocess.py`, no FastAPI imports)
for legal/financial Marathi text. Returns `{ original_text,
processed_text, sentences, tokens, stats }` where `stats = { char_count,
word_count, sentence_count, vocab_size }` over the processed output.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/preprocess `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "कलम ३०२ अंतर्गत दिनांक १२/०५/२०२४ रोजी ₹५,००,००० देय आहेत।",
           "options": {"remove_stopwords": true}}'
```

Options (all optional): `unicode_form` (`NFC` default; `NFKC`/`NFD`/`NONE`),
`remove_urls`, `remove_html` (both default `false` to preserve information),
`remove_stopwords` (default `false`), `custom_stopwords` (extends the
minimal built-in list). Empty string input returns a safe empty result
(200); invalid options return 422.

Preservation guarantees (tested): Devanagari intact; section numbers
(`३०२`), dates (`१२/०५/२०२४`), amounts (`₹५,००,०००`), case references
(`१२३/२०२४`), mixed-script tokens (`FIR`, `No.`, `123/2024`), and
negations (`नाही` is never a stopword). Sentence splitting is naive and
rule-based (danda/`॥`/`?`/`!`, guarded `.`); abbreviations and decimals
(`बी.सी.`, `१२.५`) do not split.

Honest limitations: **no stemming, lemmatization, or morphological
analysis** is implemented (inflections stay distinct); the stopword list is
minimal; no OCR (scanned PDFs still rejected); no lowercasing applied.

## Intent classification — `POST /api/v1/predict/intent`

Service (`app/services/intent_classifier.py`) + training CLI
(`training/train_intent.py`): TF-IDF pipelines comparing LogisticRegression
(probabilities) vs LinearSVC (raw decision scores, labelled as such — never
presented as probabilities), reproducible split (`--seed`, default 42;
stratification only when every class has ≥ 2 samples), vectorizer fitted on
train data only, artifacts (`intent_pipeline.joblib`, `intent_labels.json`,
`intent_metrics.json`) under `models/` (`NLP_MODEL_DIR` overrides).

```powershell
# Train on the real annotated CSV once supplied (see data/README.md):
python -m training.train_intent --data data/raw/intents.csv --text-col text --label-col intent --out-dir models

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/predict/intent `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "कलम ३०२ अंतर्गत गुन्हा दाखल झाला।"}'
```

Response: `{ intent, confidence_type, confidence, low_confidence, model }`.
Empty text → 422; no trained artifacts → **503 model-unavailable**.

**Status: NO trained model exists.** Verified 2026-10-02: the workspace
contains no annotated dataset and no annotation guidelines (recursive
listing; sibling folders belong to other projects and were not used), so
training was not run and no metrics are reported. The CLI validates schema,
drops/counts missing text and labels, reports class frequencies and
insufficient (< 2-sample) classes, and refuses unsuitable data (exit 2/3)
instead of fabricating a model.

## NER — `POST /api/v1/predict/entities`

Service (`app/services/ner.py`): **hybrid baseline, NOT a trained NER
model.** Regex for DATE, MONEY (`₹`/रुपये), CASE_NUMBER (`नं. १२३/२०२४`,
bare `१२३/२०२४`), SECTION (`कलम ३०२`) plus a configurable gazetteer of
legal/financial proper names (`custom_gazetteer` supported; groups
toggleable via options). Each entity returns `{ text, label, start, end,
method }` with offsets into the original text. Overlaps resolve
deterministically (earliest start, longest span, DATE > MONEY >
CASE_NUMBER > SECTION > ORGANIZATION > PERSON); identical spans deduped;
gazetteer holds multi-word names only so generic words are not labelled.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/predict/entities `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "कलम ३०२ अंतर्गत दिनांक १२/०५/२०२४ रोजी सुनावणी होईल।"}'
```

**Schema status:** no CCE-2 guidelines, entity annotations (spans/BIO/
columns/plain), or trained NER model exist in the workspace, so the labels
(`DATE`, `MONEY`, `CASE_NUMBER`, `SECTION`, `ORGANIZATION`, `PERSON`) are
baseline-defined and must be reconciled with CCE-2 when supplied. No
dataset entity vocabulary is used. `entity_prf` (exact-span P/R/F1) ships
as a utility but no project-level scores are reported — no gold spans
exist.

## Analytics & evaluation — `/api/v1/analytics/*`

- `GET /api/v1/analytics/dataset` — chart-ready stats for the configured
  dataset (`NLP_DATASET_PATH`, default `data/raw/intents.csv`): record
  counts, missing/duplicates, intent distribution, entity distribution
  (only with a parseable JSON-spans/BIO annotation column), text-length
  avg + histograms, sentence lengths, top words/bigrams, vocabulary size.
  Missing dataset → `{available: false, reason}` (200, no invented values).
- `GET /api/v1/analytics/intent-metrics` — persisted held-out metrics
  (accuracy, macro/weighted precision/recall/F1, classification report,
  confusion matrix) or `{available: false, reason}` when unevaluated.
- `POST /api/v1/analytics/ner` — entity-level P/R/F1 for caller-supplied
  gold spans (exact span+label match); empty gold → `{evaluated: false,
  reason}`.
- CLI: `python -m training.analytics --data <csv> [--entity-col E] [--out FILE]`.

**Status: dataset and intent evaluation unavailable** (no project CSV
supplied, no model trained); NER project-level evaluation infeasible (no
gold spans). All three endpoints return explicit statuses instead.

## Error behaviour

| Input | Status | Note |
|---|---|---|
| Unsupported extension (`.exe`, `.docx`, none) | 400 | Only `.txt`/`.pdf` accepted |
| Empty (0-byte) file | 400 | — |
| Over 10 MiB | 413 | — |
| Blank text / no extractable text | 422 | Includes scanned PDFs |
| Password-protected PDF | 422 | Upload an unprotected copy |
| Malformed / non-PDF bytes as `.pdf` | 422 | `%PDF-` header checked |

## Tests

```powershell
python -m pytest tests/ -v
```

47 tests: Marathi preservation (TXT, PDF, pasted text), paragraph
boundaries, invalid/empty/oversize inputs, encrypted/malformed/blank
PDFs, temp-file cleanup, API route checks, plus preprocessing
(normalization, segmentation, legal-identifier preservation, stopwords,
stats, empty input, `/preprocess` route), plus intent classification
(dataset validation, pipeline save/load, honest metrics, unavailable and
empty-input handling, `/predict/intent` route) — 64 total, plus NER
(regex/gazetteer entities, overlap resolution, offset invariant,
`/predict/entities` route) — 83 total, plus analytics (dataset stats,
extended intent metrics, NER-eval and unavailable-status endpoints) —
99 total, plus integration (all endpoints, uploads, flat aliases,
exports, CORS, live boot, no-fabrication checks) — 111 total, plus
frontend contracts, page routing, and JS execution checks — 123 total.

## Known limitations

- **No annotated dataset supplied** → dataset stats/records, intent
  training/prediction, and intent metrics are unavailable (explicit
  statuses, exit codes 2/3); `models/` and `data/raw/` stay empty.
- **NER is a hybrid baseline** (regex + gazetteer), not a trained model;
  labels are baseline-defined pending CCE-2 guidelines; no dataset entity
  vocabulary; person/organization coverage is limited to curated terms.
- **No stemming, lemmatization, or morphological analysis**; minimal
  stopword list; naive rule-based sentence splitting; no OCR (scanned PDFs
  rejected); test fixture PDFs need a system Devanagari font.
- Probabilities come from LogisticRegression only when trained, with no
  post-hoc calibration; LinearSVC yields raw decision scores.
- `pip check` reports pre-existing conflicts in unrelated global packages
  (google-*, datasets, utim-cli) — none involve this project's pinned
  dependencies, and all 13 pins verify exactly.

## Legal disclaimer

**This tool does not provide legal advice.** It performs text processing
(extraction, cleaning, classification baselines, pattern-based entity
spotting) for study and triage. Outputs may be incomplete or wrong; always
consult a qualified legal professional for any legal or financial matter.

## Professor demo checklist (5 minutes)

1. `powershell -ExecutionPolicy Bypass -File start-dev.ps1` → open
   `http://127.0.0.1:8000/app/` (header shows Backend Connected).
2. Click a preset sample (court order / bank notice / FIR) → Analyze →
   show live tokens, entities with offsets, stats, and warnings.
3. Upload a `.txt`/`.pdf` → same results via file path; try an `.exe` to
   show graceful rejection.
4. Export JSON + CSV → open the downloads, point at real response data.
5. Scroll to Dataset/Metrics → honest "not connected" states (no dataset or
   trained model yet); open `/docs` for the API contract.

## Common errors

| Symptom | Cause / fix |
|---|---|
| Dashboard shows "Backend offline" | Start the backend (`start-dev.ps1`); default API is `http://127.0.0.1:8000`. Override via `localStorage.nlp_api_base`. |
| `503` on intent / `intent-unavailable` warning | No `models/` artifacts: train first (`training/train_intent.py` needs `data/raw/intents.csv`). |
| Dataset panels say "not connected" | No annotated CSV yet — place it at `data/raw/intents.csv`. |
| `422` on uploads | Only `.txt`/text-based `.pdf` ≤ 10 MiB; scanned PDFs unsupported (no OCR). |
| `pip install` fails | Use Python 3.12; rerun from the project root. |
