# IMPLEMENTATION_PLAN — Marathi Legal & Financial Document Analysis (Intent + NER)

> Phase 0 — Inspection only. No backend features built in this phase.
> Workspace: `D:\HARSH\projects\Timepass\NLP`
> Date (UTC): 2026-10-02

## 1. Workspace inspection (verified)

Three independent readers agree the workspace is empty:

1. `read` on `D:\HARSH\projects\Timepass\NLP` → `(0 entries)`.
2. `glob` with pattern `**/*` in same path → `No files found`.
3. PowerShell `Get-ChildItem -Force -Recurse` → no output, plus `cmd /c dir /a` → `0 File(s), 2 Dir(s)` (`.` and `..` only).

Parent check:

- `D:\HARSH\projects\Timepass\` contains: `calculator, College election, NLP, Portfolio, resume, rock paper sicssors, tic tac toe, VoteSphere`.
- `NLP/` itself contains no hidden files, no subdirectories.

### 1.1 Searched categories — all ABSENT

| Category | Expected patterns searched | Result |
|---|---|---|
| CCE-1 / CCE-2 reports | `*CCE*`, `*cce*`, `*.pdf`, `*.docx` | Not found |
| SLA / synopsis | `*SLA*`, `*sla*`, `*synopsis*`, `*proposal*` | Not found |
| Dataset / annotation | `*.csv`, `*.json`, `*.jsonl`, `*.xlsx`, `*.conll`, `*.bio`, `data/**` | Not found |
| Notebooks | `*.ipynb` | Not found |
| Models / checkpoints | `*.pt`, `*.bin`, `*.onnx`, `model*/**`, `*.pkl`, `*.joblib` | Not found |
| Source code | `*.py`, `app/**`, `src/**`, `requirements*.txt`, `pyproject.toml` | Not found |

No files were deleted, replaced, or reorganized. There was nothing to preserve.

## 2. Dataset findings — explicitly UNAVAILABLE

Because no dataset file exists in the workspace, the following are all marked **unavailable / unverifiable** (not invented):

- File format and location: **unavailable — no dataset file present.**
- Number of records: **unavailable.**
- Column names: **unavailable.**
- Missing values and duplicates: **unavailable — nothing to profile.**
- Intent labels and frequencies: **unavailable.**
- Entity labels and annotation format (BIO / BIOES / spaCy spans / Doccano / Label Studio): **unavailable — no annotation sample inspected.**
- Language and domain: **claimed by project title (Marathi, legal + financial) but no data sample verified.**

Verification loop: attempted `read`, `glob`, recursive `Get-ChildItem`, and `dir /a` (covers hidden/system files). No alternative reader (e.g., pandas/JSON loader) was applicable since there is no file path to open. Limitation reported as blocker, not as data.

## 3. Existing code / reusable components

- **None.** No `.py`, notebook, or model artifact exists to reuse.

## 4. Suitability for supervised intent classification + trained NER

- **Cannot be determined from workspace.** No labeled examples were found.
- Supervised intent classification requires `(text, intent_label)` pairs; trained NER requires token-level or span-level entity annotations. Neither is present.
- Blocker for next phase: user must supply (a) annotated dataset path + format, or (b) permission to create a starter/sample dataset and annotation schema.

## 5. Environment (verified via execution)

- `python --version` → `Python 3.12.0`
- `pip --version` → `pip 26.1.2` (`.../Python312/Lib/site-packages`, `python 3.12`)
- Key installed packages (via `pip list`, filtered):
  - `fastapi 0.115.12`, `uvicorn 0.34.3`, `pydantic 2.11.3`, `pydantic-settings 2.9.1`
  - `torch 2.14.0`, `torchvision 0.29.0`, `transformers 5.12.1`, `sentence-transformers 5.6.0`
  - `scikit-learn 1.9.0`, `spacy 3.8.14`, `pandas 3.0.3`, `numpy 2.4.6`
  - `SQLAlchemy 2.0.40`, `onnx 1.22.0`, `onnxruntime 1.29.0`
  - `pytest 8.3.5`, `pytest-asyncio 0.26.0`, `pytest-cov 7.1.0`
- Total installed packages: ~296 lines in `pip list`.
- Git repo: workspace is not a git repo (`Is directory a git repo: no` per env; no `.git` found).
- No `requirements.txt` / `pyproject.toml` exists yet — dependencies are currently global-site only and undocumented in-repo.

## 6. Proposed modular FastAPI backend (fits empty workspace)

Create only when Phase 1 is approved. Do not scaffold yet per this phase's scope:

```text
NLP/
├── IMPLEMENTATION_PLAN.md      # this file (Phase 0 deliverable)
├── requirements.txt            # pinned from §5 + indic/Marathi tokenizer needs
├── README.md                   # setup + run instructions (Phase 1)
├── app/
│   ├── main.py                 # FastAPI app factory, router wiring, health check
│   ├── core/
│   │   ├── config.py           # settings via pydantic-settings
│   │   └── logging.py
│   ├── api/v1/
│   │   ├── router.py
│   │   ├── intent.py           # POST /predict/intent
│   │   ├── entities.py         # POST /predict/entities
│   │   └── analyze.py          # POST /analyze (intent + NER combined)
│   ├── schemas/
│   │   ├── request.py          # AnalyzeRequest { text }
│   │   └── response.py         # IntentResult, EntityResult, AnalyzeResponse
│   ├── services/
│   │   ├── preprocessor.py     # Marathi normalization, cleanup
│   │   ├── intent_classifier.py# interface; rule-based stub → trained model
│   │   └── ner_extractor.py    # interface; regex/stub → trained model
│   └── models/
│       └── registry.py         # model loading, versioning, lazy load
├── data/
│   ├── raw/                    # user-supplied dataset goes here (gitignored if large)
│   ├── processed/
│   └── README.md               # dataset source, license, schema description
├── training/
│   ├── train_intent.py
│   ├── train_ner.py
│   └── eval.py
├── tests/
│   └── test_health.py
└── models/                     # trained artifacts (gitignored, documented)
```

Design notes:

- Marathi preprocessing isolated in `services/preprocessor.py` so tokenizer/model swaps don't touch API layer.
- `intent_classifier.py` / `ner_extractor.py` expose a stable interface; Phase 1 can ship deterministic stubs so API contracts are testable before training data exists.
- Training scripts kept outside `app/` to keep the serving image lean.
- Suitable starter models (to decide in Phase 1 once data is known): multilingual/Indic BERT variants (e.g., `ai4bharat/indic-bert`, `l3cube-pune/marathi-bert*`, `xlm-roberta-base`) fine-tuned for intent; token-classification head or spaCy pipeline for NER.

## 7. Intended file changes so far

- **Created in this phase:** `IMPLEMENTATION_PLAN.md` only.
- **No other files created, modified, or deleted.**

## 8. Remaining phases (proposed, awaiting prompt)

1. **Phase 1 — Contracts + stubs:** `requirements.txt`, `app/main.py` + health check, Pydantic schemas, stub `/analyze` endpoint, `tests/test_health.py`. Needs no dataset.
2. **Phase 2 — Data:** ingest user-supplied annotated dataset into `data/raw/`; profile it (records, columns, label distributions, annotation format); document schema in `data/README.md`.
3. **Phase 3 — Training:** `training/train_intent.py`, `training/train_ner.py`, `training/eval.py`; save artifacts to `models/`.
4. **Phase 4 — Inference wiring:** replace stubs with real model loading via `models/registry.py`; add Marathi preprocessing; error handling + logging.
5. **Phase 5 — Packaging:** README run instructions, pinned deps, sample curl/PowerShell examples, college-submission artifacts.

## 9. Blockers / what is needed next

1. **Dataset:** no annotated file in workspace. Please provide the dataset file (or its location/format) and the intent/entity label schema before Phase 3 can proceed.
2. **Scope confirmation:** confirm the `app/` layout above is acceptable, and whether persistence (DB) or auth is required for submission — default proposal is stateless API, no DB.
