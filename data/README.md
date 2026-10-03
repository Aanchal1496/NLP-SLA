# Project dataset (annotated) — EXPECTED, NOT PRESENT

## Status (verified 2026-10-02)

**BLOCKER: no annotated project dataset has been supplied.** The workspace
contains no CSV/JSON/XLSX dataset and no annotation guidelines (checked via
recursive file listing of this project folder). Training refuses to run
until a real dataset is placed here. Nothing has been fabricated in its
place: `POST /api/v1/predict/intent` returns **503 model-unavailable** and
`python -m training.train_intent` exits with code 2/3 and a clear message.

Do NOT borrow data from sibling folders (e.g. other student projects) — the
classifier must train on this project's own Marathi legal/financial
annotations.

## Expected schema

Place the file at `data/raw/<name>.csv` (UTF-8, Devanagari text intact):

| column | meaning | example |
|---|---|---|
| `text` | document / sentence text (Marathi) | `कलम ३०२ अंतर्गत गुन्हा दाखल झाला।` |
| `intent` | gold intent label (exact string) | `fir_nond` |

Column names are configurable (`--text-col`, `--label-col`); the target
label column is whichever `--label-col` points at (default `intent`).
Every distinct value in that column becomes a trained intent — use the
existing project labels verbatim, do not rename them.

## Validation rules enforced by `training/dataset.py`

- Missing file / unreadable / empty file → refuse (exit 2).
- Missing text/label columns → refuse, listing actual columns (exit 2).
- Rows with missing or blank text/labels are dropped and counted.
- Zero usable rows, fewer than 2 distinct labels, or fewer than 6 usable
  rows → refuse (exit 3, unsuitable).
- Classes with fewer than 2 samples are reported as INSUFFICIENT;
  stratification is applied only when every class has ≥ 2 samples.
