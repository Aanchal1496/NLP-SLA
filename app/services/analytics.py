"""Dataset analytics + evaluation-status service (route-independent).

- Descriptive statistics are computed from ACTUAL data passed in (or a real
  file via :func:`analyze_dataset_file`). Nothing is hardcoded and synthetic
  fixtures never leave the test process.
- Analytics is descriptive, not gating: it reports on whatever usable rows
  exist (even a single label), unlike training validation which refuses.
- Entity distribution is included only when an annotation column is present
  and parseable (JSON span lists ``[{"label": ...}]`` or BIO tag strings);
  otherwise it is ``None`` with an explicit reason.
- NER project-level evaluation is impossible without gold spans (none exist
  in this workspace); :func:`ner_evaluation_status` says so explicitly.
  Callers WITH gold spans should POST them to
  ``/api/v1/analytics/ner`` (exact span+label matching) or use
  ``app.services.ner.entity_prf`` directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.services.marathi_preprocess import segment_sentences, tokenize_words
from training.dataset import DatasetError, load_dataset

TOP_N = 20
CHAR_BIN_EDGES = (50, 150, 300, 600)
WORD_BIN_EDGES = (5, 10, 20, 30, 50)


def histogram(values: list[int], edges: tuple[int, ...]) -> dict[str, Any]:
    """Chart-ready histogram: human-readable bin labels + counts."""
    bounds = [0, *list(edges), float("inf")]
    labels: list[str] = []
    counts = [0] * (len(bounds) - 1)
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        labels.append(f"{int(lo) + 1}+" if hi == float("inf") else f"{int(lo) + 1}-{int(hi)}")
        counts[i] = sum(1 for v in values if lo < v <= hi)
    return {"bins": labels, "counts": counts}


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def frequent_terms(tokens: list[str], top_n: int = TOP_N) -> list[dict[str, Any]]:
    ordered = sorted(Counter(tokens).items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"token": tok, "count": cnt} for tok, cnt in ordered[:top_n]]


def frequent_bigrams(tokens: list[str], top_n: int = TOP_N) -> list[dict[str, Any]]:
    pairs = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    ordered = sorted(Counter(pairs).items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"bigram": bg, "count": cnt} for bg, cnt in ordered[:top_n]]


@dataclass
class CleanedRecords:
    texts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    dropped_missing_text: int = 0
    dropped_missing_label: int = 0


def clean_records(frame: Any, text_col: str, label_col: str) -> CleanedRecords:
    """Drop blank text/label rows, counting them (no refusal here)."""
    cleaned = CleanedRecords()
    for _, row in frame.iterrows():
        raw_text, raw_label = row[text_col], row[label_col]
        text_str = "" if pd.isna(raw_text) else str(raw_text)
        label_str = "" if pd.isna(raw_label) else str(raw_label).strip()
        if not text_str.strip():
            cleaned.dropped_missing_text += 1
            continue
        if not label_str:
            cleaned.dropped_missing_label += 1
            continue
        cleaned.texts.append(text_str)
        cleaned.labels.append(label_str)
    return cleaned


def parse_entity_column(values: list[Any]) -> tuple[dict[str, int] | None, str | None]:
    """Count entity labels from a JSON-spans or BIO column.

    Returns (frequencies, None) on success, else (None, reason).
    """
    import json

    freq: Counter[str] = Counter()
    parsed_any = False
    for value in values:
        if isinstance(value, list):  # pre-parsed span lists (programmatic frames)
            labels = []
            for item in value:
                if isinstance(item, dict) and item.get("label"):
                    labels.append(str(item["label"]))
                elif isinstance(item, str):
                    labels.append(item)
            parsed_any = True
            freq.update(labels)
            continue
        if pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        labels: list[str] = []
        if raw.startswith("["):
            try:
                items = json.loads(raw)
            except Exception:
                return None, "entity column is not valid JSON nor BIO tags"
            if not isinstance(items, list):
                return None, "entity column JSON rows must be lists"
            for item in items:
                if isinstance(item, dict) and item.get("label"):
                    labels.append(str(item["label"]))
                elif isinstance(item, str):
                    labels.append(item)
            parsed_any = True
        else:  # BIO tags: "O O B-DATE I-DATE O"
            for tag in raw.split():
                if tag.startswith(("B-", "I-")):
                    labels.append(tag[2:])
            parsed_any = True
        freq.update(labels)
    if not parsed_any:
        return None, "entity column present but holds no parseable annotations"
    return dict(freq), None


def compute_text_analytics(texts: list[str], top_n: int = TOP_N) -> dict[str, Any]:
    char_lengths = [len(t) for t in texts]
    all_tokens: list[str] = []
    sentence_word_counts: list[int] = []
    for text in texts:
        tokens = tokenize_words(text)
        all_tokens.extend(tokens)
        for sentence in segment_sentences(text):
            sentence_word_counts.append(len(tokenize_words(sentence)))
    return {
        "record_count": len(texts),
        "average_chars": _average([float(c) for c in char_lengths]),
        "min_chars": min(char_lengths) if char_lengths else 0,
        "max_chars": max(char_lengths) if char_lengths else 0,
        "char_length_histogram": histogram(char_lengths, CHAR_BIN_EDGES),
        "sentence_word_counts": {
            "average_words": _average([float(c) for c in sentence_word_counts]),
            "total_sentences": len(sentence_word_counts),
            "histogram": histogram(sentence_word_counts, WORD_BIN_EDGES),
        },
        "frequent_words": frequent_terms(all_tokens, top_n),
        "frequent_bigrams": frequent_bigrams(all_tokens, top_n),
        "vocabulary_size": len(set(all_tokens)),
    }


def distribution_chart(counter: dict[str, int]) -> dict[str, Any]:
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "labels": [label for label, _ in ordered],
        "counts": [count for _, count in ordered],
        "total": sum(counter.values()),
    }


def analyze_dataset_file(
    path: str, text_col: str, label_col: str, entity_col: str | None = None
) -> dict[str, Any]:
    """Full chart-ready analytics for a REAL dataset file.

    Raises DatasetError for missing/unreadable/mis-shaped files.
    """
    frame = load_dataset(path, text_col, label_col)  # schema-checked
    cleaned = clean_records(frame, text_col, label_col)
    duplicate_rows = int(frame.duplicated().sum())
    intent_counter = dict(Counter(cleaned.labels))
    payload: dict[str, Any] = {
        "available": True,
        "source": path,
        "records": {
            "total": int(len(frame)),
            "used": len(cleaned.texts),
            "dropped_missing_text": cleaned.dropped_missing_text,
            "dropped_missing_label": cleaned.dropped_missing_label,
            "duplicate_rows": duplicate_rows,
        },
        "intent_distribution": distribution_chart(intent_counter),
        "text_analytics": compute_text_analytics(cleaned.texts),
    }
    if entity_col and entity_col in frame.columns:
        freq, reason = parse_entity_column(list(frame[entity_col]))
        payload["entity_distribution"] = (
            distribution_chart(freq) if freq is not None else None
        )
        payload["entity_distribution_reason"] = reason
    else:
        payload["entity_distribution"] = None
        payload["entity_distribution_reason"] = (
            "no entity annotation column supplied or present"
        )
    return payload


def read_dataset_records(
    path: str, text_col: str, label_col: str, page: int, page_size: int
) -> dict[str, Any]:
    """Paginated raw records from a REAL dataset file (1-based page).

    Raises DatasetError for missing/unreadable/mis-shaped files. NaN cells
    become ``None`` so the payload is strict-JSON safe.
    """
    import json

    frame = load_dataset(path, text_col, label_col)  # schema-checked
    total = int(len(frame))
    start = (page - 1) * page_size
    window = frame.iloc[start : start + page_size]
    records = json.loads(window.to_json(orient="records", force_ascii=False))
    return {
        "available": True,
        "source": path,
        "page": page,
        "page_size": page_size,
        "total_records": total,
        "returned_records": len(records),
        "records": records,
    }


def ner_evaluation_status() -> dict[str, Any]:
    """Honest project-level NER status: no gold spans exist to compare."""
    return {
        "evaluated": False,
        "reason": (
            "No annotated entity spans exist in this workspace, so "
            "entity-level precision/recall/F1 cannot be computed. Supply gold "
            "spans to POST /api/v1/analytics/ner for an exact span+label "
            "comparison."
        ),
        "matching_criteria": (
            "exact match on (start, end, label) against the original text"
        ),
    }
