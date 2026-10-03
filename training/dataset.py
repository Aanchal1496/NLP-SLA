"""Dataset loading + validation for intent classification.

Operates ONLY on a real, user-supplied annotated file. This module never
creates, synthesizes, or imputes training records: rows with missing/blank
text or labels are dropped and counted, and unsuitable datasets raise
instead of producing a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

MIN_USABLE_ROWS = 6  # below this, no honest train/test split is possible


class DatasetError(Exception):
    """The dataset file is missing, unreadable, or has the wrong schema."""


class UnsuitableDatasetError(Exception):
    """The dataset exists but cannot support supervised training."""


@dataclass
class DatasetReport:
    path: str
    text_col: str
    label_col: str
    rows_total: int = 0
    rows_used: int = 0
    rows_dropped_missing_text: int = 0
    rows_dropped_missing_label: int = 0
    class_frequencies: dict[str, int] = field(default_factory=dict)
    insufficient_classes: list[str] = field(default_factory=list)
    stratify_supported: bool = False
    texts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"dataset: {self.path}",
            f"rows_total={self.rows_total} rows_used={self.rows_used} "
            f"dropped_text={self.rows_dropped_missing_text} "
            f"dropped_label={self.rows_dropped_missing_label}",
            f"classes={len(self.class_frequencies)} "
            f"stratify_supported={self.stratify_supported}",
        ]
        for label, count in sorted(
            self.class_frequencies.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            flag = "  <-- INSUFFICIENT (<2 samples)" if count < 2 else ""
            lines.append(f"  {label}: {count}{flag}")
        return "\n".join(lines)


def decide_stratify(class_frequencies: dict[str, int]) -> bool:
    """Stratification needs >= 2 samples in EVERY class (sklearn rule)."""
    return bool(class_frequencies) and all(c >= 2 for c in class_frequencies.values())


def load_dataset(path: str, text_col: str, label_col: str) -> Any:
    """Read the annotated file; schema errors raise DatasetError."""
    import pandas as pd

    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DatasetError(
            f"Dataset file not found: '{path}'. "
            "Place the annotated project CSV at data/raw/ and pass --data."
        ) from exc
    except Exception as exc:
        raise DatasetError(f"Could not read dataset file '{path}': {exc}") from exc
    if frame.empty:
        raise DatasetError(f"Dataset file '{path}' contains no rows.")
    missing = [c for c in (text_col, label_col) if c not in frame.columns]
    if missing:
        raise DatasetError(
            f"Required column(s) {missing} not in dataset. "
            f"Available columns: {list(frame.columns)}. "
            f"Pass --text-col/--label-col matching the real schema."
        )
    return frame


def validate_dataset(frame: Any, path: str, text_col: str, label_col: str) -> DatasetReport:
    """Drop invalid rows (counting them), report frequencies, gate training."""
    report = DatasetReport(
        path=path, text_col=text_col, label_col=label_col, rows_total=len(frame)
    )
    texts: list[str] = []
    labels: list[str] = []
    for _, row in frame.iterrows():
        text = row[text_col]
        label = row[label_col]
        text_ok = not (text is None or (isinstance(text, float) and text != text))
        label_ok = not (label is None or (isinstance(label, float) and label != label))
        text_str = str(text) if text_ok else ""
        label_str = str(label).strip() if label_ok else ""
        if not text_str.strip():
            report.rows_dropped_missing_text += 1
            continue
        if not label_str:
            report.rows_dropped_missing_label += 1
            continue
        texts.append(text_str)
        labels.append(label_str)
    report.texts = texts
    report.labels = labels
    report.rows_used = len(texts)
    report.class_frequencies = dict(Counter(labels))
    report.insufficient_classes = sorted(
        [lab for lab, cnt in report.class_frequencies.items() if cnt < 2]
    )
    report.stratify_supported = decide_stratify(report.class_frequencies)
    if report.rows_used == 0:
        raise UnsuitableDatasetError(
            "No usable rows after dropping missing text/labels "
            f"(total={report.rows_total}). Cannot train."
        )
    if len(report.class_frequencies) < 2:
        raise UnsuitableDatasetError(
            f"Only {len(report.class_frequencies)} intent label(s) present "
            f"{sorted(report.class_frequencies)}; supervised classification "
            "needs at least 2 distinct labels."
        )
    if report.rows_used < MIN_USABLE_ROWS:
        raise UnsuitableDatasetError(
            f"Only {report.rows_used} usable rows (< {MIN_USABLE_ROWS}); "
            "no honest train/test split is possible. Collect more annotations."
        )
    return report
