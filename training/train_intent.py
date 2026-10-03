"""Train + compare TF-IDF intent classifiers on the REAL project dataset.

Usage:
    python -m training.train_intent --data data/raw/intents.csv \\
        --text-col text --label-col intent --out-dir models

Compares TF-IDF + LogisticRegression vs TF-IDF + LinearSVC on one
reproducible split (fixed seed; stratification only when every class has
>= 2 samples), selects the higher macro-F1 (ties -> LogisticRegression,
which yields probabilities), and persists:
    <out-dir>/intent_pipeline.joblib   fitted Pipeline (vectorizer+model)
    <out-dir>/intent_labels.json       sorted label list
    <out-dir>/intent_metrics.json      split config, both models' actual
                                       evaluation scores, class frequencies

Vectorization is fitted on TRAIN data only (single Pipeline fit) -- no
leakage. Exit codes: 0 success, 2 dataset error, 3 unsuitable dataset.
Metrics are computed from the actual held-out split; nothing is fabricated.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

import joblib

from app.services.intent_classifier import (
    LABELS_FILENAME,
    METRICS_FILENAME,
    PIPELINE_FILENAME,
    build_pipelines,
)
from training.dataset import (
    DatasetError,
    UnsuitableDatasetError,
    load_dataset,
    validate_dataset,
)

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def split_data(
    texts: list[str], labels: list[str], test_size: float,
    random_state: int, stratify_supported: bool,
) -> tuple[Any, Any, Any, Any]:
    from sklearn.model_selection import train_test_split

    stratify = labels if stratify_supported else None
    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=stratify,
    )


def evaluate(
    pipeline: Any, X_test: Any, y_test: Any, labels: list[str]
) -> dict[str, Any]:
    """Held-out scores only -- the training split is never evaluated here."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(
            precision_score(y_test, pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test, pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        },
        "test_size": int(len(y_test)),
    }


def select_best(results: dict[str, dict[str, Any]]) -> str:
    ordered = sorted(results.items(), key=lambda kv: (-kv[1]["macro_f1"], kv[0]))
    # Deterministic tie-break favouring the probability-yielding model.
    top_score = ordered[0][1]["macro_f1"]
    tied = sorted([name for name, res in ordered if res["macro_f1"] == top_score])
    if "tfidf_logreg" in tied:
        return "tfidf_logreg"
    return tied[0]


def train_and_persist(
    data: str, text_col: str, label_col: str, out_dir: str,
    test_size: float = DEFAULT_TEST_SIZE, random_state: int = DEFAULT_SEED,
) -> dict[str, Any]:
    frame = load_dataset(data, text_col, label_col)
    report = validate_dataset(frame, data, text_col, label_col)
    print(report.summary())
    if not report.stratify_supported:
        print(
            "WARNING: stratification disabled "
            f"(insufficient classes: {report.insufficient_classes}). "
            "Split is unstratified; minority classes may miss the test set."
        )
    X_train, X_test, y_train, y_test = split_data(
        report.texts, report.labels, test_size, random_state,
        report.stratify_supported,
    )
    pipelines = build_pipelines(random_state=random_state)
    eval_labels = sorted(report.class_frequencies)
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)  # vectorizer sees TRAIN text only
        results[name] = evaluate(pipeline, X_test, y_test, eval_labels)
        print(f"{name}: accuracy={results[name]['accuracy']:.4f} "
              f"macro_f1={results[name]['macro_f1']:.4f}")
    selected = select_best(results)
    print(f"selected: {selected}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipelines[selected], os.path.join(out_dir, PIPELINE_FILENAME))
    labels = sorted(report.class_frequencies)
    with open(os.path.join(out_dir, LABELS_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False, indent=2)
    metrics = {
        "dataset": os.path.abspath(data),
        "text_col": text_col,
        "label_col": label_col,
        "rows_total": report.rows_total,
        "rows_used": report.rows_used,
        "rows_dropped_missing_text": report.rows_dropped_missing_text,
        "rows_dropped_missing_label": report.rows_dropped_missing_label,
        "class_frequencies": report.class_frequencies,
        "insufficient_classes": report.insufficient_classes,
        "stratify_used": report.stratify_supported,
        "test_size": test_size,
        "random_state": random_state,
        "train_size": int(len(y_train)),
        "train_size": int(len(y_train)),
        "results": results,
        "selected_model": selected,
        "labels": labels,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, METRICS_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"artifacts written to '{out_dir}': "
          f"{PIPELINE_FILENAME}, {LABELS_FILENAME}, {METRICS_FILENAME}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Marathi intent classifier.")
    parser.add_argument("--data", required=True, help="Path to annotated CSV.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="intent")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_and_persist(
            data=args.data, text_col=args.text_col, label_col=args.label_col,
            out_dir=args.out_dir, test_size=args.test_size,
            random_state=args.seed,
        )
    except DatasetError as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 2
    except UnsuitableDatasetError as exc:
        print(f"UNSUITABLE DATASET: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
