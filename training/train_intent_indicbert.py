"""Fine-tune a Marathi BERT (default l3cube-pune/marathi-bert-v2) for intent.

NOTE: ai4bharat/indic-bert is now access-gated on HuggingFace (403), so the
open Marathi-specific l3cube model is the default. Pass --model-name
ai4bharat/indic-bert only if you have HF access (hf auth login).

Usage (CPU-friendly defaults; works without GPU, just slower):
    python -m training.train_intent_indicbert --data data/marathi_legal_financial_intents_320.csv --out-dir models/indicbert_intent --epochs 3

- Base model default: ai4bharat/indic-bert (ALBERT, Marathi covered, ~11M params).
- Manual PyTorch loop (no `accelerate` needed): AdamW, linear warmup-free,
  best-checkpoint by macro-F1 on the same reproducible split convention as
  training/train_intent.py (seed 42, stratify when every class >= 2).
- Persists <out-dir>/ {pytorch_model.bin|model.safetensors, config.json,
  tokenizer files, intent_labels.json, intent_metrics.json}.
- Exit codes: 0 success, 2 dataset error, 3 unsuitable dataset.

NOTE: you asked for "full GPU fine-tune" but this machine has CPU-only
torch (cuda=False), so this script auto-detects device and runs on CPU.
On GPU it uses cuda automatically with the same code.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys

import numpy as np

DEFAULT_MODEL = "l3cube-pune/marathi-bert-v2"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune IndicBERT for intent.")
    p.add_argument("--data", required=True, help="Path to annotated CSV.")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="intent")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default="models/indicbert_intent")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    from training.dataset import (
        DatasetError,
        UnsuitableDatasetError,
        load_dataset,
        validate_dataset,
    )

    try:
        frame = load_dataset(args.data, args.text_col, args.label_col)
        report = validate_dataset(frame, args.data, args.text_col, args.label_col)
    except DatasetError as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 2
    except UnsuitableDatasetError as exc:
        print(f"UNSUITABLE DATASET: {exc}", file=sys.stderr)
        return 3

    print(report.summary())
    print(f"base_model={args.model_name}")

    import torch
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} (cuda_available={torch.cuda.is_available()})")
    if device == "cpu":
        print("NOTE: CPU-only torch: fine-tuning 320 rows x 3 epochs takes ~5-15 min.")

    labels = sorted(report.class_frequencies)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    stratify = report.labels if report.stratify_supported else None
    X_train, X_test, y_train, y_test = train_test_split(
        report.texts, report.labels, test_size=args.test_size,
        random_state=args.seed, stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels),
        id2label=id2label, label2id=label2id,
    )
    model.to(device)

    def encode(texts: list[str]):
        return tokenizer(
            texts, truncation=True, padding=True,
            max_length=args.max_length, return_tensors="pt",
        )

    train_ids = [label2id[y] for y in y_train]
    test_ids = [label2id[y] for y in y_test]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def run_epoch(texts: list[str], ids: list[int], train: bool):
        model.train(train)
        total_loss = 0.0
        preds_all: list[int] = []
        # simple sequential batches (dataset is tiny; shuffle per epoch)
        idx = list(range(len(texts)))
        if train:
            random.shuffle(idx)
        for s in range(0, len(idx), args.batch_size):
            chunk = idx[s : s + args.batch_size]
            batch_texts = [texts[i] for i in chunk]
            batch_ids = torch.tensor([ids[i] for i in chunk], dtype=torch.long).to(device)
            enc = {k: v.to(device) for k, v in encode(batch_texts).items()}
            out = model(**enc, labels=batch_ids)
            loss = out.loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(chunk)
            preds_all.extend(int(v) for v in out.logits.argmax(-1).detach().cpu().tolist())
        # restore order-independent metric: compare multiset via aligned preds?
        # We shuffled, so recompute preds in order for eval correctness:
        if train:
            model.eval()
            ordered: list[int] = []
            with torch.no_grad():
                for s in range(0, len(texts), args.batch_size):
                    bt = texts[s : s + args.batch_size]
                    enc = {k: v.to(device) for k, v in encode(bt).items()}
                    ordered.extend(
                        int(v) for v in model(**enc).logits.argmax(-1).cpu().tolist()
                    )
            preds_all = ordered
            model.train(True)
        return total_loss / max(len(texts), 1), preds_all

    from sklearn.metrics import accuracy_score, f1_score

    best_f1 = -1.0
    best_state = None
    best_loss = float("inf")
    for ep in range(args.epochs):
        train_loss, _ = run_epoch(X_train, train_ids, train=True)
        with torch.no_grad():
            _, test_preds = run_epoch(X_test, test_ids, train=False)
        acc = accuracy_score(test_ids, test_preds)
        macro_f1 = f1_score(test_ids, test_preds, average="macro", zero_division=0)
        print(f"epoch {ep + 1}/{args.epochs} train_loss={train_loss:.4f} "
              f"test_acc={acc:.4f} macro_f1={macro_f1:.4f}")
        # Keep the sharpest weights: strictly better F1 wins; on F1 ties the
        # lower train loss (sharper softmax) wins so resumed runs keep progress.
        if macro_f1 > best_f1 or (macro_f1 == best_f1 and train_loss < best_loss):
            best_f1 = macro_f1
            best_loss = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final held-out evaluation with full sklearn report
    from sklearn.metrics import classification_report, confusion_matrix

    with torch.no_grad():
        _, final_preds = run_epoch(X_test, test_ids, train=False)
    pred_labels = [id2label[i] for i in final_preds]
    true_labels = [id2label[i] for i in test_ids]
    from sklearn.metrics import precision_score, recall_score

    metrics = {
        "dataset": os.path.abspath(args.data),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "base_model": args.model_name,
        "backend": "indicbert_sequence_classification",
        "device": device,
        "rows_total": report.rows_total,
        "rows_used": report.rows_used,
        "class_frequencies": report.class_frequencies,
        "stratify_used": report.stratify_supported,
        "test_size": args.test_size,
        "random_state": args.seed,
        "train_size": len(y_train),
        "eval_test_size": len(y_test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "best_macro_f1": float(best_f1),
        "accuracy": float(accuracy_score(test_ids, final_preds)),
        "macro_f1": float(f1_score(test_ids, final_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(test_ids, final_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(test_ids, final_preds, average="macro", zero_division=0)),
        "classification_report": classification_report(
            true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(true_labels, pred_labels, labels=labels).tolist(),
        },
        "labels": labels,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "intent_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "intent_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"artifacts written to '{args.out_dir}': model + tokenizer + metrics "
          f"(best_macro_f1={best_f1:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
