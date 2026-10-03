"""Train a spaCy NER model on SILVER entity annotations.

Usage:
    python -m training.train_ner --data data/silver/ner_silver.csv \\
        --text-col text --entity-col entities --out-dir models/ner

Input CSV columns: text + entities (JSON list of
{"text","label","start","end"} char offsets, as produced by
training/make_ner_silver.py). Offsets are validated
(text[start:end] == span text); misaligned rows are dropped and counted.

Split: deterministic 80/20 shuffle (seed 42). Blank multilingual pipeline
(spacy.blank("xx") + tok2vec + ner) -- no pretrained weights, no external
download. Persist:
    <out-dir>/model/            spaCy pipeline (config.cfg + weights)
    <out-dir>/ner_labels.json   sorted label list
    <out-dir>/ner_metrics.json  silver-held-out P/R/F1 (exact span+label)

Exit codes: 0 success, 2 dataset error, 3 unsuitable dataset.
Scores are SILVER-held-out only; do not present as human-gold evaluation.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import sys

MIN_USABLE_ROWS = 20
MIN_SPANS = 20


def load_silver(path: str, text_col: str, entity_col: str):
    rows: list[tuple[str, list[tuple[int, int, str]]]] = []
    dropped_missing = 0
    dropped_misaligned = 0
    total = 0
    try:
        fh = open(path, encoding="utf-8", newline="")
    except FileNotFoundError:
        raise SystemExit(f"DATASET ERROR: file not found: '{path}'.")
    with fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"DATASET ERROR: '{path}' has no header.")
        missing = [c for c in (text_col, entity_col) if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"DATASET ERROR: column(s) {missing} not in {reader.fieldnames}."
            )
        for record in reader:
            total += 1
            text = record.get(text_col) or ""
            raw = record.get(entity_col) or ""
            if not text.strip() or not raw.strip():
                dropped_missing += 1
                continue
            try:
                items = json.loads(raw)
            except Exception:
                dropped_misaligned += 1
                continue
            spans: list[tuple[int, int, str]] = []
            ok = True
            for item in items:
                try:
                    s, e, lab = int(item["start"]), int(item["end"]), str(item["label"])
                    expect = item.get("text", text[s:e])
                except Exception:
                    ok = False
                    break
                if not (0 <= s < e <= len(text)) or text[s:e] != expect:
                    ok = False
                    break
                spans.append((s, e, lab))
            if not ok:
                dropped_misaligned += 1
                continue
            rows.append((text, spans))
    return rows, total, dropped_missing, dropped_misaligned


def to_spacy_examples(nlp, rows):
    from spacy.training import Example

    examples = []
    for text, spans in rows:
        doc = nlp.make_doc(text)
        ents = []
        for s, e, lab in spans:
            span = doc.char_span(s, e, label=lab, alignment_mode="expand")
            if span is not None:
                ents.append(span)
        doc.ents = ents
        examples.append(Example.from_dict(doc, {"entities": [(s, e, l) for s, e, l in spans]}))
    return examples


def evaluate(nlp, rows) -> dict:
    """Exact span+label P/R/F1 counted PER OCCURRENCE (multiset).

    Earlier versions used set() keys, which collapsed repeats of the same
    (start, end, label) across template-generated rows and inflated scores.
    We count each occurrence separately via Counter intersection.
    """
    from collections import Counter

    # Score PER DOCUMENT then aggregate: the same (start, end, label) in
    # two different sentences are distinct occurrences and must not collapse.
    correct = 0
    n_pred = 0
    n_gold = 0
    pred_all: Counter[tuple[int, int, str]] = Counter()
    gold_all: Counter[tuple[int, int, str]] = Counter()
    for text, spans in rows:
        doc = nlp(text)
        pred_doc: Counter[tuple[int, int, str]] = Counter()
        gold_doc: Counter[tuple[int, int, str]] = Counter()
        for ent in doc.ents:
            pred_doc[(ent.start_char, ent.end_char, ent.label_)] += 1
        for s, e, lab in spans:
            gold_doc[(s, e, lab)] += 1
        correct += sum((pred_doc & gold_doc).values())
        n_pred += sum(pred_doc.values())
        n_gold += sum(gold_doc.values())
        pred_all += pred_doc
        gold_all += gold_doc
    precision = correct / n_pred if n_pred else 0.0
    recall = correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    labels = sorted({k[2] for k in gold_all} | {k[2] for k in pred_all})
    per: dict[str, dict] = {}
    for lab in labels:
        p = sum(c for k, c in pred_all.items() if k[2] == lab)
        g = sum(c for k, c in gold_all.items() if k[2] == lab)
        c = sum((Counter({k: v for k, v in pred_all.items() if k[2] == lab}) & Counter(
            {k: v for k, v in gold_all.items() if k[2] == lab})).values())
        pr = c / p if p else 0.0
        rc = c / g if g else 0.0
        f = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        per[lab] = {"precision": pr, "recall": rc, "f1": f,
                    "predicted": p, "gold": g, "correct": c}
    return {"precision": precision, "recall": recall, "f1": f1,
            "predicted_spans": n_pred, "gold_spans": n_gold,
            "correct": correct, "per_label": per}


def train_and_persist(data, text_col, entity_col, out_dir,
                      seed=42, n_iter=30, test_size=0.2):
    import spacy
    from spacy.util import minibatch, compounding

    rows, total, drop_missing, drop_mis = load_silver(data, text_col, entity_col)
    if len(rows) < MIN_USABLE_ROWS:
        print(f"UNSUITABLE DATASET: only {len(rows)} usable rows (< {MIN_USABLE_ROWS}).", file=sys.stderr)
        raise SystemExit(3)
    n_spans = sum(len(s) for _, s in rows)
    if n_spans < MIN_SPANS:
        print(f"UNSUITABLE DATASET: only {n_spans} spans (< {MIN_SPANS}).", file=sys.stderr)
        raise SystemExit(3)
    labels = sorted({lab for _, spans in rows for _, _, lab in spans})
    if len(labels) < 1:
        print("UNSUITABLE DATASET: no entity labels found.", file=sys.stderr)
        raise SystemExit(3)

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = max(1, int(len(rows) * test_size))
    test_idx = set(idx[:n_test])
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_idx]
    test_rows = [rows[i] for i in range(len(rows)) if i in test_idx]

    nlp = spacy.blank("xx")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for lab in labels:
        ner.add_label(lab)

    train_examples = to_spacy_examples(nlp, train_rows)
    optimizer = nlp.begin_training()
    for it in range(n_iter):
        rng.shuffle(train_examples)
        losses: dict = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            nlp.update(batch, drop=0.35, losses=losses, sgd=optimizer)
        # ascii-safe log
        print(f"iter {it + 1}/{n_iter} losses={ {k: round(float(v), 2) for k, v in losses.items()} }")

    metrics = evaluate(nlp, test_rows)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model")
    nlp.to_disk(model_path)
    with open(os.path.join(out_dir, "ner_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False, indent=2)
    payload = {
        "dataset": os.path.abspath(data),
        "data_kind": "SILVER synthetic (make_ner_silver.py) -- NOT human gold",
        "text_col": text_col,
        "entity_col": entity_col,
        "rows_total": total,
        "rows_used": len(rows),
        "rows_dropped_missing": drop_missing,
        "rows_dropped_misaligned": drop_mis,
        "labels": labels,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "n_iter": n_iter,
        "seed": seed,
        "silver_heldout": metrics,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "ner_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"silver-held-out P={metrics['precision']:.3f} "
          f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
          f"({metrics['correct']}/{metrics['gold_spans']} correct)")
    print(f"artifacts written to '{out_dir}': model/, ner_labels.json, ner_metrics.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train spaCy NER on silver CSV.")
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--entity-col", default="entities")
    p.add_argument("--out-dir", default="models/ner")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_and_persist(args.data, args.text_col, args.entity_col,
                          args.out_dir, args.seed, args.n_iter, args.test_size)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
