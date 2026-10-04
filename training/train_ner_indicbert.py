"""Fine-tune IndicBERT for Marathi NER (token classification, BIO).

Usage:
    python -m training.make_ner_silver  # (already produces data/silver/ner_silver.csv)
    python -m training.train_ner_indicbert --data data/silver/ner_silver.csv --out-dir models/ner_indicbert --epochs 3

- Base model default: ai4bharat/indic-bert (same encoder as intent path).
- Converts silver char spans -> BIO tags aligned to HF subwords via
  offset_mapping; subword continuations + specials get label -100.
- Manual PyTorch loop (no accelerate). Best checkpoint by silver-held-out
  span-F1 (exact start/end/label, decoded from BIO).
- Persists <out-dir>/ {model, tokenizer, ner_labels.json, ner_metrics.json}.
- Exit codes: 0 success, 2 dataset error, 3 unsuitable dataset.
- Scores are SILVER-held-out only; do not present as human-gold evaluation.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune IndicBERT for NER (BIO).")
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--entity-col", default="entities")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default="models/ner_indicbert")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-size", type=float, default=0.2)
    return p


def load_silver(path: str, text_col: str, entity_col: str):
    import csv

    rows: list[tuple[str, list[tuple[int, int, str]]]] = []
    dropped = 0
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
            raise SystemExit(f"DATASET ERROR: column(s) {missing} not in {reader.fieldnames}.")
        for record in reader:
            total += 1
            text = record.get(text_col) or ""
            raw = record.get(entity_col) or ""
            if not text.strip() or not raw.strip():
                dropped += 1
                continue
            try:
                items = json.loads(raw)
            except Exception:
                dropped += 1
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
                dropped += 1
                continue
            rows.append((text, spans))
    return rows, total, dropped


def bio_decode(text: str, offsets, pred_ids: list[int], id2label: dict[int, str]):
    """BIO tag ids + token offsets -> char spans (start, end, label)."""
    spans: list[tuple[int, int, str]] = []
    cur_label: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    for (s, e), pid in zip(offsets, pred_ids):
        if s == e == 0:
            # special token: close any open span
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))  # type: ignore[arg-type]
                cur_label, cur_start, cur_end = None, None, None
            continue
        tag = id2label[int(pid)]
        if tag == "O":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))  # type: ignore[arg-type]
                cur_label, cur_start, cur_end = None, None, None
        elif tag.startswith("B-"):
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))  # type: ignore[arg-type]
            cur_label = tag[2:]
            cur_start, cur_end = int(s), int(e)
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur_label == lab and cur_end == int(s):
                cur_end = int(e)
            elif cur_label == lab:
                # subword gap (whitespace inside?) -- extend end anyway
                cur_end = int(e)
            else:
                if cur_label is not None:
                    spans.append((cur_start, cur_end, cur_label))  # type: ignore[arg-type]
                cur_label = lab
                cur_start, cur_end = int(s), int(e)
    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))  # type: ignore[arg-type]
    return [(s, e, lab) for s, e, lab in spans if s is not None and e is not None]


def span_f1(pred_sets: list[set], gold_sets: list[set]) -> dict:
    from collections import Counter

    correct = n_pred = n_gold = 0
    for ps, gs in zip(pred_sets, gold_sets):
        correct += len(ps & gs)
        n_pred += len(ps)
        n_gold += len(gs)
    p = correct / n_pred if n_pred else 0.0
    r = correct / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "predicted_spans": n_pred, "gold_spans": n_gold, "correct": correct}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    rows, total, dropped = load_silver(args.data, args.text_col, args.entity_col)
    if len(rows) < 20:
        print(f"UNSUITABLE DATASET: only {len(rows)} usable rows (< 20).", file=sys.stderr)
        return 3
    ent_labels = sorted({lab for _, spans in rows for _, _, lab in spans})
    if not ent_labels:
        print("UNSUITABLE DATASET: no entity labels found.", file=sys.stderr)
        return 3
    tag_list = ["O"] + [f"B-{l}" for l in ent_labels] + [f"I-{l}" for l in ent_labels]
    tag2id = {t: i for i, t in enumerate(tag_list)}
    id2tag = {i: t for t, i in tag2id.items()}

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"labels={ent_labels} rows={len(rows)} device={device}")

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = max(1, int(len(rows) * args.test_size))
    test_idx = set(idx[:n_test])
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_idx]
    test_rows = [rows[i] for i in range(len(rows)) if i in test_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(tag_list),
        id2label=id2tag, label2id=tag2id,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def encode_batch(texts: list[str], span_lists: list[list[tuple[int, int, str]]] | None):
        enc = tokenizer(
            texts, truncation=True, padding=True, max_length=args.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        labels = None
        if span_lists is not None:
            blist = []
            for k, text in enumerate(texts):
                char_to_tag = ["O"] * len(text)
                # mark B/I per char then map to first subword token covering char
                for s, e, lab in span_lists[k]:
                    for c in range(s, min(e, len(text))):
                        char_to_tag[c] = f"B-{lab}" if c == s else f"I-{lab}"
                seq_tags: list[int] = []
                for so, eo in offsets[k].tolist():
                    if so == eo == 0:
                        seq_tags.append(-100)
                    else:
                        # label from first char of token
                        c = min(int(so), len(text) - 1)
                        seq_tags.append(tag2id.get(char_to_tag[c], tag2id["O"]))
                blist.append(seq_tags)
            labels = torch.tensor(blist, dtype=torch.long)
        return enc, offsets, labels

    def eval_spans(eval_rows) -> tuple[dict, float]:
        model.eval()
        pred_sets: list[set] = []
        gold_sets: list[set] = [set(spans) for _, spans in eval_rows]
        with torch.no_grad():
            for s in range(0, len(eval_rows), args.batch_size):
                chunk = eval_rows[s : s + args.batch_size]
                texts = [t for t, _ in chunk]
                enc, offsets, _ = encode_batch(texts, None)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.cpu()
                for b in range(len(texts)):
                    seqlen = int((enc["attention_mask"][b]).sum().item())
                    pids = logits[b, :seqlen].argmax(-1).tolist()
                    offs = offsets[b, :seqlen].tolist()
                    # drop [CLS]/[SEP] zero-offsets inside decode
                    pred_sets.append(set(bio_decode(texts[b], offs, pids, id2tag)))
        return pred_sets, gold_sets

    best_f1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train(True)
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        total_loss = 0.0
        nb = 0
        for s in range(0, len(order), args.batch_size):
            chunk_idx = order[s : s + args.batch_size]
            texts = [train_rows[i][0] for i in chunk_idx]
            spans_l = [train_rows[i][1] for i in chunk_idx]
            enc, _, labs = encode_batch(texts, spans_l)
            enc = {k: v.to(device) for k, v in enc.items()}
            labs = labs.to(device)
            out = model(**enc, labels=labs)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total_loss += float(out.loss.detach().cpu())
            nb += 1
        pred_sets, gold_sets = eval_spans(test_rows)
        m = span_f1(pred_sets, gold_sets)
        print(f"epoch {ep + 1}/{args.epochs} loss={total_loss / max(nb, 1):.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_sets, gold_sets = eval_spans(test_rows)
    final = span_f1(pred_sets, gold_sets)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "ner_labels.json"), "w", encoding="utf-8") as fh:
        json.dump({"entities": ent_labels, "tags": tag_list}, fh, ensure_ascii=False, indent=2)
    payload = {
        "dataset": os.path.abspath(args.data),
        "data_kind": "SILVER synthetic -- NOT human gold",
        "base_model": args.model_name,
        "backend": "indicbert_token_classification_bio",
        "device": device,
        "rows_total": total,
        "rows_used": len(rows),
        "rows_dropped": dropped,
        "entities": ent_labels,
        "tags": tag_list,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "silver_heldout": final,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "ner_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"silver-held-out F1={final['f1']:.3f} artifacts -> '{args.out_dir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
