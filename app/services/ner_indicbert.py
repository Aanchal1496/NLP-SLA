"""IndicBERT NER inference (BIO token classification, merges with baseline).

Artifacts in <model_dir>/ner_indicbert/ (from training/train_ner_indicbert.py).
Decodes BIO tags back to char spans via tokenizer offset_mapping so
``text[start:end] == span text`` holds. ``predict_merged()`` unions baseline
regex+gazetteer spans with IndicBERT spans (exact-duplicate labels collapse
to method="indicbert_ner"; baseline-only overlaps are kept as-is).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from app.core.config import settings

SUBDIR = "ner_indicbert"
LABELS_FILE = "ner_labels.json"
METRICS_FILE = "ner_metrics.json"


class ModelUnavailableError(Exception):
    """Raised when no IndicBERT NER artifacts can be loaded."""


@dataclass
class IndicBertMention:
    text: str
    label: str
    start: int
    end: int
    method: str = "indicbert_ner"


@dataclass
class IndicBertNerResult:
    original_text: str
    entities: list[IndicBertMention] = field(default_factory=list)


_cache: dict[str, dict] = {}


def clear_cache() -> None:
    _cache.clear()


def model_path(model_dir: str | None = None) -> str:
    return os.path.join(model_dir or settings.model_dir, SUBDIR)


def load_indicbert_ner(model_dir: str | None = None) -> dict:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    key = model_dir or settings.model_dir
    if key in _cache:
        return _cache[key]
    root = model_path(model_dir)
    if not os.path.isdir(root):
        raise ModelUnavailableError(
            f"No IndicBERT NER model at '{root}'. Train one with: "
            "python -m training.train_ner_indicbert "
            "--data data/silver/ner_silver.csv --out-dir models/ner_indicbert"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(root)
        model = AutoModelForTokenClassification.from_pretrained(root)
        model.eval()
    except Exception as exc:
        raise ModelUnavailableError(
            f"IndicBERT NER model at '{root}' could not be loaded ({exc})."
        ) from exc
    id2tag = {int(k): str(v) for k, v in getattr(model.config, "id2label", {}).items()}
    bundle = {"model": model, "tokenizer": tokenizer, "id2tag": id2tag, "root": root}
    _cache[key] = bundle
    return bundle


def _decode(text: str, offsets: list[list[int]], pred_ids: list[int], id2tag: dict[int, str]):
    spans: list[tuple[int, int, str]] = []
    cur_lab: str | None = None
    cur_s: int | None = None
    cur_e: int | None = None

    def flush():
        nonlocal cur_lab, cur_s, cur_e
        if cur_lab is not None and cur_s is not None and cur_e is not None:
            spans.append((cur_s, cur_e, cur_lab))
        cur_lab, cur_s, cur_e = None, None, None

    for (s, e), pid in zip(offsets, pred_ids):
        if s == e == 0:
            flush()
            continue
        tag = id2tag.get(int(pid), "O")
        if tag == "O":
            flush()
        elif tag.startswith("B-"):
            flush()
            cur_lab, cur_s, cur_e = tag[2:], int(s), int(e)
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur_lab == lab:
                cur_e = int(e)
            else:
                flush()
                cur_lab, cur_s, cur_e = lab, int(s), int(e)
    flush()
    out: list[IndicBertMention] = []
    for s, e, lab in spans:
        if 0 <= s < e <= len(text):
            out.append(IndicBertMention(text=text[s:e], label=lab, start=s, end=e))
    out.sort(key=lambda m: (m.start, m.end))
    return out


def predict_indicbert_ner(text: str, model_dir: str | None = None,
                           max_length: int = 256) -> IndicBertNerResult:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    import torch

    bundle = load_indicbert_ner(model_dir)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    id2tag: dict[int, str] = bundle["id2tag"]
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu()
    seqlen = int(enc["attention_mask"][0].sum().item())
    pred_ids = logits[:seqlen].argmax(-1).tolist()
    mentions = _decode(text, offsets[:seqlen], pred_ids, id2tag)
    return IndicBertNerResult(original_text=text, entities=mentions)


def predict_merged(text: str, model_dir: str | None = None):
    """Baseline spans + IndicBERT spans (union; IndicBERT wins exact dupes)."""
    from app.services import ner as baseline

    base = baseline.extract_entities(text).entities
    try:
        extra = predict_indicbert_ner(text, model_dir).entities
    except (ModelUnavailableError, ValueError, TypeError):
        return base
    except Exception:
        return base
    seen = {(m.start, m.end, m.label) for m in base}
    merged = list(base)
    for m in extra:
        if (m.start, m.end, m.label) in seen:
            # replace baseline method with indicbert on exact dupes
            for i, b in enumerate(merged):
                if (b.start, b.end, b.label) == (m.start, m.end, m.label):
                    merged[i] = baseline.EntityMention(
                        text=b.text, label=b.label, start=b.start,
                        end=b.end, method="indicbert_ner",
                    )
                    break
        else:
            overlap = any(not (m.end <= b.start or m.start >= b.end) for b in merged)
            if not overlap:
                merged.append(baseline.EntityMention(
                    text=m.text, label=m.label, start=m.start,
                    end=m.end, method="indicbert_ner",
                ))
    merged.sort(key=lambda m: (m.start, m.end))
    return merged


def indicbert_ner_status(model_dir: str | None = None) -> dict:
    root = model_path(model_dir)
    if not os.path.isdir(root):
        return {"available": False,
                "reason": f"No IndicBERT NER model at '{root}'. Baseline regex+gazetteer still serves entities.",
                "path": root}
    labels: dict = {}
    try:
        with open(os.path.join(root, LABELS_FILE), encoding="utf-8") as fh:
            labels = json.load(fh)
    except Exception:
        pass
    return {"available": True, "path": root, "labels": labels,
            "backend": "indicbert_token_classification_bio", "method": "indicbert_ner"}
