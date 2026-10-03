"""Generate SYNTHETIC SILVER NER annotations (not human gold).

Why this exists: data/raw/intents.csv has only 2 baseline-regex hits across
64 rows, so supervised NER cannot be trained on it. This script generates
deterministic Marathi legal/financial sentences from templates whose entity
spans are KNOWN BY CONSTRUCTION (offsets computed programmatically, then
verified as text[start:end] == span text).

Output: data/silver/ner_silver.csv with columns: text, entities
  entities = JSON list of {"text","label","start","end"} (char offsets).

Labels are the documented BASELINE_LABELS (DATE, MONEY, CASE_NUMBER,
SECTION, ORGANIZATION). PERSON is excluded by default: no curated Marathi
person list exists; pass --with-person to add a tiny illustrative list.

This is SILVER data for bootstrapping only. Do not present models trained
on it as human-annotated; metrics are reported as silver-held-out scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

ORG_PHRASES = [
    "मुंबई उच्च न्यायालय",
    "सर्वोच्च न्यायालय",
    "महाराष्ट्र शासन",
    "भारतीय रिझर्व्ह बँक",
    "रिझर्व्ह बँक",
    "भारतीय दंड संहिता",
    "मोटार वाहन कायदा",
]

DATES = ["१२/०५/२०२४", "०१/०१/२०२३", "15/08/2024", "12-05-2024", "२५/१२/२०२३"]
MONEYS = ["₹५,००,०००", "₹ ५०००", "₹1,00,000", "५००० रुपये", "१,५०,००० रुपये"]
SECTIONS = ["कलम ३०२", "कलम ४२०", "धारा 302", "Section 138", "कलम ३७६(अ)"]
CASE_NOS = ["सी.आर. नं. १२३/२०२४", "CR No. 45/2023", "नं. ७८/२०२४", "१२३/२०२४"]
PERSONS = ["राहुल शर्मा", "सुनीता पाटील", "अमित देशमुख"]

TEMPLATES = [
    ("दिनांक {DATE} रोजी सुनावणी होईल।", ["DATE"]),
    ("रक्कम {MONEY} देय आहे।", ["MONEY"]),
    ("{SECTION} अंतर्गत गुन्हा दाखल झाला।", ["SECTION"]),
    ("{CASE} अन्वये तपास सुरू आहे।", ["CASE_NUMBER"]),
    ("{ORG} ने निर्णय दिला।", ["ORGANIZATION"]),
    ("दिनांक {DATE} रोजी {SECTION} अंतर्गत सुनावणी होईल।", ["DATE", "SECTION"]),
    ("{CASE} मधील आरोपीस {MONEY} दंड ठोठावला।", ["CASE_NUMBER", "MONEY"]),
    ("{ORG} ने दिनांक {DATE} रोजी {SECTION} खाली आदेश दिला।", ["ORGANIZATION", "DATE", "SECTION"]),
    ("{ORG} च्या परिपत्रकानुसार रक्कम {MONEY} भरणे आवश्यक आहे।", ["ORGANIZATION", "MONEY"]),
    ("फिर्यादी {PERSON} हजर झाला।", ["PERSON"]),
]


def build_sentence(rng: random.Random, with_person: bool) -> tuple[str, list[dict]]:
    # Filter out PERSON template unless enabled.
    cands = [t for t in TEMPLATES if with_person or "PERSON" not in t[1]]
    template, slots = rng.choice(cands)
    picks: dict[str, str] = {}
    for slot in slots:
        if slot == "DATE":
            picks[slot] = rng.choice(DATES)
        elif slot == "MONEY":
            picks[slot] = rng.choice(MONEYS)
        elif slot == "SECTION":
            picks[slot] = rng.choice(SECTIONS)
        elif slot == "CASE_NUMBER":
            picks[slot] = rng.choice(CASE_NOS)
        elif slot == "ORGANIZATION":
            picks[slot] = rng.choice(ORG_PHRASES)
        elif slot == "PERSON":
            picks[slot] = rng.choice(PERSONS)
    text = template
    text = text.replace("{DATE}", picks.get("DATE", ""))
    text = text.replace("{MONEY}", picks.get("MONEY", ""))
    text = text.replace("{SECTION}", picks.get("SECTION", ""))
    text = text.replace("{CASE}", picks.get("CASE_NUMBER", ""))
    text = text.replace("{ORG}", picks.get("ORGANIZATION", ""))
    text = text.replace("{PERSON}", picks.get("PERSON", ""))
    spans: list[dict] = []
    for slot in slots:
        val = picks[slot]
        start = text.find(val)
        assert start >= 0, f"value not found: {val!r} in {template!r}"
        end = start + len(val)
        assert text[start:end] == val
        spans.append({"text": val, "label": slot, "start": start, "end": end})
    spans.sort(key=lambda s: s["start"])
    return text, spans


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate silver NER CSV.")
    ap.add_argument("--out", default="data/silver/ner_silver.csv")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-person", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows: list[tuple[str, list[dict]]] = []
    for _ in range(args.n):
        rows.append(build_sentence(rng, args.with_person))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "entities"])
        for text, spans in rows:
            w.writerow([text, json.dumps(spans, ensure_ascii=False)])
    # ascii-safe summary (no Devanagari to stdout for cp1252 consoles)
    from collections import Counter
    c: Counter[str] = Counter()
    for _, spans in rows:
        for s in spans:
            c[s["label"]] += 1
    print(f"silver rows written: {len(rows)} -> {args.out}")
    print(f"label_counts: {dict(sorted(c.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
