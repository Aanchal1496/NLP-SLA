"""Dataset-analytics CLI: chart-ready JSON from the REAL dataset file.

Usage:
    python -m training.analytics --data data/raw/intents.csv \\
        --text-col text --label-col intent [--entity-col entities] [--out FILE]

Exit codes: 0 success (JSON on stdout / written to --out), 2 dataset error.
Statistics are computed from the actual file; nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.services.analytics import analyze_dataset_file
from training.dataset import DatasetError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marathi dataset analytics.")
    parser.add_argument("--data", required=True, help="Path to annotated CSV.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="intent")
    parser.add_argument("--entity-col", default=None)
    parser.add_argument("--out", default=None, help="Write JSON to FILE.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = analyze_dataset_file(
            args.data, args.text_col, args.label_col, args.entity_col
        )
    except DatasetError as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"analytics written to '{args.out}'")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
