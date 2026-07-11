#!/usr/bin/env python3
"""Score predictions against a gold dir with the local (unofficial) metric.

Usage:
    python scripts/evaluate.py --pred output --gold data/gold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npr.evaluate import evaluate  # noqa: E402
from npr.io_utils import read_gold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    gold = read_gold(args.gold)
    pred = read_gold(args.pred)  # predictions share the JSON layout
    scores = evaluate(gold, pred)
    print(json.dumps(scores.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
