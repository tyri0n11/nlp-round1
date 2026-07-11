#!/usr/bin/env python3
"""Fill drug `candidates` in existing predictions using the offline RxNorm cache.

Re-runs ONLY the linking stage over output/*.json (no LLM), then rebuilds
output.zip. Use after scripts/resolve_rxnav.py has produced rxnorm.json.

    python scripts/apply_candidates.py --pred output --zip output.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npr.utils.io import read_gold, write_outputs  # noqa: E402
from npr.pipeline.linking import RxNormLinker  # noqa: E402
from npr.utils.schema import LINKED_TYPES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output")
    ap.add_argument("--zip", default="output.zip")
    ap.add_argument("--rxnorm", default="data/resources/rxnorm.json")
    args = ap.parse_args()

    linker = RxNormLinker.from_json(args.rxnorm)
    preds = read_gold(args.pred)
    filled = 0
    drugs = 0
    for concepts in preds.values():
        linker.apply(concepts)
        for c in concepts:
            if c.type in LINKED_TYPES:
                drugs += 1
                if c.candidates:
                    filled += 1

    write_outputs(preds, args.pred, args.zip)
    print(f"drugs={drugs}, with candidates={filled} "
          f"({100*filled/drugs:.0f}%) -> {args.pred}/ + {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
