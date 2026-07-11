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
from npr.pipeline._06_linking import ICD10Linker, RxNormLinker  # noqa: E402
from npr.utils.schema import LINKED_TYPES, TYPE_DIAGNOSIS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output")
    ap.add_argument("--zip", default="output.zip")
    ap.add_argument("--rxnorm", default="data/resources/rxnorm.json")
    ap.add_argument("--icd10", default="data/resources/icd10.json")
    args = ap.parse_args()

    rx = RxNormLinker.from_json(args.rxnorm)
    icd = ICD10Linker.from_json(args.icd10)
    preds = read_gold(args.pred)
    drugs = drugs_hit = dx = dx_hit = 0
    for concepts in preds.values():
        rx.apply(concepts)
        icd.apply(concepts)
        for c in concepts:
            if c.type in LINKED_TYPES:
                drugs += 1
                drugs_hit += bool(c.candidates)
            elif c.type == TYPE_DIAGNOSIS:
                dx += 1
                dx_hit += bool(c.candidates)

    write_outputs(preds, args.pred, args.zip)
    print(f"THUỐC={drugs_hit}/{drugs}, CHẨN_ĐOÁN(ICD-10)={dx_hit}/{dx} "
          f"-> {args.pred}/ + {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
