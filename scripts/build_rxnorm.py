#!/usr/bin/env python3
"""Build data/resources/rxnorm.json (ingredient/name -> [RXCUI]) lookup.

Two sources supported:

  A) RxNorm RRF release (offline, recommended for the private-test rebuild):
       python scripts/build_rxnorm.py --rrf /path/to/RxNorm/rrf/RXNCONSO.RRF

     Uses only English (LAT=ENG) atoms; keys are lowercased STR values, values
     are the RXCUI list. This yields the exact codes the metric expects.

  B) A pre-extracted TSV/CSV "name<TAB>rxcui" you already have:
       python scripts/build_rxnorm.py --tsv names.tsv

The REST API path is intentionally omitted: the competition forbids external
API calls at inference time. Build this table offline, ship rxnorm.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def from_rrf(path: str) -> dict:
    table: dict[str, set] = defaultdict(set)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # RXNCONSO.RRF columns are pipe-delimited: RXCUI at [0], LAT [1],
            # STR at [14].
            cols = line.rstrip("\n").split("|")
            if len(cols) < 15:
                continue
            rxcui, lat, name = cols[0], cols[1], cols[14]
            if lat != "ENG" or not name:
                continue
            table[name.lower().strip()].add(rxcui)
    return {k: sorted(v) for k, v in table.items()}


def from_tsv(path: str) -> dict:
    table: dict[str, set] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            name, rxcui = row[0], row[1]
            table[name.lower().strip()].add(rxcui)
    return {k: sorted(v) for k, v in table.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf")
    ap.add_argument("--tsv")
    ap.add_argument("--out", default="data/resources/rxnorm.json")
    args = ap.parse_args()

    if args.rrf:
        table = from_rrf(args.rrf)
    elif args.tsv:
        table = from_tsv(args.tsv)
    else:
        ap.error("provide --rrf or --tsv")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(table)} names -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
