#!/usr/bin/env python3
"""Clean an existing output/ in place (no NER re-run):

  1. strip leading filler verbs ("cho bumetanide" -> "bumetanide") + re-align
  2. drop THUỐC concepts the LLM flags as non-drugs (is_drug=false)
  3. re-link RxNorm candidates, dedup, rewrite output/ + output.zip

Writes data/resources/non_drugs.json (spans judged not-a-drug) so the pipeline
can reuse the filter on future runs.

    python scripts/clean_output.py --pred output [--no-llm-filter]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))          # for `scripts.resolve_rxnav`
sys.path.insert(0, str(_ROOT / "src"))  # for `npr`

from npr.pipeline import _04_align as align  # noqa: E402
from npr.utils.io import read_gold, read_inputs, write_outputs  # noqa: E402
from npr.pipeline._06_linking import RxNormLinker, normalize_span  # noqa: E402
from npr.pipeline import _dedup  # noqa: E402
from npr.pipeline._03_postprocess import (  # noqa: E402
    DrugValidator,
    llm_normalize,
    strip_leading_noise,
)
from npr.utils.schema import TYPE_DRUG, validate_concept  # noqa: E402


def build_is_drug_fn(backend, cache_path: Path):
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def is_drug(span: str):
        key = normalize_span(span)
        if key in cache:
            return cache[key]
        d = llm_normalize(span, backend)
        val = bool(d.get("is_drug")) if d else None  # None = unknown -> keep
        cache[key] = val
        return val

    def flush():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    return is_drug, flush


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output")
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--zip", default="output.zip")
    ap.add_argument("--rxnorm", default="data/resources/rxnorm.json")
    ap.add_argument("--non-drugs", default="data/resources/non_drugs.json")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--no-llm-filter", action="store_true",
                    help="only strip prefixes; skip the is_drug drug filter")
    args = ap.parse_args()

    raw = {rid: text for rid, text in read_inputs(args.input)}
    preds = read_gold(args.pred)
    linker = RxNormLinker.from_json(args.rxnorm)

    validator = None
    flush = lambda: None
    if not args.no_llm_filter:
        from npr.pipeline._01_ner_llm import OllamaBackend
        backend = OllamaBackend(model=args.model, think=False)
        is_drug, flush = build_is_drug_fn(backend, Path(args.non_drugs))
        validator = DrugValidator(is_drug)

    n_before = n_after = n_dropped_nondrug = 0
    for rid, concepts in preds.items():
        n_before += len(concepts)
        for c in concepts:  # 1) trim + reset for re-alignment
            c.text = strip_leading_noise(c.text)
            c.position = [0, 0]
        concepts = align.assign_positions(raw[rid], concepts)  # 2) re-align
        if validator is not None:  # 3) drop non-drugs
            kept = validator.filter(concepts)
            n_dropped_nondrug += len(concepts) - len(kept)
            concepts = kept
        linker.apply(concepts)  # 4) re-link
        concepts = _dedup([c for c in concepts if not validate_concept(c, raw[rid])])
        preds[rid] = concepts
        n_after += len(concepts)

    flush()
    write_outputs(preds, args.pred, args.zip)
    print(f"concepts {n_before} -> {n_after} "
          f"(dropped non-drug THUỐC: {n_dropped_nondrug}) | {args.pred}/ + {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
