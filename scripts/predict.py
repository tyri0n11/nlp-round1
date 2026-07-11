#!/usr/bin/env python3
"""Run inference over an input dir and produce output/ + output.zip.

Usage:
    python scripts/predict.py --input data/input --out output --zip output.zip
    python scripts/predict.py --no-llm            # heuristic baseline only
    python scripts/predict.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npr.io_utils import read_inputs, write_outputs  # noqa: E402
from npr.pipeline import Pipeline, PipelineConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--out", default="output")
    ap.add_argument("--zip", default="output.zip")
    ap.add_argument("--no-llm", action="store_true", help="heuristic baseline only")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "transformers"])
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--rxnorm", default="data/resources/rxnorm.json")
    args = ap.parse_args()

    cfg = PipelineConfig(
        use_llm=not args.no_llm,
        llm_backend=args.backend,
        llm_model=args.model,
        rxnorm_path=args.rxnorm,
    )
    pipe = Pipeline(cfg)

    records = read_inputs(args.input)
    predictions = {}
    t0 = time.time()
    for i, (rid, text) in enumerate(records, 1):
        concepts = pipe.run(text)
        predictions[rid] = concepts
        print(f"[{i}/{len(records)}] {rid}.txt -> {len(concepts)} concepts", flush=True)

    write_outputs(predictions, args.out, args.zip)
    dt = time.time() - t0
    total = sum(len(v) for v in predictions.values())
    print(f"\nWrote {len(predictions)} files, {total} concepts to {args.out}/ "
          f"and {args.zip} in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
