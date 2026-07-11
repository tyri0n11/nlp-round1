#!/usr/bin/env python3
"""Resolve CHẨN_ĐOÁN spans to ICD-10 codes with the self-hosted LLM.

Mirrors the drug resolver but for diagnoses: Qwen3-8B maps a Vietnamese
diagnosis to a WHO ICD-10 code (language-independent). Codes are regex-validated
and cached to data/resources/icd10.json ({normalized_span: [code]}).

Inference stays self-hosted (<=9B) and API-free — compliant.

    python scripts/06_resolve_icd10.py --from-output output
    python scripts/06_resolve_icd10.py --from-output output --no-dot   # K703 form
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npr.utils.io import read_gold  # noqa: E402
from npr.pipeline._06_linking import normalize_span  # noqa: E402
from npr.utils.schema import TYPE_DIAGNOSIS  # noqa: E402

_ICD = re.compile(r"^[A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?$")  # WHO ICD-10 shape
_SYS = "Bạn là chuyên gia mã hoá ICD-10 (WHO). Chỉ trả JSON, không giải thích."
_USER = (
    'Cho một chẩn đoán/bệnh lý tiếng Việt, trả về mã ICD-10 chuẩn WHO chính xác '
    'nhất. Nếu không chắc, trả mã 3 ký tự của nhóm. JSON: '
    '{{"icd10":"<mã, vd K70.3>"}}\nChẩn đoán: "{dx}"'
)


def llm_icd10(span: str, backend) -> str | None:
    try:
        out = backend.generate(_SYS, _USER.format(dx=span))
    except Exception:
        return None
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        code = str(json.loads(m.group(0)).get("icd10", "")).strip().upper()
    except json.JSONDecodeError:
        return None
    return code if _ICD.match(code) else None


def collect_dx(out_dir: str) -> list:
    preds = read_gold(out_dir)
    seen, spans = set(), []
    for concepts in preds.values():
        for c in concepts:
            if c.type == TYPE_DIAGNOSIS:
                k = normalize_span(c.text)
                if k and k not in seen:
                    seen.add(k)
                    spans.append(c.text)
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-output", required=True)
    ap.add_argument("--out", default="data/resources/icd10.json")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--no-dot", action="store_true", help="store codes without '.'")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    from npr.pipeline._01_ner_llm import OllamaBackend
    backend = OllamaBackend(model=args.model, think=False)

    spans = collect_dx(args.from_output)
    out = Path(args.out)
    table = json.loads(out.read_text()) if out.exists() else {}
    print(f"resolving {len(spans)} unique diagnosis spans via {args.model} ...")
    for i, span in enumerate(spans, 1):
        key = normalize_span(span)
        if key in table:
            continue
        code = llm_icd10(span, backend)
        if code and args.no_dot:
            code = code.replace(".", "")
        table[key] = [code] if code else []
        print(f"[{i}/{len(spans)}] {span[:40]!r} -> {table[key]}", flush=True)
        time.sleep(args.sleep)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    hit = sum(1 for v in table.values() if v)
    print(f"\nwrote {len(table)} diagnoses ({hit} coded) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
