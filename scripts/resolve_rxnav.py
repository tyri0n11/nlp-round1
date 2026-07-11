#!/usr/bin/env python3
"""Build data/resources/rxnorm.json by resolving drug spans via the RxNav API.

BUILD-TIME ONLY — the competition forbids external API calls at *inference*
time, so we query RxNav once here, cache {normalized_span: [rxcui]} to JSON, and
the offline linker (npr.linking.rxnorm) reads that cache at inference.

Strategy per drug span (matches the gold's SCD granularity):
  1. clean: drop route/frequency tokens, drop volumes; if the span has no dose
     form word, append "oral tablet" (the dominant form for `po` orders).
  2. RxNav approximateTerm -> top candidates.
  3. prefer a TTY=SCD (generic clinical drug, e.g. 308135) over branded (SBD).

Usage:
  python scripts/resolve_rxnav.py --from-output output      # spans from preds
  python scripts/resolve_rxnav.py --spans spans.txt         # one span per line
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npr.utils.io import read_gold  # noqa: E402
from npr.pipeline._06_linking import normalize_span  # noqa: E402
from npr.utils.schema import TYPE_DRUG  # noqa: E402

RXNAV = "https://rxnav.nlm.nih.gov/REST"

_FREQ = re.compile(
    r"\b(daily|bid|tid|qid|qhs|qam|qpm|prn|q\d+h(:prn)?|qod|once|weekly|:prn)\b", re.I
)
_FORM = re.compile(
    r"\b(tablet|capsule|suspension|solution|cream|ointment|gel|drops?|inhal\w*|"
    r"patch|spray|syrup|lotion|suppository|injection|powder)\b", re.I
)
_VOL = re.compile(r"\b\d+(\.\d+)?\s*ml\b", re.I)


def _base_clean(span: str):
    """Strip route/freq/volume. Returns (term, has_dose_form_word)."""
    s = span.lower()
    s = re.sub(r"\bpo\b", " ", s)
    s = re.sub(r"\biv\b", " injection ", s)
    s = _FREQ.sub(" ", s)
    has_form = bool(_FORM.search(s))
    s = _VOL.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(), has_form


def clean_for_query(span: str) -> str:
    """Term for approximateTerm: append 'oral tablet' when no form word (the
    dominant form for `po` orders). /drugs uses the base term instead."""
    s, has_form = _base_clean(span)
    if not has_form:
        s += " oral tablet"
    return s.strip()


def _get(url: str):
    return json.load(urllib.request.urlopen(url, timeout=20))


def _approx(term: str, n: int = 8) -> list:
    enc = urllib.parse.quote(term)
    d = _get(f"{RXNAV}/approximateTerm.json?term={enc}&maxEntries={n}&option=1")
    return d.get("approximateGroup", {}).get("candidate", [])


_STOP = {"mg", "mcg", "ml", "oral", "tablet", "capsule", "po", "iv", "g"}


def _drugs_scd(term: str) -> list:
    """/drugs SCD concepts as (rxcui, name). Precise but lower recall."""
    enc = urllib.parse.quote(term)
    d = _get(f"{RXNAV}/drugs.json?name={enc}")
    out = []
    for g in d.get("drugGroup", {}).get("conceptGroup", []):
        if g.get("tty") == "SCD":
            for c in g.get("conceptProperties", []):
                out.append((c["rxcui"], c["name"]))
    return out


def _pick_scd(term: str, scd: list):
    """Prefer single-ingredient SCD (no '/') best covering the query tokens."""
    if not scd:
        return None
    single = [(r, n) for r, n in scd if "/" not in n] or scd
    toks = [t for t in re.findall(r"[a-z0-9.]+", term.lower()) if t not in _STOP]

    def score(name: str):
        nl = name.lower()
        return (sum(t in nl for t in toks), -len(name))  # coverage, then shortest

    return max(single, key=lambda x: score(x[1]))[0]


def _tty(rxcui: str) -> str:
    try:
        d = _get(f"{RXNAV}/rxcui/{rxcui}/property.json?propName=TTY")
        props = d.get("propConceptGroup", {}).get("propConcept", [])
        return props[0]["propValue"] if props else ""
    except Exception:
        return ""


_NORM_SYS = "You are a pharmacology normalizer. Output ONLY JSON, no prose."
_NORM_USER = (
    'Normalize this clinical drug mention (may be Vietnamese, a brand name, '
    'misspelled, with dose/route/frequency) to JSON:\n'
    '{{"is_drug":true/false,"ingredient_en":"generic English ingredient, '
    'RxNorm style","strength":"number+unit or null","form":"oral tablet/'
    'capsule/suspension/injection or null"}}\n'
    'If it is NOT a real medication (symptom, time phrase, instruction), set '
    'is_drug=false.\nMention: "{span}"\nJSON:'
)


def llm_normalize(span: str, backend) -> dict:
    """Use the LLM to canonicalise brand/Vietnamese/messy names and flag
    non-drugs. Returns {} on failure (caller falls back to regex cleaning)."""
    try:
        out = backend.generate(_NORM_SYS, _NORM_USER.format(span=span))
    except Exception:
        return {}
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _norm_query(d: dict) -> str | None:
    if not d.get("is_drug"):
        return None
    ing = (d.get("ingredient_en") or "").strip()
    if not ing or ing.lower() == "null":
        return None
    st = str(d.get("strength") or "").strip()
    if st.lower() == "null":
        st = ""
    return f"{ing} {st}".strip()


def resolve(span: str, sleep: float = 0.1, backend=None) -> list:
    # 0) optional LLM normalization: brand->generic, VN->EN, filter non-drugs
    if backend is not None:
        d = llm_normalize(span, backend)
        if d:  # got a parse
            q = _norm_query(d)
            if q is None:
                return []  # LLM says not a real drug -> no candidate
            try:
                scd = _drugs_scd(q)
                picked = _pick_scd(q, scd)
                if picked:
                    return [picked]
            except Exception:
                pass
            # keep the normalized ingredient for the regex/approx fallback below
            span = f"{q} {d.get('form') or 'oral tablet'}"

    base, _ = _base_clean(span)
    # 1) precise path: /drugs (base term, no appended form) -> single-ing SCD
    if base:
        try:
            scd = _drugs_scd(base)
            picked = _pick_scd(base, scd)
            if picked:
                return [picked]
        except Exception:
            pass
    # 2) higher-recall fallback: approximateTerm (with 'oral tablet') + TTY=SCD
    term = clean_for_query(span)
    if not term:
        return []
    try:
        cands = _approx(term)
    except Exception:
        return []
    if not cands:
        return []
    # dedup rxcuis, preserve approximate rank
    seen, ranked = set(), []
    for c in cands:
        r = c.get("rxcui")
        if r and r not in seen:
            seen.add(r)
            ranked.append(r)
    # prefer the first SCD (generic clinical drug — matches gold granularity)
    for r in ranked[:6]:
        time.sleep(sleep)
        if _tty(r) == "SCD":
            return [r]
    return [ranked[0]]  # fall back to best lexical match


def collect_spans_from_output(out_dir: str) -> list:
    preds = read_gold(out_dir)
    spans = []
    seen = set()
    for concepts in preds.values():
        for c in concepts:
            if c.type == TYPE_DRUG:
                key = normalize_span(c.text)
                if key and key not in seen:
                    seen.add(key)
                    spans.append(c.text)
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-output")
    ap.add_argument("--spans")
    ap.add_argument("--out", default="data/resources/rxnorm.json")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--llm", action="store_true",
                    help="LLM-normalize names (brand->generic, VN->EN, filter non-drugs)")
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()

    backend = None
    if args.llm:
        from npr.pipeline._01_ner_llm import OllamaBackend
        backend = OllamaBackend(model=args.model, think=False)

    if args.from_output:
        spans = collect_spans_from_output(args.from_output)
    elif args.spans:
        spans = [l.strip() for l in Path(args.spans).read_text().splitlines() if l.strip()]
    else:
        ap.error("provide --from-output or --spans")

    # merge into any existing cache so reruns are incremental
    out = Path(args.out)
    table = json.loads(out.read_text()) if out.exists() else {}
    print(f"resolving {len(spans)} unique drug spans via RxNav ...")
    for i, span in enumerate(spans, 1):
        key = normalize_span(span)
        if key in table:
            continue
        codes = resolve(span, args.sleep, backend=backend)
        table[key] = codes
        print(f"[{i}/{len(spans)}] {span!r} -> {codes}", flush=True)
        time.sleep(args.sleep)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    hit = sum(1 for v in table.values() if v)
    print(f"\nwrote {len(table)} spans ({hit} resolved) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
