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
from npr.io_utils import read_gold  # noqa: E402
from npr.linking.rxnorm import normalize_span  # noqa: E402
from npr.schema import TYPE_DRUG  # noqa: E402

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


def resolve(span: str, sleep: float = 0.1) -> list:
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
    args = ap.parse_args()

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
        codes = resolve(span, args.sleep)
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
