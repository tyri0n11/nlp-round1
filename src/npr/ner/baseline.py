"""Regex/heuristic NER baseline.

Not competitive on its own — it exists so the full pipeline (positions,
linking, assertions, output.zip) runs end-to-end with zero model deps, and as
a fallback when the LLM misses obviously-structured drug lines. It targets the
most regular pattern in the data: numbered/bulleted medication lists with a
dose + route + frequency, e.g. "metoprolol 25mg po bid".
"""
from __future__ import annotations

import re
from typing import List

from ..schema import TYPE_DRUG, Concept

# dose/route/frequency signals typical of medication orders
_UNIT = r"(?:mg|mcg|g|ml|units?)"
_ROUTE = r"(?:po|iv|im|sc|sl|pr|top|inh)"
_FREQ = r"(?:daily|bid|tid|qid|qhs|qam|qpm|prn|q\d+h(?::prn)?|qod|once|weekly)"
# <name>[ modifiers] <dose><unit> [route] [frequency]  — matched anywhere.
_DRUG = re.compile(
    r"(?<![a-zàáâãèéêìíòóôõùúýăđĩũơưạ-ỹ])"          # not mid-word (incl. VN letters)
    r"([a-z][a-z0-9\-]+(?:\s+[a-z0-9\-\./]+){0,5}?"  # drug name + modifiers
    r"\s+\d[\d\.\-]*\s*" + _UNIT + r"\b"           # dose + unit
    r"(?:\s+" + _ROUTE + r")?"                     # optional route
    r"(?:\s+" + _FREQ + r"(?::prn)?)?)",           # optional frequency
    re.I,
)


def extract_drugs(text: str) -> List[Concept]:
    out: List[Concept] = []
    for m in _DRUG.finditer(text):
        span = m.group(1).strip()
        s = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
        e = s + len(span)
        out.append(Concept(text=span, type=TYPE_DRUG, position=[s, e]))
    return out


def extract(text: str) -> List[Concept]:
    """Heuristic entry point. Currently drugs only."""
    concepts = extract_drugs(text)
    concepts.sort(key=lambda c: c.position[0])
    return concepts
