"""RxNorm candidate linking for drug concepts.

The example candidates ("308135", "243670", ...) are RxNorm RXCUIs. RxNorm is
not shipped with this repo; build a lookup table once from an RxNorm release
(RRF) or the public REST API and cache it as JSON, then this linker does
offline ingredient-based matching.

Table format (data/resources/rxnorm.json):
    {"amlodipine": ["308135", ...], "aspirin": ["243670"], ...}
keyed by lowercased ingredient / brand string.

Linking strategy (offline, deterministic):
  1. lowercase the drug span, strip dose/route/frequency tokens -> ingredient
  2. exact table hit -> its codes
  3. else longest-ingredient substring hit
  4. else [] (no candidate)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..utils.schema import LINKED_TYPES, TYPE_DIAGNOSIS, Concept

# tokens to strip when reducing a drug span to its ingredient
_NOISE = re.compile(
    r"\b(?:po|iv|im|sc|sl|pr|top|inh|oral|suspension|tablet|cap(?:sule)?s?|"
    r"solution|susp|xl|er|sr|cr|mg|mcg|g|ml|units?|daily|bid|tid|qid|qhs|qam|"
    r"qpm|prn|qod|q\d+h|once|weekly)\b",
    re.I,
)
_NUM = re.compile(r"[\d\.\-]+")


def ingredient(span: str) -> str:
    s = span.lower()
    s = _NOISE.sub(" ", s)
    s = _NUM.sub(" ", s)
    s = re.sub(r":\w+", " ", s)  # drop ":prn" leftovers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_span(span: str) -> str:
    """Cache key for a full drug span (dose + form matter for RxNorm SCD).

    Only route/frequency abbreviations and whitespace are normalised so the
    same order written slightly differently maps to one cache entry.
    """
    s = span.lower()
    s = re.sub(r":\s*prn\b", " prn", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class RxNormLinker:
    def __init__(self, table: Dict[str, List[str]] | None = None):
        self.table = table or {}
        # precompute ingredient keys sorted longest-first for substring fallback
        self._keys = sorted(self.table.keys(), key=len, reverse=True)

    @classmethod
    def from_json(cls, path: str | Path) -> "RxNormLinker":
        p = Path(path)
        if not p.exists():
            return cls({})
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def link(self, span: str) -> List[str]:
        # 1) exact full-span cache (built offline from RxNav; dose+form aware)
        norm = normalize_span(span)
        if norm in self.table:
            return list(self.table[norm])
        # 2) ingredient-level fallback (if the table is ingredient-keyed)
        ing = ingredient(span)
        if not ing:
            return []
        if ing in self.table:
            return list(self.table[ing])
        # first word is usually the ingredient
        head = ing.split(" ")[0]
        if head in self.table:
            return list(self.table[head])
        for k in self._keys:
            if k and (k in ing):
                return list(self.table[k])
        return []

    def apply(self, concepts: List[Concept]) -> List[Concept]:
        for c in concepts:
            if c.type in LINKED_TYPES:
                c.candidates = self.link(c.text)
        return concepts


class ICD10Linker:
    """Offline diagnosis -> ICD-10 code lookup (table built by 06_resolve_icd10)."""

    def __init__(self, table: Dict[str, List[str]] | None = None,
                 target_type: str = TYPE_DIAGNOSIS):
        self.table = table or {}
        self.target_type = target_type

    @classmethod
    def from_json(cls, path: str | Path, target_type: str = TYPE_DIAGNOSIS) -> "ICD10Linker":
        p = Path(path)
        if not p.exists():
            return cls({}, target_type)
        return cls(json.loads(p.read_text(encoding="utf-8")), target_type)

    def link(self, span: str) -> List[str]:
        return list(self.table.get(normalize_span(span), []))

    def apply(self, concepts: List[Concept]) -> List[Concept]:
        for c in concepts:
            if c.type == self.target_type and not c.candidates:
                c.candidates = self.link(c.text)
        return concepts
