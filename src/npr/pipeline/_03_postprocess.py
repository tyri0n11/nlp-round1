"""Post-processing to clean up LLM concept spans.

Two fixes:
  1. strip_leading_noise — drop administration/instruction verbs the model
     often glues to the front of a span ("cho bumetanide 2mg" -> "bumetanide
     2mg", "được chỉ định aspirin" -> "aspirin"). Improves text (WER) and kills
     the duplicate that appears once with and once without the prefix.
  2. DrugValidator — reuse the LLM drug-name normalizer's is_drug flag to drop
     THUỐC concepts that are actually instructions/time phrases the NER
     mislabeled ("được kê đơn", "bắt đầu 3 tuần trước"). Improves precision.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from ..utils.schema import TYPE_DRUG, Concept

# --- LLM drug-name normalizer (shared by resolver + is_drug filter) ---------
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
    """Canonicalise a brand/Vietnamese/messy drug name and flag non-drugs.

    `backend` is any object with .generate(system, user) -> str (e.g.
    OllamaBackend). Returns {} on failure so callers can fall back.
    """
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

# leading filler verbs/phrases (longest first so multi-word ones match first)
_LEADING = [
    "được kê đơn", "được chỉ định", "được cho", "cho dùng", "chỉ định",
    "điều trị bằng", "điều trị với", "dự phòng bằng", "sử dụng", "được kê",
    "tiếp tục", "khởi động", "bắt đầu", "chuyển sang",
    "cho", "dùng", "uống", "tiêm", "truyền", "kê", "thêm", "và",
]
_LEADING_RE = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in _LEADING) + r")\s+", re.I
)


def strip_leading_noise(text: str) -> str:
    """Remove leading filler verbs; iterate for stacked ones ("cho dùng X")."""
    prev = None
    s = text
    while s != prev:
        prev = s
        s = _LEADING_RE.sub("", s, count=1).lstrip()
    return s or text  # never return empty


def clean_spans(concepts: List[Concept]) -> List[Concept]:
    for c in concepts:
        c.text = strip_leading_noise(c.text)
    return concepts


class DrugValidator:
    """Drop THUỐC concepts the LLM says are not real medications.

    `is_drug_fn(span) -> Optional[bool]`: True/False, or None if unknown (then
    the concept is kept — never drop on uncertainty).
    """

    def __init__(self, is_drug_fn: Callable[[str], Optional[bool]]):
        self.is_drug = is_drug_fn

    def filter(self, concepts: List[Concept]) -> List[Concept]:
        out = []
        for c in concepts:
            if c.type == TYPE_DRUG and self.is_drug(c.text) is False:
                continue
            out.append(c)
        return out
