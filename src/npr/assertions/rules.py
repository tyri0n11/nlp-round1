"""Rule-based assertion backfill / correction.

The LLM proposes assertions; these deterministic rules catch the highest-signal
Vietnamese cues and enforce section-based defaults (medications listed under a
"thuốc trước khi nhập viện" / "tiền sử" heading are historical). Rules only add
an assertion when the model left it empty, except negation which overrides.
"""
from __future__ import annotations

import re
from typing import List

from ..schema import (
    ASSERT_ABSENT,
    ASSERT_HISTORICAL,
    ASSERT_POSSIBLE,
    LINKED_TYPES,
    Concept,
)

# window of text just before a concept used to read local cues
_WINDOW = 40

_NEG = re.compile(r"\b(không|chưa|phủ định|loại trừ|âm tính)\b", re.I)
_POSS = re.compile(r"(nghi ngờ|có thể|khả năng|theo dõi)", re.I)
_HIST = re.compile(
    r"(tiền sử|trước khi nhập viện|trước nhập viện|đã (?:từng|dùng|sử dụng)|"
    r"bệnh sử|đã được chẩn đoán)",
    re.I,
)


def _context(raw: str, c: Concept) -> str:
    s = max(0, c.position[0] - _WINDOW)
    return raw[s:c.position[0]]


def apply(raw: str, concepts: List[Concept]) -> List[Concept]:
    for c in concepts:
        ctx = _context(raw, c)
        if _NEG.search(ctx):
            c.assertions = [ASSERT_ABSENT]
            continue
        if c.assertions:
            continue  # trust model-provided assertion
        # pre-admission medications default to historical
        if c.type in LINKED_TYPES and _HIST.search(ctx):
            c.assertions = [ASSERT_HISTORICAL]
        elif _POSS.search(ctx):
            c.assertions = [ASSERT_POSSIBLE]
    return concepts
