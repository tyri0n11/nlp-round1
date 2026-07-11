"""Rule-based assertion backfill / correction.

The LLM proposes assertions; these deterministic rules catch the highest-signal
Vietnamese cues and enforce section-based defaults (medications listed under a
"thuốc trước khi nhập viện" / "tiền sử" heading are historical). Rules only add
an assertion when the model left it empty, except negation which overrides.
"""
from __future__ import annotations

import re
from typing import List

from ..utils.schema import (
    ASSERT_ABSENT,
    ASSERT_FAMILY,
    ASSERT_HISTORICAL,
    ASSERT_POSSIBLE,
    LINKED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_SYMPTOM,
    Concept,
)

# window of text just before a concept used to read local cues
_WINDOW = 40

_NEG = re.compile(r"\b(không|chưa|phủ định|loại trừ|âm tính)\b", re.I)
# diagnostic uncertainty only — NOT bare "có thể" (that also means "is able to")
_POSS = re.compile(
    r"(nghi ngờ|nghĩ (?:nhiều )?(?:đến|tới)|khả năng|theo dõi|chưa loại trừ|"
    r"chưa rõ|chưa xác định|gợi ý|hướng (?:đến|tới)|lo ngại|\?)",
    re.I,
)
_HIST = re.compile(
    r"(tiền sử|trước khi nhập viện|trước nhập viện|đã (?:từng|dùng|sử dụng)|"
    r"bệnh sử|đã được chẩn đoán)",
    re.I,
)
# isFamily only makes sense for a condition someone else has
_FAMILY_OK = {TYPE_DIAGNOSIS, TYPE_SYMPTOM}
_FAMILY_CUE = re.compile(
    r"(gia đình|người thân|mẹ|bố|cha|anh|chị|em|ông|bà|con)\b", re.I
)


def _context(raw: str, c: Concept) -> str:
    s = max(0, c.position[0] - _WINDOW)
    return raw[s:c.position[0]]


def _sanitize(raw: str, c: Concept) -> None:
    """Drop LLM assertions that are type-inconsistent or lack textual support."""
    a = list(c.assertions)
    ctx = _context(raw, c)
    if ASSERT_FAMILY in a and (c.type not in _FAMILY_OK or not _FAMILY_CUE.search(ctx)):
        # a drug/test/procedure can't be "someone else's"; need a family cue
        a = [x for x in a if x != ASSERT_FAMILY]
    if ASSERT_POSSIBLE in a and not _POSS.search(ctx):
        # kill false-positive isPossible (e.g. "có thể tự đi lại" = capability)
        a = [x for x in a if x != ASSERT_POSSIBLE]
    c.assertions = a


def resanitize(raw: str, concepts: List[Concept]) -> List[Concept]:
    """Sanitize already-set assertions on existing output (no backfill)."""
    for c in concepts:
        _sanitize(raw, c)
    return concepts


def apply(raw: str, concepts: List[Concept]) -> List[Concept]:
    for c in concepts:
        _sanitize(raw, c)
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
