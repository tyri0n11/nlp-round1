"""Map extracted surface strings back to exact [start, end) char offsets.

The LLM returns concept texts; positions must be char offsets into the raw
record. We find non-overlapping occurrences left-to-right, preferring an exact
match and falling back to a whitespace-insensitive match (clinical text has
irregular spacing, e.g. "Không  buồn nôn").
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..utils.schema import Concept


def _find_exact(haystack: str, needle: str, start: int) -> Optional[Tuple[int, int]]:
    idx = haystack.find(needle, start)
    if idx == -1:
        return None
    return idx, idx + len(needle)


def _find_flexible(haystack: str, needle: str, start: int) -> Optional[Tuple[int, int]]:
    """Match `needle` allowing runs of whitespace to differ."""
    norm = re.escape(needle.strip())
    norm = re.sub(r"\\\s+", r"\\s+", norm)
    if not norm:
        return None
    m = re.compile(norm).search(haystack, start)
    if not m:
        return None
    return m.start(), m.end()


def assign_positions(raw: str, concepts: List[Concept]) -> List[Concept]:
    """Fill each concept's position and snap its text to the raw substring.

    Advances a cursor so repeated mentions map to successive occurrences.
    Concepts whose text cannot be located are dropped (they cannot be scored).
    """
    out: List[Concept] = []
    cursor = 0
    for c in concepts:
        text = c.text.strip()
        if not text:
            continue
        span = _find_exact(raw, text, cursor)
        if span is None:
            span = _find_flexible(raw, text, cursor)
        if span is None:
            # retry from the beginning (order from the model may be imperfect)
            span = _find_exact(raw, text, 0) or _find_flexible(raw, text, 0)
        if span is None:
            continue
        s, e = span
        c.position = [s, e]
        c.text = raw[s:e]  # snap to the exact substring so text==raw[s:e]
        cursor = e
        out.append(c)
    return out
