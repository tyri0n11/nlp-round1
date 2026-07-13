"""Data schema and controlled vocabularies for the Viettel AI NLP round-1 task.

Output format (per record, a JSON list of concepts):

    [
      {
        "text": "amlodipine 10 mg po daily",   # exact surface string
        "type": "THUỐC",                        # concept type (see TYPES)
        "candidates": ["308135"],               # RxNorm codes (drugs) — may be []
        "assertions": ["isHistorical"],         # assertion tags — may be []
        "position": [58, 83]                    # [start, end) char offsets into the raw text
      },
      ...
    ]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# --- Concept types --------------------------------------------------------
# Confirmed from the official task rules (5 types):
TYPE_DRUG = "THUỐC"                    # drug -> RxNorm candidates
TYPE_SYMPTOM = "TRIỆU_CHỨNG"           # symptom (no candidates)
TYPE_DIAGNOSIS = "CHẨN_ĐOÁN"           # diagnosis -> ICD-10 candidates (a set)
TYPE_TEST_NAME = "TÊN_XÉT_NGHIỆM"      # test name, e.g. "TWBC", "NEUT%"
TYPE_TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"  # test result value, e.g. "14,43"

TYPES = [
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_DIAGNOSIS,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
]

# Types that carry RxNorm candidate codes. Only drugs are linked to RxNorm.
LINKED_TYPES = {TYPE_DRUG}

# --- Assertions -----------------------------------------------------------
# i2b2/n2c2-style assertion tags. The example uses `isHistorical` for
# pre-admission medications. Keep this list as the closed vocabulary the
# model/rules are allowed to emit.
ASSERT_HISTORICAL = "isHistorical"
ASSERT_PRESENT = "isPresent"
ASSERT_ABSENT = "isAbsent"
ASSERT_POSSIBLE = "isPossible"
ASSERT_CONDITIONAL = "isConditional"
ASSERT_HYPOTHETICAL = "isHypothetical"
ASSERT_FAMILY = "isFamily"  # associated with someone else

ASSERTIONS = [
    ASSERT_HISTORICAL,
    ASSERT_PRESENT,
    ASSERT_ABSENT,
    ASSERT_POSSIBLE,
    ASSERT_CONDITIONAL,
    ASSERT_HYPOTHETICAL,
    ASSERT_FAMILY,
]


@dataclass
class Concept:
    text: str
    type: str
    position: List[int]                      # [start, end)
    candidates: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Field order matches the reference example for readability.
        return {
            "text": self.text,
            "type": self.type,
            "candidates": list(self.candidates),
            "assertions": list(self.assertions),
            "position": [int(self.position[0]), int(self.position[1])],
        }

    @staticmethod
    def from_dict(d: dict) -> "Concept":
        return Concept(
            text=d["text"],
            type=d["type"],
            position=list(d.get("position", [0, 0])),
            candidates=list(d.get("candidates", []) or []),
            assertions=list(d.get("assertions", []) or []),
        )


def validate_concept(c: Concept, raw_text: Optional[str] = None) -> List[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: List[str] = []
    if c.type not in TYPES:
        problems.append(f"unknown type {c.type!r}")
    for a in c.assertions:
        if a not in ASSERTIONS:
            problems.append(f"unknown assertion {a!r}")
    if len(c.position) != 2 or c.position[0] > c.position[1]:
        problems.append(f"bad position {c.position!r}")
    if c.type not in LINKED_TYPES and c.candidates:
        problems.append(f"type {c.type!r} should not carry candidates")
    if raw_text is not None and len(c.position) == 2:
        s, e = c.position
        if not (0 <= s <= e <= len(raw_text)):
            problems.append(f"position {c.position!r} out of bounds (len={len(raw_text)})")
        elif raw_text[s:e] != c.text:
            problems.append(
                f"position/text mismatch: raw[{s}:{e}]={raw_text[s:e]!r} != {c.text!r}"
            )
    return problems
