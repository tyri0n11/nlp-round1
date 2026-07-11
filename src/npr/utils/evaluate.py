"""Local (unofficial) scorer for the round-1 metric.

    final_score = 0.3*text_score + 0.3*assertions_score + 0.4*candidates_score

IMPORTANT — interpretation notes
--------------------------------
The official BTC evaluator is authoritative. The formula in the brief leaves a
couple of aggregation details under-specified, so this module encodes an
explicit, documented interpretation you can sanity-check against and swap out:

* Concepts are aligned between gold and prediction by an exact (text, type)
  key. The brief states that a right-text / wrong-type prediction is counted
  as a *new* concept and scores 0 on all three sub-metrics — that is exactly
  what keying on (text, type) produces (the mismatched pair never aligns).

* text_score = mean over samples of (1 - WER(i)), where WER(i) is the
  word-level edit distance between the concatenation of gold concept texts and
  the concatenation of predicted concept texts (in file order), normalised by
  the gold word count.

* J_X(i) (assertions / candidates) is the alignment-weighted mean of per-concept
  Jaccard: matched pairs contribute their Jaccard; unmatched gold/pred concepts
  contribute 0. For candidates the per-concept weight is (len(gold_cands)+1),
  matching the global candidates weighting in the brief; assertions use weight 1.
  A sample with no gold and no pred items for field X scores J_X(i)=1.

* candidates_score uses the brief's global weighting:
      sum_i  J_cand(i) * W(i)  /  sum_i W(i),   W(i) = Σ_k (len(gold_cand_k)+1)
  assertions_score is the simple mean of J_assert(i) over samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import Concept


# --- word error rate ------------------------------------------------------
def _tokens(concepts: Sequence[Concept]) -> List[str]:
    words: List[str] = []
    for c in concepts:
        words.extend(c.text.split())
    return words


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(gold: Sequence[Concept], pred: Sequence[Concept]) -> float:
    g, p = _tokens(gold), _tokens(pred)
    if not g:
        return 0.0 if not p else 1.0
    return _edit_distance(g, p) / len(g)


# --- alignment ------------------------------------------------------------
def _key(c: Concept) -> Tuple[str, str]:
    return (c.text.strip(), c.type)


def _align(gold: Sequence[Concept], pred: Sequence[Concept]):
    """Greedy 1-1 alignment on (text, type). Returns (pairs, unmatched_gold,
    unmatched_pred)."""
    pred_by_key: Dict[Tuple[str, str], List[Concept]] = {}
    for c in pred:
        pred_by_key.setdefault(_key(c), []).append(c)
    pairs: List[Tuple[Concept, Concept]] = []
    unmatched_gold: List[Concept] = []
    for gc in gold:
        bucket = pred_by_key.get(_key(gc))
        if bucket:
            pairs.append((gc, bucket.pop(0)))
        else:
            unmatched_gold.append(gc)
    unmatched_pred = [c for b in pred_by_key.values() for c in b]
    return pairs, unmatched_gold, unmatched_pred


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pooled_field_jaccard(gold, pred, field: str) -> Tuple[float, float]:
    """Literal reading of the brief: J_X(i) is ONE set-Jaccard over all field-X
    values pooled across the sample's concepts. Returns (J, W).

    NOTE: this reading conflicts with the brief's own note that a right-text /
    wrong-type concept scores 0 (pooled empty-vs-empty would score 1), which is
    why `aligned` is the default. Kept for comparison.
    """
    g_pool = [v for c in gold for v in getattr(c, field)]
    p_pool = [v for c in pred for v in getattr(c, field)]
    w_sample = sum(len(getattr(c, field)) + 1 for c in gold)
    return _jaccard(g_pool, p_pool), w_sample


def _sample_field_jaccard(gold, pred, field: str, weighted: bool) -> Tuple[float, float]:
    """Return (J_field(i), W(i)) for one sample."""
    pairs, um_gold, um_pred = _align(gold, pred)
    num = 0.0
    den = 0.0
    w_sample = 0.0
    for gc, pc in pairs:
        g_items = getattr(gc, field)
        p_items = getattr(pc, field)
        w = (len(g_items) + 1) if weighted else 1.0
        num += _jaccard(g_items, p_items) * w
        den += w
        w_sample += (len(g_items) + 1)
    for gc in um_gold:
        w = (len(getattr(gc, field)) + 1) if weighted else 1.0
        den += w
        w_sample += (len(getattr(gc, field)) + 1)
    for pc in um_pred:
        w = 1.0
        den += w
    j = 1.0 if den == 0 else num / den
    return j, w_sample


@dataclass
class Scores:
    text_score: float
    assertions_score: float
    candidates_score: float

    @property
    def final_score(self) -> float:
        return 0.3 * self.text_score + 0.3 * self.assertions_score + 0.4 * self.candidates_score

    def as_dict(self) -> dict:
        return {
            "text_score": self.text_score,
            "assertions_score": self.assertions_score,
            "candidates_score": self.candidates_score,
            "final_score": self.final_score,
        }


def evaluate(gold: Dict[str, List[Concept]], pred: Dict[str, List[Concept]],
             mode: str = "aligned") -> Scores:
    """mode="aligned" (default): per-concept Jaccard keyed by (text,type), the
    only reading consistent with the brief's wrong-type note. mode="pooled":
    literal sample-level set Jaccard from the written formula (for comparison)."""
    ids = list(gold.keys())
    n = len(ids) or 1
    field_j = _pooled_field_jaccard if mode == "pooled" else (
        lambda g, p, f: _sample_field_jaccard(g, p, f, weighted=(f == "candidates"))
    )

    text_sum = 0.0
    assert_sum = 0.0
    cand_num = 0.0
    cand_den = 0.0

    for rid in ids:
        g = gold.get(rid, [])
        p = pred.get(rid, [])
        text_sum += 1.0 - wer(g, p)

        j_a, _ = field_j(g, p, "assertions")
        assert_sum += j_a

        j_c, w_i = field_j(g, p, "candidates")
        cand_num += j_c * w_i
        cand_den += w_i

    return Scores(
        text_score=text_sum / n,
        assertions_score=assert_sum / n,
        candidates_score=(cand_num / cand_den) if cand_den else 1.0,
    )
