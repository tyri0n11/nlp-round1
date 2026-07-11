"""Smoke + metric tests. Run: python -m pytest tests/ (or python tests/test_pipeline.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npr.pipeline._04_align import assign_positions
from npr.utils.evaluate import evaluate, wer
from npr.pipeline import Pipeline, PipelineConfig
from npr.utils.schema import Concept, TYPE_DRUG, TYPE_SYMPTOM, validate_concept

EXAMPLE = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po "
    "daily 2. aspirin 81 mg po daily"
)


def test_positions_snap_to_raw():
    cs = [Concept("amlodipine 10 mg po daily", TYPE_DRUG, [0, 0]),
          Concept("aspirin 81 mg po daily", TYPE_DRUG, [0, 0])]
    out = assign_positions(EXAMPLE, cs)
    assert len(out) == 2
    for c in out:
        s, e = c.position
        assert EXAMPLE[s:e] == c.text
        assert not validate_concept(c, EXAMPLE)


def test_wer_perfect_and_empty():
    g = [Concept("ho", TYPE_SYMPTOM, [0, 2])]
    assert wer(g, g) == 0.0
    assert wer([], []) == 0.0
    assert wer([], [g[0]]) == 1.0


def test_evaluate_identity_is_one():
    gold = {"1": [
        Concept("amlodipine 10 mg po daily", TYPE_DRUG, [56, 81], ["308135"], ["isHistorical"]),
        Concept("ho", TYPE_SYMPTOM, [90, 92], [], []),
    ]}
    scores = evaluate(gold, gold)
    assert abs(scores.final_score - 1.0) < 1e-9


def test_wrong_type_scores_zero_candidates():
    gold = {"1": [Concept("ho", TYPE_SYMPTOM, [0, 2], [], [])]}
    pred = {"1": [Concept("ho", "CHẨN_ĐOÁN", [0, 2], [], [])]}
    s = evaluate(gold, pred)
    # text matches (same words) but assertion/candidate alignment fails
    assert s.text_score == 1.0
    assert s.assertions_score == 0.0


def test_json_recovery_and_synonym_map():
    from npr.pipeline._01_ner_llm import _coerce, _extract_json
    # truncated array (no closing ]) + trailing broken object
    trunc = ('[{"text":"ho","type":"TRIỆU_CHỨNG","assertions":[]},'
             '{"text":"sốt","type":"BỆNH_LÝ","assertions":["isHistorical"]},'
             '{"text":"đau')
    cs = _coerce(_extract_json(trunc))
    assert [(c.type, c.text) for c in cs] == [("TRIỆU_CHỨNG", "ho"), ("CHẨN_ĐOÁN", "sốt")]


def test_ispresent_dropped():
    from npr.pipeline._01_ner_llm import _coerce
    cs = _coerce([{"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": ["isPresent"]}])
    assert cs[0].assertions == []


def test_baseline_pipeline_runs_without_llm():
    pipe = Pipeline(PipelineConfig(use_llm=False))
    out = pipe.run(EXAMPLE)
    assert all(not validate_concept(c, EXAMPLE) for c in out)
    assert any(c.type == TYPE_DRUG for c in out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
