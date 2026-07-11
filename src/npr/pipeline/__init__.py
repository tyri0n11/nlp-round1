"""Inference pipeline, stages run in file-number order:

  _01_ner_llm       extract concepts with the LLM (Qwen3-8B)
  _02_ner_baseline  regex drug fallback / merge
  _03_postprocess   strip "cho/dùng" prefixes; is_drug filter helpers
  _04_align         surface string -> exact [start, end) offsets
  _05_assertions    cue-based assertion backfill
  _06_linking       RxNorm RXCUI lookup
  _00_orchestrator  runs them in order (see Pipeline.run)
"""
from ..config import PipelineConfig
from ._00_orchestrator import Pipeline, _dedup

__all__ = ["Pipeline", "PipelineConfig", "_dedup"]
