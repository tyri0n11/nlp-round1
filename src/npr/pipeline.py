"""End-to-end inference pipeline: raw text -> validated Concept list.

Stages:
  1. NER: LLM extractor (Qwen2.5-7B) and/or heuristic baseline propose
     {text, type, assertions}.
  2. Positions: align surface strings to exact [start, end) offsets.
  3. Assertions: rule-based backfill / negation override.
  4. Linking: RxNorm candidates for drug concepts.
  5. Validate + dedup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import align
from .assertions import rules as assert_rules
from .linking.rxnorm import RxNormLinker
from .ner import baseline as ner_baseline
from .postprocess import clean_spans
from .schema import Concept, validate_concept


@dataclass
class PipelineConfig:
    use_llm: bool = True
    use_baseline_fallback: bool = True
    llm_backend: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_think: Optional[bool] = False  # False disables slow <think> on qwen3
    rxnorm_path: str = "data/resources/rxnorm.json"


def _dedup(concepts: List[Concept]) -> List[Concept]:
    seen = set()
    out = []
    for c in sorted(concepts, key=lambda x: (x.position[0], x.position[1])):
        key = (c.position[0], c.position[1], c.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.linker = RxNormLinker.from_json(cfg.rxnorm_path)
        self._extractor = None
        if cfg.use_llm:
            from .ner.llm import LLMExtractor, make_backend

            kw = {"model": cfg.llm_model}
            if cfg.llm_backend == "ollama":
                kw["think"] = cfg.llm_think
            backend = make_backend(cfg.llm_backend, **kw)
            self._extractor = LLMExtractor(backend)

    def run(self, raw: str) -> List[Concept]:
        proposed: List[Concept] = []
        if self._extractor is not None:
            proposed = self._extractor.extract(raw)
        if not proposed and self.cfg.use_baseline_fallback:
            proposed = ner_baseline.extract(raw)
        elif self.cfg.use_baseline_fallback:
            # merge baseline drug spans the LLM may have missed
            proposed = proposed + ner_baseline.extract(raw)

        proposed = clean_spans(proposed)  # strip "cho/dùng/..." prefixes
        located = align.assign_positions(raw, proposed)
        located = _dedup(located)
        located = assert_rules.apply(raw, located)
        self.linker.apply(located)

        # final validation: drop anything that cannot be scored cleanly
        clean: List[Concept] = []
        for c in located:
            if not validate_concept(c, raw):
                clean.append(c)
        return _dedup(clean)
