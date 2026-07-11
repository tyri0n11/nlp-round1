"""Pipeline configuration (+ optional YAML loader for config/default.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    use_llm: bool = True
    use_baseline_fallback: bool = True
    llm_backend: str = "ollama"          # ollama | transformers
    llm_model: str = "qwen3:8b"          # <=9B, self-hosted (competition rule)
    llm_think: Optional[bool] = False    # False disables slow <think> on qwen3
    rxnorm_path: str = "data/resources/rxnorm.json"


def load_yaml(path: str) -> PipelineConfig:
    """Build a PipelineConfig from config/default.yaml (needs pyyaml)."""
    import yaml

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    ner = d.get("ner", {})
    linking = d.get("linking", {})
    return PipelineConfig(
        use_llm=ner.get("use_llm", True),
        use_baseline_fallback=ner.get("use_baseline_fallback", True),
        llm_backend=ner.get("llm_backend", "ollama"),
        llm_model=ner.get("llm_model", "qwen3:8b"),
        llm_think=ner.get("llm_think", False),
        rxnorm_path=linking.get("rxnorm_path", "data/resources/rxnorm.json"),
    )
