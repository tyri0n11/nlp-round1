"""LLM-based concept extractor (self-hosted, <=9B).

Default target: Qwen2.5-7B-Instruct. Backends are pluggable so the same
pipeline runs on a Mac (Ollama / llama.cpp GGUF — practical for Apple Silicon)
and on a CUDA box (vLLM / transformers) for the private-test rebuild.

The model only proposes {text, type, assertions}. Char positions are recovered
by string alignment (npr.align) and RxNorm candidates are added downstream by
npr.linking — keeping the LLM job small and its output easy to validate.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

from ..schema import ASSERTIONS, TYPES, Concept

SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất khái niệm y khoa từ bệnh án tiếng Việt. "
    "Chỉ trả về JSON hợp lệ, không giải thích."
)

_TYPE_LIST = ", ".join(TYPES)
_ASSERT_LIST = ", ".join(ASSERTIONS)

USER_TEMPLATE = """Trích xuất tất cả khái niệm y khoa trong đoạn văn dưới đây.

Loại khái niệm (type) cho phép: {types}
Nhãn assertion cho phép: {asserts}

Quy tắc:
- text: sao chép CHÍNH XÁC chuỗi con xuất hiện trong văn bản (giữ nguyên chữ hoa/thường, dấu, số, đơn vị liều).
- THUỐC: gồm cả liều và đường dùng nếu có (vd "amlodipine 10 mg po daily").
- TRIỆU_CHỨNG / CHẨN_ĐOÁN / THỦ_THUẬT / XÉT_NGHIỆM: chỉ lấy cụm mô tả, KHÔNG kèm assertion cho triệu chứng (để danh sách rỗng) trừ khi bị phủ định/nghi ngờ/tiền sử.
- assertions: chỉ dùng khi phù hợp; thuốc/khái niệm thuộc tiền sử -> ["isHistorical"]; phủ định -> ["isAbsent"]; nghi ngờ -> ["isPossible"]. Mặc định để [].
- Giữ đúng thứ tự xuất hiện. Không bịa khái niệm không có trong văn bản.

Trả về JSON: danh sách các object {{"text":..., "type":..., "assertions":[...]}}.

Văn bản:
<<<
{text}
>>>
"""


def build_prompt(text: str) -> str:
    return USER_TEMPLATE.format(types=_TYPE_LIST, asserts=_ASSERT_LIST, text=text)


def _extract_json(s: str) -> list:
    """Best-effort parse of a JSON array from a model completion."""
    s = s.strip()
    # strip markdown fences
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []


def _coerce(items: list) -> List[Concept]:
    out: List[Concept] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        typ = str(it.get("type", "")).strip()
        if not text or typ not in TYPES:
            continue
        asserts = [a for a in (it.get("assertions") or []) if a in ASSERTIONS]
        out.append(Concept(text=text, type=typ, position=[0, 0], assertions=asserts))
    return out


# --- backends -------------------------------------------------------------
class OllamaBackend:
    """Talks to a local Ollama server (default on Apple Silicon)."""

    def __init__(self, model: str = "qwen2.5:7b-instruct", host: str | None = None,
                 temperature: float = 0.0, num_ctx: int = 8192):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.num_ctx = num_ctx

    def generate(self, system: str, user: str) -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]


class TransformersBackend:
    """transformers pipeline for MPS (Mac) or CUDA. Loads lazily."""

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", device: str | None = None,
                 max_new_tokens: int = 2048):
        self.model_id = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tok = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)

    def generate(self, system: str, user: str) -> str:
        self._ensure()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)


def make_backend(name: str, **kwargs):
    name = (name or "ollama").lower()
    if name == "ollama":
        return OllamaBackend(**kwargs)
    if name in ("transformers", "hf"):
        return TransformersBackend(**kwargs)
    raise ValueError(f"unknown LLM backend {name!r}")


class LLMExtractor:
    def __init__(self, backend):
        self.backend = backend

    def extract(self, text: str) -> List[Concept]:
        completion = self.backend.generate(SYSTEM_PROMPT, build_prompt(text))
        return _coerce(_extract_json(completion))
