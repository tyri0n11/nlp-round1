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

from ..utils.schema import ASSERTIONS, TYPES, Concept

SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất khái niệm y khoa từ bệnh án tiếng Việt. "
    "Chỉ trả về JSON hợp lệ, không giải thích."
)

_TYPE_LIST = ", ".join(TYPES)
_ASSERT_LIST = ", ".join(ASSERTIONS)

USER_TEMPLATE = """Trích xuất tất cả khái niệm y khoa trong đoạn văn dưới đây.

Loại khái niệm (type) cho phép: {types}
Nhãn assertion cho phép: {asserts}

CHỈ dùng 5 loại type sau (KHÔNG tự chế loại khác):
- THUỐC: tên thuốc, gồm cả liều/đường dùng nếu có. VD "amlodipine 10 mg po daily", "Chlorpheniramine 0.4 MG/ML".
- TRIỆU_CHỨNG: triệu chứng lâm sàng. VD "ho", "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi", "khó thở".
- CHẨN_ĐOÁN: chẩn đoán/tên bệnh. VD "bệnh trào ngược dạ dày - thực quản", "xơ gan do rượu".
- TÊN_XÉT_NGHIỆM: TÊN xét nghiệm/chỉ số. VD "TWBC", "NEUT% (Tỷ lệ % bạch cầu trung tính)", "LYPH%".
- KẾT_QUẢ_XÉT_NGHIỆM: chỉ GIÁ TRỊ (con số) kết quả. VD "14,43", "76,4", "12,8".
  QUAN TRỌNG: tên và giá trị là 2 concept TÁCH RIÊNG. VD "TWBC 14,43" -> {{"text":"TWBC","type":"TÊN_XÉT_NGHIỆM"}} VÀ {{"text":"14,43","type":"KẾT_QUẢ_XÉT_NGHIỆM"}}.

Quy tắc:
- text: sao chép CHÍNH XÁC chuỗi con trong văn bản (giữ nguyên hoa/thường, dấu, số, dấu phẩy thập phân).
- Trích NGẮN GỌN, chỉ cụm cốt lõi. BỎ từ dẫn: "bệnh nhân", "xuất hiện", "cảm thấy", "có triệu chứng", "được chẩn đoán", "tình trạng", "ghi nhận", "điều trị".
- KHÔNG trích thủ thuật/phẫu thuật (không thuộc 5 loại trên) trừ khi là xét nghiệm.
- assertions: MẶC ĐỊNH []. Chỉ gán khi rõ: tiền sử/trước nhập viện -> ["isHistorical"]; phủ định (không, chưa, âm tính) -> ["isAbsent"]; nghi ngờ/theo dõi -> ["isPossible"].
- Giữ đúng thứ tự xuất hiện. Không bịa.

Trả về JSON: danh sách các object {{"text":..., "type":..., "assertions":[...]}}.

Văn bản:
<<<
{text}
>>>
"""


def build_prompt(text: str) -> str:
    return USER_TEMPLATE.format(types=_TYPE_LIST, asserts=_ASSERT_LIST, text=text)


def _recover_objects(s: str) -> list:
    """Parse each top-level {...} object independently, skipping malformed or
    truncated ones. Recovers a partial list when the array is cut off or a
    single object is broken (instead of losing the whole record)."""
    items: list = []
    depth = 0
    buf = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == "{":
            if depth == 0:
                buf = []
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                try:
                    items.append(json.loads("".join(buf)))
                except json.JSONDecodeError:
                    pass
                buf = []
        elif depth > 0:
            buf.append(ch)
    return items


def _extract_json(s: str) -> list:
    """Best-effort parse of a JSON array from a model completion."""
    s = s.strip()
    # strip markdown fences and any <think>...</think> block
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    # fall back to per-object recovery (handles truncation / one bad object)
    return _recover_objects(s)


# map free-form types the model tends to invent onto the allowed vocabulary
_TYPE_SYNONYMS = {
    "BỆNH_LÝ": "CHẨN_ĐOÁN",
    "BỆNH": "CHẨN_ĐOÁN",
    "THUỐC_ĐIỀU_TRỊ": "THUỐC",
    "TRIỆU_CHỨNG_LÂM_SÀNG": "TRIỆU_CHỨNG",
    "XÉT_NGHIỆM": "TÊN_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN_HÌNH_ẢNH": "TÊN_XÉT_NGHIỆM",
    "XÉT_NGHIỆM_CẬN_LÂM_SÀNG": "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ": "KẾT_QUẢ_XÉT_NGHIỆM",
    "GIÁ_TRỊ_XÉT_NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
}


def _coerce(items: list) -> List[Concept]:
    out: List[Concept] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        typ = str(it.get("type", "")).strip().upper().replace(" ", "_")
        typ = _TYPE_SYNONYMS.get(typ, typ)
        if not text or typ not in TYPES:
            continue
        # "present" is the default in the gold (example shows current
        # symptoms with assertions=[]), so drop isPresent to avoid Jaccard loss.
        asserts = [a for a in (it.get("assertions") or [])
                   if a in ASSERTIONS and a != "isPresent"]
        out.append(Concept(text=text, type=typ, position=[0, 0], assertions=asserts))
    return out


# --- backends -------------------------------------------------------------
class OllamaBackend:
    """Talks to a local Ollama server (default on Apple Silicon)."""

    def __init__(self, model: str = "qwen2.5:7b-instruct", host: str | None = None,
                 temperature: float = 0.0, num_ctx: int = 8192, think: bool | None = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.num_ctx = num_ctx
        # For hybrid-reasoning models (e.g. qwen3): False disables the slow
        # <think> phase. None = leave model default (non-reasoning models ignore).
        self.think = think

    def generate(self, system: str, user: str) -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": 6144},
        }
        if self.think is not None:
            payload["think"] = self.think
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
