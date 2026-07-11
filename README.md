# Viettel AI — NLP Round 1

Clinical-Vietnamese concept extraction: for each record we output a JSON list of
concepts with `text`, `type`, `candidates` (RxNorm codes for drugs),
`assertions`, and char `position`. Scored by
`0.3*text (1-WER) + 0.3*assertions (Jaccard) + 0.4*candidates (Jaccard)`.

## Approach

A self-hosted LLM (**Qwen3-8B** with `<think>` disabled, ≤9B per competition
rules; Qwen2.5-7B-Instruct is a drop-in alternate) proposes
`{text, type, assertions}`; the pipeline then:

1. **aligns** each surface string to exact `[start, end)` offsets (`npr.align`),
2. **backfills assertions** with Vietnamese cue rules (`npr.assertions.rules`),
3. **links drugs** to RxNorm RXCUIs offline (`npr.linking.rxnorm`),
4. **validates** every concept (`raw[s:e] == text`, known type/assertions).

A dependency-free **heuristic baseline** (`npr.ner.baseline`, drug lists) runs as
a fallback / merge source, so the pipeline produces a valid `output.zip` even
with no model available.

```
src/npr/
  schema.py        types (THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, THỦ_THUẬT, XÉT_NGHIỆM),
                   assertion vocab, Concept, validator
  io_utils.py      read N.txt -> write N.json (+ output.zip with output/ root)
  align.py         surface string -> exact char offsets
  ner/llm.py       Qwen2.5-7B extractor; Ollama + transformers backends
  ner/baseline.py  regex drug-list fallback
  linking/rxnorm.py  offline RXCUI lookup
  assertions/rules.py  cue-based assertion backfill
  pipeline.py      orchestration
  evaluate.py      local (unofficial) WER + Jaccard scorer
scripts/           predict.py, evaluate.py, build_rxnorm.py
```

## Setup

Requires Python 3.9+. The core pipeline uses only the standard library.

### Option A — Ollama (recommended on Apple Silicon)

```bash
ollama serve &                       # start the local server
ollama pull qwen3:8b                 # ≤9B, self-hosted (default; think disabled)
# alternate: ollama pull qwen2.5:7b-instruct
```

### Option B — transformers (MPS on Mac, or CUDA for the rebuild box)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install transformers torch accelerate sentencepiece   # see requirements.txt
```

### RxNorm candidates (offline)

Candidates are RxNorm RXCUIs. Build the lookup table once from an RxNorm RRF
release (no external API is used at inference time — competition rule):

```bash
python3 scripts/build_rxnorm.py --rrf /path/to/RxNorm/rrf/RXNCONSO.RRF
# -> data/resources/rxnorm.json   (ingredient/name -> [RXCUI])
```

Without this file, drug `candidates` are left empty (everything else still runs).

## Run inference → `output.zip`

```bash
# LLM pipeline (Ollama)
python3 scripts/predict.py --input data/input --out output --zip output.zip

# transformers backend
python3 scripts/predict.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct

# heuristic baseline only (no model, instant)
python3 scripts/predict.py --no-llm
```

`output.zip` unzips to `output/1.json … 100.json` as required by the brief.

## Evaluate locally

`scripts/evaluate.py` implements an **unofficial** reading of the metric (see the
interpretation notes in `src/npr/evaluate.py` — the BTC evaluator is
authoritative). Point it at a gold dir laid out like the predictions:

```bash
python3 scripts/evaluate.py --pred output --gold data/gold
```

## Tests

```bash
python3 tests/test_pipeline.py      # or: python3 -m pytest tests/
```

## Notes / TODO for a competitive submission

- **No labeled train data / RxNorm dict shipped with the task** — provide them to
  lift `candidates` and `assertions` quality. Current linking/assertions are
  offline heuristics.
- Few-shot examples in `ner/llm.py` (`USER_TEMPLATE`) can be expanded, or the
  model fine-tuned/LoRA-tuned once gold data exists.
- Confirm the exact closed set of `type`/`assertions` labels against the BTC
  annotation guide and update `schema.py`.
```
