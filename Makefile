# Viettel AI NLP round-1 — tiện chạy bằng `make <lệnh>`
# Chạy `make help` để xem danh sách.

PY      ?= python3
INPUT   ?= data/input
OUT     ?= output
ZIP     ?= output.zip
MODEL   ?= qwen3:8b
GOLD    ?= data/gold

.DEFAULT_GOAL := help

## help: liệt kê các lệnh
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'

## serve: bật server model Ollama (nền)
serve:
	@curl -s http://localhost:11434/api/tags >/dev/null 2>&1 \
		&& echo "Ollama đang chạy ✅" \
		|| (nohup ollama serve >/tmp/ollama.log 2>&1 & sleep 3; echo "đã bật Ollama")

## model: tải model về (1 lần)
model:
	ollama pull $(MODEL)

## predict: chạy AI trích xuất -> output/ + output.zip (~30 phút)
predict: serve
	$(PY) scripts/01_predict.py --input $(INPUT) --out $(OUT) --zip $(ZIP) --model $(MODEL)

## baseline: chạy nhanh KHÔNG cần AI (chỉ luật, để test dây chuyền)
baseline:
	$(PY) scripts/01_predict.py --input $(INPUT) --out $(OUT) --zip $(ZIP) --no-llm

## resolve: tra mã thuốc RxNorm qua LLM-normalize + RxNav (1 lần, cần mạng)
resolve:
	$(PY) scripts/02_resolve_rxnav.py --from-output $(OUT) --llm --model $(MODEL)

## resolve-fast: bản regex (không LLM) — nhanh hơn, kém chính xác trên tên brand
resolve-fast:
	$(PY) scripts/02_resolve_rxnav.py --from-output $(OUT)

## candidates: điền mã thuốc vào output có sẵn (không chạy lại AI) + nén zip
candidates:
	$(PY) scripts/03_apply_candidates.py --pred $(OUT) --zip $(ZIP)

## filter: dọn output — cắt tiền tố + lọc THUỐC giả (is_drug) + relink
filter: serve
	$(PY) scripts/04_clean_output.py --pred $(OUT) --zip $(ZIP) --model $(MODEL)

## submit: dây chuyền đầy đủ predict -> resolve -> candidates -> filter
submit: predict resolve candidates filter
	@echo "Bài nộp sẵn sàng: $(ZIP)"

## eval: chấm điểm với đáp án gold (nếu có) tại data/gold
eval:
	$(PY) scripts/05_evaluate.py --pred $(OUT) --gold $(GOLD)

## test: chạy unit test
test:
	$(PY) tests/test_pipeline.py

## stats: thống kê nhanh output hiện tại
stats:
	@$(PY) -c "import json,glob;cs=[c for f in glob.glob('$(OUT)/*.json') for c in json.load(open(f))];from collections import Counter;print('files:',len(glob.glob('$(OUT)/*.json')),'concepts:',len(cs));print('types:',dict(Counter(c['type'] for c in cs)));d=[c for c in cs if c['type']=='THUỐC'];print('thuốc:',len(d),'có mã:',sum(1 for c in d if c['candidates']))"

## clean: xoá output/ và zip
clean:
	rm -rf $(OUT) $(ZIP)

.PHONY: help serve model predict baseline resolve resolve-fast candidates filter submit eval test stats clean
