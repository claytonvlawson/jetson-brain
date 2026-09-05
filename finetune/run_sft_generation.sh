#!/usr/bin/env bash
# Generate fresh, full-length RAG answers + judge scores for every remaining
# gold set, to be filtered (score >= 80) into SFT training pairs for the
# llama3.2:3b fine-tune. Runs entirely against the Mac-local index-mac/bm25.pkl
# so it needs no Jetson access. See finetune/build_sft_dataset.py for the
# filter/format step that consumes this output.
set -uo pipefail
cd "$(dirname "$0")/.."

source finetune/venv/bin/activate
export BRAIN_ROOT="$PWD/mac-brain-root"
export BRAIN_INDEX="$PWD/index-mac/bm25.pkl"
export BRAIN_AUTOLEARN=0
export BRAIN_SEED=42
export BRAIN_TEMPERATURE=0

OUT_DIR="finetune/sft_raw"
mkdir -p "$OUT_DIR"

run_set() {
  local gold="$1" mode="$2"
  local name
  name="$(basename "$gold" .jsonl)"
  echo "=== $name (mode=$mode) ==="
  python3 eval_brain.py \
    --gold "$gold" --model llama3.2:3b --judge-model qwen2.5:7b \
    --threshold 90 --no-cache --two-phase --mode "$mode" \
    --answers-out "$OUT_DIR/${name}_answers.jsonl" \
    --json > "$OUT_DIR/${name}_scores.json" 2> "$OUT_DIR/${name}.log"
  echo "=== $name done (exit $?) ==="
}

run_set parent_hard_gold.jsonl parent
run_set gold_questions.jsonl default
run_set factual_gold.jsonl default
run_set heldout_gold.jsonl default
run_set depth_gold.jsonl default
run_set multihop_gold.jsonl default
run_set meta_gold.jsonl default

echo "ALL SETS DONE"
