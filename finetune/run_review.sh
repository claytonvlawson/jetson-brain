#!/usr/bin/env bash
# Direct-Bash review driver, replacing the flaky Workflow subagent approach:
# runs stock llama3.2:3b and llama3.2:3b-parent-ft against parent_gold.jsonl and
# parent_hard_gold.jsonl, sequentially per model to avoid Ollama contention.
set -uo pipefail
cd "$(dirname "$0")/.."

source finetune/venv/bin/activate
export BRAIN_ROOT="$PWD/mac-brain-root"
export BRAIN_INDEX="$PWD/index-mac/bm25.pkl"
export BRAIN_AUTOLEARN=0
export BRAIN_SEED=42
export BRAIN_TEMPERATURE=0

OUT="finetune/review"
mkdir -p "$OUT"

run_one() {
  local tag="$1" safe="$2" gold="$3" base="$4"
  echo "=== $safe / $base ==="
  python3 eval_brain.py --gold "$gold" --model "$tag" --judge-model qwen2.5:7b \
    --threshold 90 --no-cache --two-phase --mode parent \
    --answers-out "$OUT/${safe}_${base}_answers.jsonl" --json \
    > "$OUT/${safe}_${base}_scores.json" 2> "$OUT/${safe}_${base}.log"
  echo "=== $safe / $base done (exit $?) ==="
}

# parent_gold for both models is already running independently (stock PID
# $1, ft-v1 PID $2, passed in by the caller). Wait for both to exit before
# starting parent_hard_gold, so at most one eval hits Ollama at a time.
STOCK_PID="${1:?stock parent_gold PID required}"
FT_PID="${2:?ft-v1 parent_gold PID required}"
echo "waiting on stock pid $STOCK_PID and ft pid $FT_PID to finish parent_gold..."
while kill -0 "$STOCK_PID" 2>/dev/null || kill -0 "$FT_PID" 2>/dev/null; do
  sleep 15
done
echo "both parent_gold runs finished, starting parent_hard_gold"

run_one llama3.2:3b llama3_2_3b parent_hard_gold.jsonl parent_hard_gold
run_one llama3.2:3b-parent-ft llama3_2_3b-parent-ft parent_hard_gold.jsonl parent_hard_gold

echo "REVIEW QUEUE DONE"
