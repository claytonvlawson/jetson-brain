#!/usr/bin/env bash
# Background driver: run eval_brain.py (RAG answer + judge score) over all 10
# 100-question shards of persona_gold_1000.jsonl, mode=parent, against the
# Mac-local index. Each shard writes its own answers/scores/log so a crash
# mid-run only loses the in-flight shard, not the whole job (eval_brain.py's
# --answers-out only writes at the very end of a run, so a single 1000-row
# invocation would lose everything on a crash hours in).
#
# Run this via nohup + disown, not in the foreground and not via a Workflow
# subagent expected to wait out the full run (see
# jetson-brain-parent-finetune-v1-verdict memory: a review subagent gave up
# after ~2 min on a 15-35 min run and fabricated placeholder scores).
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> ~/jetson-brain

source finetune/venv/bin/activate
export BRAIN_ROOT="$PWD/mac-brain-root"
export BRAIN_INDEX="$PWD/index-mac/bm25.pkl"
export BRAIN_AUTOLEARN=0
export BRAIN_SEED=42
export BRAIN_TEMPERATURE=0

SHARD_DIR="finetune/persona_gold/shards"
OUT_DIR="finetune/persona_gold/eval_out"
mkdir -p "$OUT_DIR"

for shard in "$SHARD_DIR"/shard_*.jsonl; do
  name="$(basename "$shard" .jsonl)"
  done_marker="$OUT_DIR/${name}.done"
  if [ -f "$done_marker" ]; then
    echo "=== $name already done, skipping ==="
    continue
  fi
  echo "=== $name starting $(date) ==="
  python3 eval_brain.py \
    --gold "$shard" --model llama3.2:3b --judge-model qwen2.5:7b \
    --threshold 90 --no-cache --two-phase --mode parent \
    --answers-out "$OUT_DIR/${name}_answers.jsonl" \
    --json > "$OUT_DIR/${name}_scores.json" 2> "$OUT_DIR/${name}.log"
  status=$?
  echo "=== $name done (exit $status) $(date) ==="
  # eval_brain.py exits 5 when some questions score below --threshold, which
  # is the expected/normal case here (threshold 90 is stricter than the >=80
  # SFT-pair filter) -- only a real crash/error should skip the done marker.
  if [ $status -eq 0 ] || [ $status -eq 5 ]; then
    touch "$done_marker"
  fi
done

echo "ALL SHARDS DONE $(date)"
touch "$OUT_DIR/ALL_DONE"
