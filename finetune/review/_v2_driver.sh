#!/bin/bash
# v2 eval driver: rank-8 and rank-16 v2 models plus a fresh stock run, same judge/threshold
# as the v1 review, seeded. Sequential on purpose (one Ollama, 8 GB-class contention).
cd /Users/you.lawson/jetson-brain || exit 1
source finetune/venv/bin/activate
export BRAIN_ROOT="$PWD/mac-brain-root" BRAIN_INDEX="$PWD/index-mac/bm25.pkl" \
       BRAIN_AUTOLEARN=0 BRAIN_SEED=42 BRAIN_TEMPERATURE=0
for m in llama3.2:3b-parent-ft-v2-r8 llama3.2:3b-parent-ft-v2-r16 llama3.2:3b; do
  tag=$(echo "$m" | tr ':.' '__' | sed 's/llama3_2_3b/llama3_2_3b/')
  for g in parent_gold parent_hard_gold; do
    echo "=== $m / $g === $(date)"
    python3 eval_brain.py --gold "$g.jsonl" --model "$m" --judge-model qwen2.5:7b \
        --threshold 90 --no-cache --two-phase --isolate --mode parent --json \
        --answers-out "finetune/review/${tag}_${g}_v2run_answers.jsonl" \
        > "finetune/review/${tag}_${g}_v2run_scores.json" \
        2> "finetune/review/${tag}_${g}_v2run.log"
    echo "=== $m / $g done (exit $?) === $(date)"
  done
done
echo "V2 REVIEW DONE $(date)"
