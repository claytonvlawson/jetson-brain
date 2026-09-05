#!/usr/bin/env python3
"""Turn finetune/sft_raw/*_answers.jsonl + *_scores.json into an MLX chat-format
SFT dataset (train/valid JSONL) for fine-tuning llama3.2:3b.

Each *_answers.jsonl row (written by eval_brain.py --answers-out) has
{id, question, answer, sources, served_by}. Each *_scores.json (written by
eval_brain.py --json) has {"results": [{id, score, ...}, ...]}. We join them on
id and keep only answers scoring >= --min-score, since those are judged
complete/accurate enough to train on -- an answer with real gaps would just
teach the model to repeat the gap.

Usage:
    python3 finetune/build_sft_dataset.py --min-score 80
"""
import argparse
import glob
import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "sft_raw")

# The system prompt the fine-tuned model should learn to answer under. Kept in
# sync with brain.py's persona prompts would be ideal, but for a first pass a
# single representative prompt is enough -- MLX chat template applies it to
# every example uniformly.
SYSTEM_PROMPT = (
    "You are a knowledgeable, warm assistant for parents of deaf and hard of "
    "hearing children, grounded in ASL, Deaf culture, and language deprivation "
    "research. Answer clearly and completely."
)


def load_scores(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return {r["id"]: r["score"] for r in d.get("results", [])}


def load_answers(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-score", type=int, default=80)
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "data"))
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = []
    skipped_no_scores = 0
    skipped_low_score = 0
    skipped_no_answer = 0

    for answers_path in sorted(glob.glob(os.path.join(args.raw_dir, "*_answers.jsonl"))):
        name = os.path.basename(answers_path).replace("_answers.jsonl", "")
        scores_path = os.path.join(args.raw_dir, name + "_scores.json")
        if not os.path.exists(scores_path):
            print("skip %s: no scores file" % name, file=sys.stderr)
            skipped_no_scores += len(load_answers(answers_path))
            continue
        scores = load_scores(scores_path)
        answers = load_answers(answers_path)
        for qid, row in answers.items():
            score = scores.get(qid)
            if score is None:
                skipped_no_scores += 1
                continue
            answer = (row.get("answer") or "").strip()
            if not answer:
                skipped_no_answer += 1
                continue
            if score < args.min_score:
                skipped_low_score += 1
                continue
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": answer},
                ],
                "_meta": {"source": name, "id": qid, "score": score},
            })

    print("kept %d pair(s); skipped %d (no answer), %d (score < %d), %d (no score row)"
          % (len(pairs), skipped_no_answer, skipped_low_score, args.min_score, skipped_no_scores))

    if len(pairs) < 150:
        print("WARNING: dataset has fewer than 150 examples -- fine-tuning on "
              "this is likely to overfit/regurgitate rather than generalize. "
              "Consider lowering --min-score or expanding data collection "
              "before training.", file=sys.stderr)

    random.Random(args.seed).shuffle(pairs)
    n_valid = max(1, int(len(pairs) * args.valid_frac)) if pairs else 0
    valid, train = pairs[:n_valid], pairs[n_valid:]

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, rows in (("train", train), ("valid", valid)):
        out_path = os.path.join(args.out_dir, split_name + ".jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
        print("wrote %d example(s) to %s" % (len(rows), out_path))

    # Keep the _meta alongside for provenance/debugging, not used by mlx_lm.
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row["_meta"]) + "\n")


if __name__ == "__main__":
    main()
