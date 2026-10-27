#!/usr/bin/env python3
"""Expand the ~55 vault-grounded parent-topic gold questions into ~1000 persona-
flavored variants for the v3 parent-LoRA fine-tune (target: cross the ~150-200
usable-SFT-pair floor that v1/v2 fell short of; see
jetson-brain-parent-finetune-v1-verdict memory).

Design: each TOPIC carries the real must_cover/must_not facts (unchanged,
already vault-grounded, reused from parent_gold.jsonl / parent_expansion_gold.jsonl
/ parent_hard_gold.jsonl / new_topics.jsonl). We do NOT invent new facts per
persona -- that would need per-question grounding verification against the
vault, which doesn't scale to 1000. Instead each persona variant wraps the same
topic's question in 2 sentences of framing (always the emotional/happiness
dimension plus one other dimension, rotated) drawn from a fixed phrase bank for
economic status, education level, happiness/emotional state, deaf-identity
stance, and family closeness. must_cover/must_not stay identical to the parent
topic, so eval_brain.py's judge grades the same facts regardless of framing --
this also stress-tests whether RAG retrieval and the judge stay robust to
paraphrase/persona noise, not just clean phrasing.

Usage:
    python3 generate_persona_gold.py --target 1000 --out persona_gold_1000.jsonl
"""
import argparse
import glob
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # ~/jetson-brain

EXISTING_GOLD_FILES = [
    os.path.join(ROOT, "parent_gold.jsonl"),
    os.path.join(ROOT, "parent_expansion_gold.jsonl"),
    os.path.join(ROOT, "parent_hard_gold.jsonl"),
]
NEW_TOPICS_FILE = os.path.join(SCRIPT_DIR, "new_topics.jsonl")

DIMENSIONS = {
    "economic_status": {
        "low_income": "We're a low-income family and money for extra services is tight.",
        "lower_middle": "We're getting by paycheck to paycheck right now.",
        "middle_income": "We're a middle-income family with a decent but not unlimited budget.",
        "affluent": "Money isn't the constraint for us; we can afford whatever services we need.",
    },
    "education_level": {
        "no_college": "Neither of us finished college.",
        "some_college": "I did some college but never finished a degree.",
        "bachelors": "I have a bachelor's degree, though nothing related to this.",
        "graduate": "I have a graduate degree, though nothing in this field.",
    },
    "happiness_level": {
        "scared_grieving": "I'm honestly scared and still grieving the future I had imagined.",
        "hopeful": "I'm trying to stay hopeful and focus on next steps.",
        "angry_frustrated": "I'm frustrated and honestly a little angry at how confusing this all is.",
        "calm_confident": "I'm approaching this calmly and just want the facts.",
        "guilty": "I keep wondering if this is somehow my fault.",
        "denial_minimizing": "Honestly part of me still hopes this isn't as big a deal as people are making it sound.",
    },
    "deaf_identity_stance": {
        "oral_only": "We were hoping to focus on spoken language and hearing technology.",
        "asl_first": "We want to raise our child fully immersed in ASL and Deaf culture.",
        "bimodal_open": "We're open to whatever combination of sign and spoken language works best.",
        "resistant_to_sign": "Honestly, part of our family isn't sold on sign language at all.",
    },
    "family_closeness": {
        "isolated": "We don't have much family nearby to lean on.",
        "close_extended": "We have close extended family who want to help but don't know how.",
        "single_parent": "I'm doing this as a single parent.",
        "multigenerational": "We live in a multigenerational household and everyone has an opinion.",
    },
}

SECONDARY_DIMS = ["economic_status", "education_level", "deaf_identity_stance", "family_closeness"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_topics():
    topics = []
    seen_ids = set()
    for path in EXISTING_GOLD_FILES + [NEW_TOPICS_FILE]:
        for row in load_jsonl(path):
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            topics.append(row)
    return topics


def build_variant(topic, idx, rng):
    happy_key = rng.choice(list(DIMENSIONS["happiness_level"]))
    secondary_dim = SECONDARY_DIMS[idx % len(SECONDARY_DIMS)]
    secondary_key = rng.choice(list(DIMENSIONS[secondary_dim]))

    persona = {
        "happiness_level": happy_key,
        secondary_dim: secondary_key,
    }
    frame_sentences = [
        DIMENSIONS["happiness_level"][happy_key],
        DIMENSIONS[secondary_dim][secondary_key],
    ]
    rng.shuffle(frame_sentences)
    framed_question = " ".join(frame_sentences) + " " + topic["question"]

    return {
        "id": "%s_p%03d" % (topic["id"], idx),
        "concept": topic["concept"],
        "base_topic_id": topic["id"],
        "persona": persona,
        "question": framed_question,
        "must_cover": topic["must_cover"],
        "must_not": topic.get("must_not", []),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "persona_gold_1000.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    topics = load_topics()
    n_topics = len(topics)
    base_n = args.target // n_topics
    extra = args.target - base_n * n_topics  # distribute remainder to first `extra` topics

    rng = random.Random(args.seed)
    out_rows = []
    for t_i, topic in enumerate(topics):
        n = base_n + (1 if t_i < extra else 0)
        for v_i in range(n):
            out_rows.append(build_variant(topic, v_i, rng))

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("topics: %d" % n_topics)
    print("rows written: %d -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
