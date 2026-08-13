#!/usr/bin/env python3
"""Retrieval-only eval: does the right note reach the context window at all?

Why this exists. eval_brain.py measures end-to-end answer quality with a
llama3.2:3b judge. One gold_questions arm costs ~1742s and a 5-question swing
at n=28 may be judge noise nobody has quantified. That is far too slow and too
loud to tune a retriever against.

This harness scores the retrieval half only, with no LLM in the loop at all:
for each question we know which note holds the answer, so we can ask exactly
where that note landed. Deterministic, no judge, seconds per arm.

Two numbers matter, and they are different questions:

  rank         where the target note sits in the FULL fused ranking. This is
               retrieval capability, and it moves when fusion or embeddings
               improve even while the answer stays wrong.
  in_context   whether the target survived truncation to keep_k. This is what
               the 3B actually sees. A note at rank 7 with keep_k=6 is invisible
               no matter how good the ranking looks.

Arms are selected by environment variable only (BRAIN_HYBRID, BRAIN_DECOMPOSE,
BRAIN_ANN, BRAIN_INDEX, ...) so every run is reproducible from its own header,
and the header is written into the JSON so a result can never be separated from
the configuration that produced it.

Usage:
  BRAIN_INDEX=index-mac/bm25.pkl BRAIN_AUTOLEARN=0 \
    python3 eval_retrieval.py --gold retrieval_gold.jsonl [--json out.json]
"""

import argparse
import json
import os
import sys
import time

import brain

KS = (1, 3, 5, 8, 10, 20)


def load_gold(path):
    """Read labeled questions, skipping unlabeled ones loudly rather than
    silently scoring them as misses (which would invent a regression)."""
    rows, skipped = [], []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                skipped.append("%s:%d unparseable (%s)" % (path, ln, exc))
                continue
            expect = row.get("expect_sources") or []
            if not expect:
                skipped.append("%s (no expect_sources)" % row.get("id", "?"))
                continue
            rows.append(row)
    return rows, skipped


def source_ranking(order, chunks):
    """Collapse a chunk ranking to a note ranking, keeping each note's best
    position. Retrieval is scored per note because that is the unit a label can
    honestly name: which chunk of a long note holds a fact is an artifact of
    chunk boundaries, not of retrieval quality."""
    seen, out = set(), []
    for idx in order or []:
        src = (chunks[idx] or {}).get("source")
        if src and src not in seen:
            seen.add(src)
            out.append(src)
    return out


def norm(path):
    return (path or "").strip().lstrip("./").lower()


def score_one(row, bm25, chunks, meta):
    q = row["question"]
    expect = {norm(p) for p in row["expect_sources"]}
    t0 = time.perf_counter()
    r = brain.retrieve_context(q, bm25, chunks, meta,
                               persona=row.get("persona", "default"))
    elapsed = time.perf_counter() - t0

    # rank comes from the PRE-diversify ranking. _diversify_order returns at
    # most keep_k ids, so ranking against its output caps every multi-intent
    # question at the window size and makes ranks incomparable between arms.
    ranked = source_ranking(r.get("order_full") or r["order"], chunks)
    keep_k = r["keep_k"]
    in_ctx = {norm(c.get("source")) for c in r["retrieved"]}

    rank = None
    for i, src in enumerate(ranked, 1):
        if norm(src) in expect:
            rank = i
            break
    return {
        "id": row.get("id"),
        "question": q,
        "expect_sources": row["expect_sources"],
        "rank": rank,
        "in_context": bool(expect & in_ctx),
        "keep_k": keep_k,
        "intent": r["intent"],
        "mode": r["mode"],
        "n_subq": r["n_subq"],
        "top_score": round(r["top_score"], 2),
        "retrieved_sources": [c.get("source") for c in r["retrieved"]],
        "seconds": round(elapsed, 3),
    }


def aggregate(results):
    n = len(results)
    if not n:
        return {}
    ranks = [r["rank"] for r in results]
    found = [x for x in ranks if x]
    agg = {
        "n": n,
        "in_context": sum(1 for r in results if r["in_context"]),
        "in_context_pct": round(
            100.0 * sum(1 for r in results if r["in_context"]) / n, 1),
        "mrr": round(sum(1.0 / x for x in found) / n, 4),
        "found_anywhere": len(found),
        "median_rank": (sorted(found)[len(found) // 2] if found else None),
    }
    for k in KS:
        agg["recall@%d" % k] = sum(1 for x in found if x <= k)
    return agg


def env_fingerprint(meta, chunks):
    dense = meta.get("dense") or {}
    # _prep_dense rewrites the block, so the post-load row count lives in "n" /
    # "rows_idx", never in the build-time "ids" key. Reading "ids" here reported
    # 0 dense rows on a fully embedded index.
    return {
        "index": os.environ.get("BRAIN_INDEX", brain.INDEX_PATH),
        "chunks": len(chunks),
        "index_version": meta.get("index_version"),
        "tokenizer_version": meta.get("tokenizer_version"),
        "embed_prefix_version": dense.get("prefix_version"),
        "dense_model": dense.get("model"),
        "dense_dim": dense.get("dim"),
        "dense_rows": dense.get("n") or len(dense.get("rows_idx") or []),
        "BRAIN_HYBRID": os.environ.get("BRAIN_HYBRID", "1"),
        # Default "0" tracks brain.py: decomposition measured WORSE
        # (in_context 31/40 vs 29/40) and was flipped off. A stale default here
        # would mislabel every saved artifact's arm.
        "BRAIN_DECOMPOSE": os.environ.get("BRAIN_DECOMPOSE", "0"),
        # HyDE IS reachable from retrieve_context, unlike the obsidian flags,
        # so this arm genuinely moves the numbers below.
        "BRAIN_HYDE": os.environ.get("BRAIN_HYDE", "0"),
        "BRAIN_TOP_K_FAST": os.environ.get("BRAIN_TOP_K_FAST", "8"),
        "BRAIN_ANN": os.environ.get("BRAIN_ANN", "0"),
        "BRAIN_OBSIDIAN_RANK": os.environ.get("BRAIN_OBSIDIAN_RANK", "0"),
        # Default "0" matches brain.py:249. Recorded for ATTRIBUTION only:
        # neither obsidian flag is reachable from retrieve_context (the
        # expansion runs in _handle_question_impl), so a blocks A/B cannot move
        # a number this harness reports - but two runs must never emit
        # byte-identical headers under different settings.
        "BRAIN_OBSIDIAN_BLOCKS": os.environ.get("BRAIN_OBSIDIAN_BLOCKS", "0"),
        "BRAIN_AUTOLEARN": os.environ.get("BRAIN_AUTOLEARN", "1"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="retrieval_gold.jsonl")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--only", help="comma-separated ids to run")
    ap.add_argument("--quiet", action="store_true",
                    help="aggregate only, no per-question lines")
    args = ap.parse_args()

    rows, skipped = load_gold(args.gold)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r.get("id") in want]
    if not rows:
        sys.stderr.write("no labeled questions to run\n")
        return 2

    chunks, bm25, meta = brain.load_index()
    fp = env_fingerprint(meta, chunks)

    if os.environ.get("BRAIN_AUTOLEARN", "1") not in ("0", "false", "False", "no"):
        sys.stderr.write(
            "WARNING: BRAIN_AUTOLEARN is on; update_index.py can mutate the "
            "index mid-run. Set BRAIN_AUTOLEARN=0.\n")

    print("index: %s | %d chunks | tok=%s | dense=%s dim=%s rows=%d prefix=%s"
          % (fp["index"], fp["chunks"], fp["tokenizer_version"],
             fp["dense_model"], fp["dense_dim"], fp["dense_rows"],
             fp["embed_prefix_version"]))
    print("arms: HYBRID=%s DECOMPOSE=%s HYDE=%s TOP_K_FAST=%s ANN=%s "
          "OBSIDIAN_RANK=%s OBSIDIAN_BLOCKS=%s"
          % (fp["BRAIN_HYBRID"], fp["BRAIN_DECOMPOSE"], fp["BRAIN_HYDE"],
             fp["BRAIN_TOP_K_FAST"], fp["BRAIN_ANN"],
             fp["BRAIN_OBSIDIAN_RANK"], fp["BRAIN_OBSIDIAN_BLOCKS"]))
    if skipped:
        print("skipped %d unlabeled: %s" % (len(skipped), ", ".join(skipped[:6])
                                            + (" ..." if len(skipped) > 6 else "")))
    print("")

    results = []
    for row in rows:
        res = score_one(row, bm25, chunks, meta)
        results.append(res)
        if not args.quiet:
            rank = res["rank"]
            flag = "OK " if res["in_context"] else ("~  " if rank else "MISS")
            print("%-4s %-26s rank=%-5s keep_k=%-2d %-12s %s"
                  % (flag, (res["id"] or "?")[:26],
                     rank if rank else "-", res["keep_k"], res["mode"],
                     "" if res["in_context"] else
                     ("target at %d, window holds %d" % (rank, res["keep_k"])
                      if rank else "target not in fused ranking at all")))

    agg = aggregate(results)
    print("\n%s" % ("-" * 68))
    print("in_context %d/%d (%.1f%%)   MRR %.4f   median rank %s"
          % (agg["in_context"], agg["n"], agg["in_context_pct"], agg["mrr"],
             agg["median_rank"]))
    print("recall@k  " + "  ".join(
        "@%d=%d" % (k, agg["recall@%d" % k]) for k in KS))
    print("found anywhere in ranking: %d/%d" % (agg["found_anywhere"], agg["n"]))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp, "aggregate": agg,
                       "skipped": skipped, "results": results},
                      fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
