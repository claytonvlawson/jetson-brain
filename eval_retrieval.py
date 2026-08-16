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

# Personas retrieve_context() actually understands. row["persona"] is passed
# straight through with no validation, so a typo or a half-wired persona runs
# clean and emits default-identical numbers under a header that looks correct.
# Warn loudly, never hard-fail: a genuinely new persona must still be runnable
# here before it is whitelisted.
KNOWN_PERSONAS = ("default", "parent", "scholar")

# The language-deprivation corpus, by source-path prefix. Breadth over this
# subset is the thing the diversification work moves, so it lives in one
# greppable place instead of being spelled out at the call site. Values are
# pre-normalized through norm() at use time, so write them as they appear in
# the vault. Gulati 2018 alone is 58 of the 85 chunks under
# "Sources/Language Deprivation/", which is why a single source can dominate
# an entire retrieval window without any single-source metric noticing.
LD_SOURCE_PREFIXES = (
    "Sources/Language Deprivation/",
    "LanguageDeprivationAgent/Core/",
    "Concepts/Language Deprivation",
    "People/Wyatte C. Hall",
)


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


_LD_PREFIXES_NORM = tuple(norm(p) for p in LD_SOURCE_PREFIXES)


def is_ld_source(src):
    """True when a source path sits under the language-deprivation corpus."""
    return norm(src).startswith(_LD_PREFIXES_NORM)


def score_one(row, bm25, chunks, meta):
    q = row["question"]
    expect = {norm(p) for p in row["expect_sources"]}
    persona = row.get("persona", "default")
    if persona not in KNOWN_PERSONAS:
        sys.stderr.write(
            "!!! WARNING: row %r requests persona %r, which is not one of %s. "
            "retrieve_context() does NOT validate persona, so this run will "
            "silently behave like 'default' and its numbers are NOT evidence "
            "that %r works.\n"
            % (row.get("id", "?"), persona, "/".join(KNOWN_PERSONAS), persona))
        sys.stderr.flush()
    t0 = time.perf_counter()
    r = brain.retrieve_context(q, bm25, chunks, meta, persona=persona)
    elapsed = time.perf_counter() - t0

    # rank comes from the PRE-diversify ranking. _diversify_order returns at
    # most keep_k ids, so ranking against its output caps every multi-intent
    # question at the window size and makes ranks incomparable between arms.
    ranked = source_ranking(r.get("order_full") or r["order"], chunks)
    keep_k = r["keep_k"]
    in_ctx = {norm(c.get("source")) for c in r["retrieved"]}

    # Post-expansion in_context. retrieve_context() stops at keep_k, but
    # _handle_question_impl keeps going: expand_with_links and (when
    # BRAIN_OBSIDIAN_RANK is on) expand_with_obsidian_signal both run
    # AFTER retrieval and add chunks the model actually sees, uncapped by
    # keep_k. Scoring only r["retrieved"] is why the obsidian flags were
    # historically invisible to this harness -- see env_fingerprint's
    # BRAIN_OBSIDIAN_RANK/BLOCKS comments, now corrected. Mirrors
    # brain.py:2864-2894 exactly so this harness measures what production
    # sends to the model, not a second implementation of it.
    # Upper bound, honestly: production runs fit_context() after this, which
    # drops chunks from the TAIL until the prompt fits num_ctx - so a note that
    # only arrived via expansion is exactly the kind that can still be clipped
    # before the model sees it. in_context_expanded says "retrieval put it in
    # reach", not "the model read it". Closing that last gap needs the token
    # budget, which needs the system prompt and history this harness does not
    # build; eval_brain.py is where that question gets settled.
    added, _linked_ids = brain.expand_with_links(r["retrieved"], r["scores"], meta)
    context_chunks = r["retrieved"] + added
    if brain.OBSIDIAN_RANK_ENABLED:
        obsidian_added = brain.expand_with_obsidian_signal(
            r["retrieved"], context_chunks, meta)
        context_chunks = context_chunks + obsidian_added
    in_ctx_expanded = {norm(c.get("source")) for c in context_chunks}

    # Breadth metrics, all label-free except the last. These exist because
    # in_context is an OR over expect_sources and rank reads only the FIRST
    # match, so both are already pinned at ceiling on the multi-source
    # language-deprivation rows and cannot see a diversification change at all.
    #
    # Computed off the PRE-expansion retrieved set (r["retrieved"], the same set
    # in_ctx is built from), not the post-expansion one, because that is the set
    # retrieve_context/_diversify_order actually controls -- link and obsidian
    # expansion run afterwards and would add breadth that no retrieval arm is
    # responsible for. n_distinct_sources_expanded is emitted alongside as the
    # cheap post-expansion sibling of in_context_expanded.
    per_source = {}
    for c in r["retrieved"]:
        s = norm(c.get("source"))
        if s:
            per_source[s] = per_source.get(s, 0) + 1

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
        "in_context_expanded": bool(expect & in_ctx_expanded),
        "n_distinct_sources": len(per_source),
        "n_distinct_sources_expanded": len({s for s in in_ctx_expanded if s}),
        "n_ld_sources": sum(1 for s in per_source if is_ld_source(s)),
        # THE key metric: how many retrieved chunks come from the single most
        # dominant source. Falling is the evidence diversification worked.
        "max_chunks_one_source": (max(per_source.values()) if per_source else 0),
        # The COUNT of covered expect_sources, next to in_context's boolean OR.
        "n_expected_covered": len(expect & in_ctx),
        "n_expected": len(expect),
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
        "in_context_expanded": sum(1 for r in results if r["in_context_expanded"]),
        "in_context_expanded_pct": round(
            100.0 * sum(1 for r in results if r["in_context_expanded"]) / n, 1),
        "mrr": round(sum(1.0 / x for x in found) / n, 4),
        "found_anywhere": len(found),
        "median_rank": (sorted(found)[len(found) // 2] if found else None),
    }
    # Breadth means. Reported as means (not counts) so two arms with different
    # question sets are still directly comparable, and so a drop in
    # mean_max_chunks_one_source reads as "the window stopped being one note".
    for key in ("n_distinct_sources", "n_distinct_sources_expanded",
                "n_ld_sources", "max_chunks_one_source", "n_expected_covered"):
        agg["mean_%s" % key] = round(
            sum(r.get(key) or 0 for r in results) / float(n), 3)
    for k in KS:
        agg["recall@%d" % k] = sum(1 for x in found if x <= k)
    return agg


def _flag(value):
    """Render a brain.py boolean as the "1"/"0" the fingerprint has always used,
    so artifacts stay comparable across the switch to reading brain directly."""
    return "1" if value else "0"


def env_fingerprint(meta, chunks, rows=None):
    dense = meta.get("dense") or {}
    # Personas actually exercised by this run, read off the gold rows rather
    # than off the environment. The header recorded 16 keys and NEITHER of the
    # two things that most easily make two runs differ invisibly: retrieve_context
    # takes persona per row with no whitelist, so a misspelled persona produced
    # default-identical numbers under a byte-identical header. (retrieval_gold.jsonl
    # carries zero persona keys today, so this reads "default" until rows add one.)
    personas = sorted({(r.get("persona") or "default") for r in (rows or [])}) \
        or ["default"]
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
        # Arms are read back off the BRAIN MODULE, not re-derived from the
        # environment with a default restated here. Restating them is how an
        # artifact silently mislabels its own arm: brain.py's BRAIN_DECOMPOSE
        # default flipped 1->0 and BRAIN_HYDE flipped 0->1, and any artifact
        # saved against a hardcoded copy of the old default records an arm the
        # run never actually used. brain's constants are resolved from the
        # environment at import time, so these are what the code WILL DO --
        # which is the only thing worth fingerprinting.
        "BRAIN_HYBRID": _flag(brain.HYBRID_ENABLED),
        "BRAIN_DECOMPOSE": _flag(brain.DECOMPOSE_ENABLED),
        # HyDE IS reachable from retrieve_context, unlike the obsidian flags,
        # so this arm genuinely moves the numbers below.
        "BRAIN_HYDE": _flag(brain.HYDE_ENABLED),
        "BRAIN_TOP_K_FAST": str(brain.TOP_K_FAST),
        "BRAIN_ANN": _flag(brain.ANN_ENABLED),
        "BRAIN_OBSIDIAN_RANK": _flag(brain.OBSIDIAN_RANK_ENABLED),
        # Both obsidian flags DO move
        # in_context_expanded now (score_one replays expand_with_links +
        # expand_with_obsidian_signal after retrieve_context, same as
        # _handle_question_impl). BLOCKS still can't move plain in_context or
        # in_context_expanded, though: both are note-level ("is note X's path
        # anywhere in context"), and blocks only ever change WHICH chunk of an
        # already-included note gets pulled in, never whether the note itself
        # is present. That needs eval_brain.py's judged answer quality, not
        # this harness. Recorded here for attribution so two runs never emit
        # byte-identical headers under different settings.
        "BRAIN_OBSIDIAN_BLOCKS": _flag(brain.OBSIDIAN_BLOCKS_ENABLED),
        "BRAIN_AUTOLEARN": _flag(brain.AUTOLEARN_ENABLED),
        # Personas exercised by this run's gold rows, so a persona arm can never
        # again be mistaken for a default arm from the header alone.
        "persona": ",".join(personas),
        # Two different seeds, both previously unrecorded. BRAIN_SEED pins the
        # sampler and so does not touch this harness's numbers (no LLM in the
        # loop) but does pin the eval_brain.py runs an artifact gets compared
        # against; BRAIN_ANN_SEED pins _build_ivf's k-means and therefore CAN
        # move retrieval whenever BRAIN_ANN=1. Read off the brain module for the
        # same reason the arms above are: it is what the code WILL DO.
        "seed": (str(brain.GEN_OPTIONS["seed"])
                 if "seed" in brain.GEN_OPTIONS else None),
        "BRAIN_ANN_SEED": str(brain.ANN_SEED),
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
    fp = env_fingerprint(meta, chunks, rows)

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
    print("persona=%s seed=%s ann_seed=%s"
          % (fp["persona"], fp["seed"], fp["BRAIN_ANN_SEED"]))
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
    print("in_context %d/%d (%.1f%%)   in_context_expanded %d/%d (%.1f%%)   "
          "MRR %.4f   median rank %s"
          % (agg["in_context"], agg["n"], agg["in_context_pct"],
             agg["in_context_expanded"], agg["n"], agg["in_context_expanded_pct"],
             agg["mrr"], agg["median_rank"]))
    print("recall@k  " + "  ".join(
        "@%d=%d" % (k, agg["recall@%d" % k]) for k in KS))
    print("found anywhere in ranking: %d/%d" % (agg["found_anywhere"], agg["n"]))
    print("breadth   sources/q=%.2f (expanded %.2f)  LD sources/q=%.2f  "
          "max chunks from one source=%.2f  expected covered/q=%.2f"
          % (agg["mean_n_distinct_sources"],
             agg["mean_n_distinct_sources_expanded"],
             agg["mean_n_ld_sources"],
             agg["mean_max_chunks_one_source"],
             agg["mean_n_expected_covered"]))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp, "aggregate": agg,
                       "skipped": skipped, "results": results},
                      fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
