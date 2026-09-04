#!/usr/bin/env python3
"""Checks for HyDE retrieval (BRAIN_HYDE).

HyDE adds a THIRD RRF ranker built from a hypothetical answer, for the
vocabulary gap that neither BM25 nor a dense embedding of the question can
cross. Because it depends on a generation call, the property that matters most
is that it is TOTAL: a dead model, a timeout, a refusal or an empty reply must
degrade retrieval to exactly the non-HyDE path, never take the turn down.

No Ollama required -- chat_once is stubbed.
"""

import os
import sys

os.environ.setdefault("BRAIN_INDEX", os.path.join(os.path.dirname(__file__), "index-mac/bm25.pkl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain  # noqa: E402

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if not cond:
        FAILURES.append("%s%s" % (name, (" " + detail) if detail else ""))


Q = "Isn't a school only for deaf kids just holding them back?"

# -- default is ON -----------------------------------------------------------
# It earned the default on measurement: in_context 33/40 -> 36/40, MRR 0.6437
# -> 0.6471, three questions entered the window and none left, replicated
# byte-identical across two runs. It still costs one extra 3B generation per
# question (~48x the retrieval time on Metal), so BRAIN_HYDE=0 buys that back.
check("HyDE defaults on", brain.HYDE_ENABLED,
      os.environ.get("BRAIN_HYDE", "1") not in ("0", "false", "False", "no", ""))

# -- the happy path ----------------------------------------------------------
calls = []


def stub(reply):
    def _chat(model, messages, options=None):
        calls.append({"model": model, "messages": messages, "options": options})
        return reply
    return _chat


_real_chat = brain.chat_once
try:
    brain.chat_once = stub("The Maryland School for the Deaf runs campuses in "
                           "Frederick and Columbia and teaches in ASL and English.")
    calls.clear()
    # Snapshot the global BEFORE the call: the default is env-driven
    # (BRAIN_NUM_PREDICT, 800 since the parent-agent scope-routes work), so
    # the check is "unchanged by HyDE", not "equals a literal".
    num_predict_before = brain.GEN_OPTIONS["num_predict"]
    hyp = brain.hyde_hypothesis(Q)
    check_true("returns the hypothesis", hyp and "Frederick" in hyp)
    check("makes exactly one generation call", len(calls), 1)

    opts = calls[0]["options"] or {}
    # Greedy on purpose: the hypothesis is a retrieval probe, and sampling it
    # would make the RANKING nondeterministic even with BRAIN_SEED pinned on
    # the answer call -- an A/B could then move for reasons unrelated to the
    # change under test.
    check("hypothesis is generated greedily", opts.get("temperature"), 0.0)
    check("hypothesis is length-capped", opts.get("num_predict"),
          brain.HYDE_NUM_PREDICT)
    check_true("cap is short", brain.HYDE_NUM_PREDICT <= 512)
    check_true("does not mutate the global GEN_OPTIONS",
               brain.GEN_OPTIONS.get("temperature") != 0.0
               or "BRAIN_TEMPERATURE" in os.environ)
    check("global num_predict untouched", brain.GEN_OPTIONS["num_predict"],
          num_predict_before)
    check_true("uses the fast model, never the 7B",
               calls[0]["model"] != brain.DEEP_MODEL)
    check_true("system prompt asks for an ANSWER, not a question",
               "ANSWER" in calls[0]["messages"][0]["content"])

    # -- every failure mode returns None, none of them raise ------------------
    # None is the contract: _hyde_scores turns it into "no third ranker", and
    # retrieval proceeds exactly as if HyDE were off.
    for label, reply in (("empty string", ""),
                         ("whitespace only", "   \n  "),
                         ("None reply", None)):
        brain.chat_once = stub(reply)
        check("%s yields None" % label, brain.hyde_hypothesis(Q), None)

    # A refusal must not be embedded: it would sit near notes about not
    # knowing things and actively pull retrieval toward them.
    brain.chat_once = stub("INSUFFICIENT")
    check("a refusal yields None", brain.hyde_hypothesis(Q), None)
    brain.chat_once = stub("I don't have enough information to answer that.")
    check("a paraphrased refusal yields None", brain.hyde_hypothesis(Q), None)

    def _boom(model, messages, options=None):
        raise RuntimeError("ollama is down")

    brain.chat_once = _boom
    check("a raising model yields None", brain.hyde_hypothesis(Q), None)

    # -- _hyde_scores inherits that totality ----------------------------------
    check("_hyde_scores is None when generation fails",
          brain._hyde_scores(Q, [], {}, None), None)

    tm = {}
    brain.chat_once = stub("A hypothesis.")
    # No dense block in meta, so _dense_scores returns None and so must this.
    check("_hyde_scores is None without a dense block",
          brain._hyde_scores(Q, [], {}, tm), None)
    check_true("times itself even when it returns nothing", "hyde" in tm)

    # -- the hypothesis is ANCHORED to the question ---------------------------
    # An unanchored hypothesis that guesses the wrong entity retrieves
    # confidently for that wrong entity. Keeping the question in the embedded
    # text bounds how far the probe can wander from what was asked.
    embedded = []
    _real_dense = brain._dense_scores

    def _capture(text, chunks, meta, **kw):
        embedded.append(text)
        return None

    brain._dense_scores = _capture
    brain.chat_once = stub("Gallaudet University is in Washington DC.")
    try:
        brain._hyde_scores(Q, [], {"dense": {}}, {})
        check_true("embeds the question alongside the hypothesis",
                   embedded and Q in embedded[0],
                   "embedded=%r" % (embedded[:1],))
        check_true("embeds the hypothesis too",
                   embedded and "Gallaudet" in embedded[0])
    finally:
        brain._dense_scores = _real_dense
finally:
    brain.chat_once = _real_chat

# -- the fusion itself -------------------------------------------------------
# Everything above tests the two HyDE functions in isolation. What actually
# ships is the FUSION: HyDE enters _rrf_fuse as a third ranked list, weighted
# identically to BM25 and dense. These checks pin that contract, because the
# measured +3 questions depends on it and nothing else covered it.

# Equal weighting. _rrf_fuse takes no per-ranker weight, so a doc at rank 0 of
# the HyDE list must contribute exactly what a doc at rank 0 of the BM25 list
# does. Two rankers agreeing on a doc must outrank one ranker alone -- that is
# the whole reason a third opinion can rescue gallaudet_1.
fused = brain._rrf_fuse([[7, 1], [7, 2], [3]])
check("agreement outranks any single ranker", fused[0], 7)
check_true("a doc seen once still places", 3 in fused)

one_a = brain._rrf_fuse([[5], []])
one_b = brain._rrf_fuse([[], [5]])
check("rankers are interchangeable", one_a, one_b)

# A third ranker can only ADD evidence. Adding the HyDE list must never drop a
# doc the first two rankers found -- a bad hypothesis costs rank dilution, not
# recall. This is the property the commit message claims and the A/B relied on.
without = brain._rrf_fuse([[1, 2, 3], [2, 3, 4]])
with_hyde = brain._rrf_fuse([[1, 2, 3], [2, 3, 4], [9, 8]])
check_true("HyDE never drops a doc the other rankers found",
           set(without).issubset(set(with_hyde)),
           "without=%r with=%r" % (without, with_hyde))

# RRF_POOL truncation applies to the HyDE list exactly as to the others: a doc
# past the pool depth contributes nothing at all.
deep = brain._rrf_fuse([list(range(brain.RRF_POOL + 5))])
check_true("pool truncates the tail",
           (brain.RRF_POOL + 4) not in deep,
           "pool=%d fused=%d" % (brain.RRF_POOL, len(deep)))
check("pool keeps exactly its depth", len(deep), brain.RRF_POOL)

# The mode string is the only signal in a saved eval artifact that says whether
# HyDE actually fired. eval_retrieval.py records it per question, so if the
# suffix silently stopped appending, every arm would still LOOK distinguishable
# by env var while being identical in fact -- the exact class of bug that made
# the three obsidian_ab_*.json artifacts byte-identical.
#
# Driven through the REAL retrieve_context with the two model-dependent seams
# stubbed (_dense_scores, hyde_hypothesis), so this exercises the shipped
# fusion block rather than a re-description of it. No Ollama.


class _FakeBM25:
    def __init__(self, scores):
        self._scores = scores

    def get_scores(self, _tokens):
        return list(self._scores)


def _retrieve_with(hyde_on, hyp_reply, q_dense=None, hyp_dense=None):
    """Run retrieve_context over a 3-chunk toy index, returning (mode, order).

    q_dense/hyp_dense are the score vectors the two embedding passes return.
    They are keyed on WHICH pass is running rather than matched against the
    text, because the hypothesis is anchored with the question -- so any
    substring match on the embedded text matches the question's key too."""
    chunks = [
        {"id": 0, "text": "deaf education policy", "source": "A.md", "title": "A"},
        {"id": 1, "text": "sign language linguistics", "source": "B.md", "title": "B"},
        {"id": 2, "text": "cochlear implants", "source": "C.md", "title": "C"},
    ]
    meta = {"dense": {"model": "fake", "dim": 3}}
    bm25 = _FakeBM25([3.0, 1.0, 0.0])
    seen = {"n": 0}

    saved = (brain.HYDE_ENABLED, brain._dense_scores, brain.hyde_hypothesis)
    try:
        brain.HYDE_ENABLED = hyde_on

        def _fake_dense(text, _chunks, _meta, **kw):
            # The question is embedded first, the hypothesis second.
            seen["n"] += 1
            if seen["n"] == 1:
                return list(q_dense if q_dense is not None else [0.1, 0.9, 0.2])
            return list(hyp_dense if hyp_dense is not None else [0.1, 0.9, 0.2])

        brain._dense_scores = _fake_dense
        brain.hyde_hypothesis = lambda *a, **kw: hyp_reply
        out = brain.retrieve_context(Q, bm25, chunks, meta)
        return out.get("mode"), out.get("order_full")
    finally:
        brain.HYDE_ENABLED, brain._dense_scores, brain.hyde_hypothesis = saved


mode_off, order_off = _retrieve_with(False, None)
check("mode says hybrid when HyDE is off", mode_off, "hybrid")

mode_on, order_on = _retrieve_with(True, "a hypothetical answer")
check("mode says hybrid+hyde when HyDE fires", mode_on, "hybrid+hyde")

# Totality end-to-end: a failed hypothesis must leave the mode AND the ranking
# byte-identical to the HyDE-off path. This is the property that keeps a dead
# model from silently changing retrieval instead of just not helping.
mode_dead, order_dead = _retrieve_with(True, None)
check("a failed hypothesis reports the non-HyDE mode", mode_dead, "hybrid")
check("a failed hypothesis leaves the ranking untouched", order_dead, order_off)

# And the rescue itself, which is gallaudet_1 in miniature: chunk 2 scores 0.0
# on BM25 (so the lexical pool drops it) and -inf on the question's dense pass
# (so the dense pool drops it too). It is therefore absent from the fused order
# ENTIRELY -- rank None, exactly what gallaudet_1 measured before HyDE. Only a
# hypothesis that embeds toward it can put it back in the ranking.
_UNREACHABLE = [0.1, 0.9, float("-inf")]

_, order_norescue = _retrieve_with(False, None, q_dense=_UNREACHABLE)
check_true("the target is unreachable without HyDE", 2 not in order_norescue,
           "order=%r" % (order_norescue,))

_, order_rescue = _retrieve_with(True, "cochlear implants",
                                 q_dense=_UNREACHABLE,
                                 hyp_dense=[0.0, 0.1, 0.99])
check_true("HyDE rescues a doc neither other ranker can reach",
           2 in order_rescue, "order=%r" % (order_rescue,))

if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_hyde: all checks passed")
