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

# -- default is OFF ----------------------------------------------------------
# It costs one extra 3B generation per question, the dominant latency on an
# 8GB Orin, so it must earn the default on measurement.
check("HyDE defaults off", brain.HYDE_ENABLED,
      os.environ.get("BRAIN_HYDE", "0") not in ("0", "false", "False", "no"))

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
    check("global num_predict untouched", brain.GEN_OPTIONS["num_predict"], 640)
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

if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_hyde: all checks passed")
