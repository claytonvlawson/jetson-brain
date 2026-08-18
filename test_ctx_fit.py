#!/usr/bin/env python3
"""Checks for fit_context(), the prompt-budget guard.

The failure being guarded is silent and severe: when the assembled prompt
exceeds num_ctx, Ollama truncates from the TOP, which drops the system prompt
and the highest-ranked snippets first. The model then answers with no grounding
contract, no citation rule and no INSUFFICIENT sentinel -- and nothing anywhere
reports an error. fit_context trims from the TAIL instead, so an over-budget
turn loses its weakest chunk rather than its contract.
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


def chunk(i, chars):
    return {"id": i, "text": "w" * chars, "source": "N%d.md" % i, "title": "N%d" % i}


Q = "What is language deprivation?"
SYS = "S" * 1000

# -- a prompt that already fits is returned untouched -------------------------
small = [chunk(i, 200) for i in range(3)]
kept, info = brain.fit_context(Q, small, SYS, num_ctx=6144)
check("small prompt keeps every chunk", len(kept), 3)
check("small prompt drops nothing", info["dropped"], 0)
check_true("small prompt reports fit", info["fit"])
check_true("returns the same chunk objects", all(a is b for a, b in zip(kept, small)))

# -- an over-budget prompt loses its TAIL, in order --------------------------
# 20 x 4000 chars is ~23k tokens against a 6144 window: far over.
big = [chunk(i, 4000) for i in range(20)]
kept, info = brain.fit_context(Q, big, SYS, num_ctx=6144)
check_true("over-budget prompt drops chunks", info["dropped"] > 0,
           "dropped=%d" % info["dropped"])
check_true("over-budget prompt ends up fitting", info["fit"],
           "est=%d budget=%d" % (info["est_tokens"], info["budget_tokens"]))
check_true("kept chunks are a PREFIX of the input (tail dropped, not head)",
           kept == big[:len(kept)],
           "kept ids=%r" % ([c["id"] for c in kept],))
check("dropped count matches", info["dropped"], len(big) - len(kept))
check_true("the system prompt survives", info["budget_tokens"] > 0)

# -- the reserve is respected ------------------------------------------------
# Fitting must leave room for the answer, not fill the window to the brim.
check("budget subtracts the reserve",
      info["budget_tokens"], 6144 - brain.CTX_RESERVE_TOKENS)
check_true("estimate lands under budget", info["est_tokens"] <= info["budget_tokens"])

# -- never starve the prompt to nothing --------------------------------------
# One chunk larger than the whole window must still be sent. A question with
# zero context should reach the INSUFFICIENT path honestly rather than be
# silently emptied here, and an empty list must not loop forever.
huge = [chunk(0, 200000)]
kept, info = brain.fit_context(Q, huge, SYS, num_ctx=6144)
check("one oversized chunk is kept", len(kept), 1)
check_true("oversized chunk is reported as NOT fitting", not info["fit"])

kept, info = brain.fit_context(Q, [], SYS, num_ctx=6144)
check("empty context stays empty", kept, [])
check("empty context drops nothing", info["dropped"], 0)

for bad in (None, []):
    try:
        k, _ = brain.fit_context(Q, bad, SYS, num_ctx=6144)
        check_true("fit_context(%r) returns a list" % (bad,), isinstance(k, list))
    except Exception as exc:
        FAILURES.append("fit_context(%r) raised %r" % (bad, exc))

# -- history is charged against the budget -----------------------------------
# A long session was the original overflow route, so history must not be free.
kept_no_hist, info_no_hist = brain.fit_context(Q, big, SYS, num_ctx=6144)
kept_hist, info_hist = brain.fit_context(Q, big, SYS + "H" * 8000, num_ctx=6144)
check_true("a longer preamble keeps fewer chunks",
           len(kept_hist) < len(kept_no_hist),
           "%d vs %d" % (len(kept_hist), len(kept_no_hist)))

# -- the guard can be turned off ---------------------------------------------
# BRAIN_CTX_FIT=0 must restore the exact pre-guard behavior for a clean A/B.
_saved = brain.CTX_FIT_ENABLED
brain.CTX_FIT_ENABLED = False
try:
    kept, info = brain.fit_context(Q, big, SYS, num_ctx=6144)
    check("disabled keeps every chunk", len(kept), len(big))
    check("disabled drops nothing", info["dropped"], 0)
    check_true("disabled still reports an estimate", info["est_tokens"] > 0)
finally:
    brain.CTX_FIT_ENABLED = _saved

# -- the estimator is monotone and non-negative ------------------------------
check("empty text estimates zero", brain._estimate_tokens(""), 0)
check("None estimates zero", brain._estimate_tokens(None), 0)
check_true("longer text estimates more",
           brain._estimate_tokens("x" * 1000) > brain._estimate_tokens("x" * 100))
# Pessimistic by design: assuming FEWER chars per token overestimates the count,
# which is the safe direction. Overestimating costs one dropped tail chunk;
# underestimating costs a silently ungrounded answer.
#
# The bound is the MEASURED MINIMUM, not a round number. Against Ollama's own
# prompt_eval_count over the 40 retrieval_gold prompts the true ratio was
# 4.25 min / 4.61 mean / 4.77 max, so any value strictly below 4.25 still
# over-counts every prompt observed. The old bound of 3.6 predates that
# measurement and cost ~35% of the budget for nothing. Keep the strict
# inequality: at exactly 4.25 the estimator stops being conservative on the
# tightest prompt in the set.
MEASURED_MIN_CHARS_PER_TOKEN = 4.25
check_true("chars-per-token stays under the measured minimum",
           brain.CHARS_PER_TOKEN < MEASURED_MIN_CHARS_PER_TOKEN,
           "CHARS_PER_TOKEN=%r must be < %r" % (brain.CHARS_PER_TOKEN,
                                                MEASURED_MIN_CHARS_PER_TOKEN))

if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_ctx_fit: all checks passed")
