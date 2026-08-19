#!/usr/bin/env python3
"""Offline correctness test for the semantic web-fallback ranking
(BRAIN_WEB_SEMANTIC, _rank_excerpts_by_relevance, gather_web_context's fetch_cap
switch). No Ollama or network needed -- web_search/fetch_page/_embed are all
monkey-patched, matching test_hybrid.py's existing convention in this repo.
"""
import sys, math
sys.path.insert(0, "/Users/you.lawson/jetson-brain")

import semantic_cache
import brain

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)

def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

# ---- _rank_excerpts_by_relevance: orders by cosine, query ~ closest to "a" ---
candidates = [("url_far", "far"), ("url_close", "close"), ("url_mid", "mid")]
def fake_embed_by_text(text, host, model, timeout):
    vecs = {
        "query": [1.0, 0.0],
        "far": [0.0, 1.0],
        "close": [0.95, 0.05],
        "mid": [0.6, 0.4],
    }
    return vecs.get(text)
semantic_cache._embed = lambda text, host, model, timeout: fake_embed_by_text(
    "query" if text == "the query" else text, host, model, timeout)
ranked = brain._rank_excerpts_by_relevance("the query", candidates)
check("_rank_excerpts_by_relevance reorders by similarity",
      [href for href, _ in ranked] == ["url_close", "url_mid", "url_far"])

# ---- _rank_excerpts_by_relevance: single candidate is a no-op ---------------
one = brain._rank_excerpts_by_relevance("the query", [("only", "close")])
check("_rank_excerpts_by_relevance no-ops on <2 candidates", one == [("only", "close")])

# ---- _rank_excerpts_by_relevance: query embed failure falls back unchanged --
semantic_cache._embed = lambda text, host, model, timeout: None
same_order = brain._rank_excerpts_by_relevance("the query", candidates)
check("_rank_excerpts_by_relevance falls back to DDG order on embed failure",
      same_order == candidates)

# ---- ...and latches the shared kill switch off, so the NEXT query does not --
# pay another RERANK_TIMEOUT to rediscover the same dead embedder. This is the
# same switch _rerank_by_embedding trips on the vault side: an unreachable
# embedder is a property of the box, not of the path that noticed it.
check("_rank_excerpts_by_relevance disables the reranker after an embed failure",
      brain._RERANK_ENABLED is False)
check("...and then no-ops while disabled",
      brain._rank_excerpts_by_relevance("the query", candidates) == candidates)
brain._RERANK_ENABLED = True   # re-arm: the cases below assume a live embedder

# ---- _rank_excerpts_by_relevance: partial embed failure sinks unembeddable --
def flaky_embed(text, host, model, timeout):
    if text == "the query":
        return [1.0, 0.0]
    if text == "close":
        return [0.95, 0.05]
    return None  # "far" and "mid" fail to embed
semantic_cache._embed = flaky_embed
partial = brain._rank_excerpts_by_relevance("the query", candidates)
check("_rank_excerpts_by_relevance ranks the one embeddable candidate first",
      partial[0] == ("url_close", "close"))

# ---- gather_web_context: default (disabled) fetches only MAX_WEB_FETCH ------
fetch_calls = []
def fake_web_search(query):
    return [{"href": f"url{i}", "body": f"snippet{i}"} for i in range(brain.MAX_WEB_RESULTS)]
def fake_fetch_page(url):
    fetch_calls.append(url)
    return f"text for {url}"
brain.web_search = fake_web_search
brain.fetch_page = fake_fetch_page
brain.persist_web_fetch = None  # skip the disk-write side effect for this test
brain.WEB_SEMANTIC_ENABLED = False
fetch_calls.clear()
excerpts = brain.gather_web_context("q")
check("gather_web_context (disabled) returns MAX_WEB_FETCH excerpts",
      len(excerpts) == brain.MAX_WEB_FETCH)
check("gather_web_context (disabled) only fetches MAX_WEB_FETCH pages (unchanged cost)",
      len(fetch_calls) == brain.MAX_WEB_FETCH)

# ---- gather_web_context: enabled fetches all MAX_WEB_RESULTS, then ranks ----
def fake_embed_web(text, host, model, timeout):
    # query "q" -> [1.0, 0.0]; "text for urlN" -- make url3 the closest match.
    if text == "q":
        return [1.0, 0.0]
    idx = int(text[-1])
    return [1.0, 0.0] if idx == 3 else [0.0, float(idx + 1)]
semantic_cache._embed = fake_embed_web
brain._RERANK_ENABLED = True   # re-arm after the failure cases above
brain.WEB_SEMANTIC_ENABLED = True
fetch_calls.clear()
excerpts = brain.gather_web_context("q")
check("gather_web_context (enabled) fetches all MAX_WEB_RESULTS candidates",
      len(fetch_calls) == brain.MAX_WEB_RESULTS)
check("gather_web_context (enabled) still caps output at MAX_WEB_FETCH",
      len(excerpts) == brain.MAX_WEB_FETCH)
check("gather_web_context (enabled) ranks the semantically closest excerpt first",
      excerpts[0][0] == "url3")
brain.WEB_SEMANTIC_ENABLED = False  # restore default for any test run after this one

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
