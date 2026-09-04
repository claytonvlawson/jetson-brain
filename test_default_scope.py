"""Offline checks for the default-persona folder scope (no Ollama, no index).

The scope is a mask over chunk ids: default persona only, off-topic folders
score 0 on BM25 and -inf on the cosine rankers, and it yields to a query that
names the folder. Run: python3 test_default_scope.py
"""
import os
import sys

os.environ.setdefault("BRAIN_AUTOLEARN", "0")
import brain  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


CHUNKS = [
    {"id": 0, "source": "LanguageDeprivationAgent/Core/What Is Language Deprivation.md",
     "text": "language deprivation", "title": "What Is Language Deprivation"},
    {"id": 1, "source": "Blockchain/Security/Reentrancy.md",
     "text": "reentrancy attack", "title": "Reentrancy"},
    {"id": 2, "source": "Research/JetFuel/2026-08-01.md",
     "text": "jetfuel report", "title": "2026-08-01"},
    {"id": 3, "source": "Sources/Sign Language AI and Ethics/Bohacek Hruz 2022.md",
     "text": "pose transformer", "title": "Bohacek Hruz 2022"},
    {"id": 4, "source": "Sources/Language Deprivation/Gulati 2018.md",
     "text": "language deprivation syndrome", "title": "Gulati 2018"},
    {"id": 5, "source": "README.md", "text": "readme", "title": "README"},
]
Q = brain.query_tokens("What is language deprivation?")

exc = brain._default_scope_excluded(CHUNKS, Q, "default")
check("default persona excludes Blockchain, Research, SLR-AI, README",
      exc == {1, 2, 3, 5})
check("default persona keeps Core and Sources/Language Deprivation",
      0 not in exc and 4 not in exc)
check("parent persona is untouched",
      brain._default_scope_excluded(CHUNKS, Q, "parent") == set())
check("scholar persona is untouched",
      brain._default_scope_excluded(CHUNKS, Q, "scholar") == set())

# Yields to a query that names the folder.
Qb = brain.query_tokens("What does the blockchain reentrancy note say?")
exc_b = brain._default_scope_excluded(CHUNKS, Qb, "default")
check("a query naming Blockchain keeps that folder", 1 not in exc_b)
check("... but still excludes the others", exc_b == {2, 3, 5})

# Mask semantics.
bm = brain._mask_scores([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], exc, 0.0)
check("BM25 mask zeroes excluded ids", bm == [5.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ds = brain._mask_scores([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], exc, float("-inf"))
check("dense mask sinks excluded ids to -inf",
      ds[1] == float("-inf") and ds[3] == float("-inf") and ds[0] == 0.9)
same = [1.0, 2.0]
check("empty exclusion returns the same object",
      brain._mask_scores(same, set(), 0.0) is same)

# Off switch and override are read at import; exercise them through the
# globals the helper consults, then restore.
_saved = (brain.DEFAULT_SCOPE_ENABLED, brain._DEFAULT_EXCLUDE_PREFIXES)
try:
    brain.DEFAULT_SCOPE_ENABLED = False
    check("BRAIN_DEFAULT_SCOPE=0 disables the mask",
          brain._default_scope_excluded(CHUNKS, Q, "default") == set())
    brain.DEFAULT_SCOPE_ENABLED = True
    brain._DEFAULT_EXCLUDE_PREFIXES = ("Research/",)
    check("BRAIN_DEFAULT_EXCLUDE replaces the list",
          brain._default_scope_excluded(CHUNKS, Q, "default") == {2})
finally:
    brain.DEFAULT_SCOPE_ENABLED, brain._DEFAULT_EXCLUDE_PREFIXES = _saved

# Integration: retrieve_context on a fake BM25 with hybrid off and the
# embedding rerank stubbed to identity. The Blockchain chunk scores highest
# on BM25 and must still not be retrieved for the default persona.
class _FakeBM25:
    def get_scores(self, toks):
        return [3.0, 9.0, 1.0, 2.0, 4.0, 0.5]


_saved_hy, _saved_rr, _saved_hyde = brain.HYBRID_ENABLED, brain._rerank_by_embedding, brain.HYDE_ENABLED
try:
    brain.HYBRID_ENABLED = False
    brain.HYDE_ENABLED = False
    brain._rerank_by_embedding = lambda q, chunks, ids, **kw: list(ids)
    d = brain.retrieve_context("What is language deprivation?", _FakeBM25(),
                               CHUNKS, {}, persona="default")
    got = [c["id"] for c in d["retrieved"]]
    check("retrieve_context (default) drops the top-scoring Blockchain chunk",
          1 not in got and 4 in got and 0 in got)
    check("mode tag carries +scope", "+scope" in d["mode"])
    d2 = brain.retrieve_context("What is language deprivation?", _FakeBM25(),
                                CHUNKS, {}, persona="scholar")
    check("retrieve_context (scholar) still sees it",
          1 in [c["id"] for c in d2["retrieved"]])
finally:
    brain.HYBRID_ENABLED, brain._rerank_by_embedding, brain.HYDE_ENABLED = _saved_hy, _saved_rr, _saved_hyde

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
