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
    # Top-level Sources/ notes: the folder scope cannot tell these apart, the
    # tag scope can.
    {"id": 6, "source": "Sources/Albanie 2020 - BSL-1K.md",
     "text": "co-articulated sign recognition", "title": "Albanie 2020",
     "tags": ["source", "paper", "computer-vision", "sign-language-recognition"]},
    {"id": 7, "source": "Sources/Hall 2017 - Language Deprivation Syndrome.md",
     "text": "language deprivation syndrome", "title": "Hall 2017",
     "tags": ["source", "paper", "language-deprivation"]},
    {"id": 8, "source": "International-Law/Northern Ireland Sign Language Bill 2026.md",
     "text": "legal recognition bill", "title": "NI Bill 2026",
     "tags": ["legal", "statute", "sign-language-recognition"]},
]
Q = brain.query_tokens("What is language deprivation?")

exc = brain._default_scope_excluded(CHUNKS, Q, "default")
check("default persona excludes Blockchain, Research, SLR-AI, README, cv-tagged",
      exc == {1, 2, 3, 5, 6})
check("default persona keeps Core and Sources/Language Deprivation",
      0 not in exc and 4 not in exc)
check("tag scope keeps the LD-tagged and the legal SLR-tagged Sources notes",
      7 not in exc and 8 not in exc)
check("parent persona is untouched",
      brain._default_scope_excluded(CHUNKS, Q, "parent") == set())
check("scholar persona is untouched",
      brain._default_scope_excluded(CHUNKS, Q, "scholar") == set())

# Yields to a query that names the folder.
Qb = brain.query_tokens("What does the blockchain reentrancy note say?")
exc_b = brain._default_scope_excluded(CHUNKS, Qb, "default")
check("a query naming Blockchain keeps that folder", 1 not in exc_b)
check("... but still excludes the others", exc_b == {2, 3, 5, 6})
Qc = brain.query_tokens("Which computer vision papers cover sign recognition?")
exc_c = brain._default_scope_excluded(CHUNKS, Qc, "default")
check("a query naming the computer-vision tag keeps that tag", 6 not in exc_c)
check("... but still excludes the folders", exc_c == {1, 2, 3, 5})

# Mask semantics.
bm = brain._mask_scores([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 6.0, 2.5, 2.2], exc, 0.0)
check("BM25 mask zeroes excluded ids",
      bm == [5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.5, 2.2])
ds = brain._mask_scores([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.95, 0.3, 0.2], exc, float("-inf"))
check("dense mask sinks excluded ids to -inf",
      ds[1] == float("-inf") and ds[3] == float("-inf") and ds[6] == float("-inf")
      and ds[0] == 0.9 and ds[7] == 0.3)
same = [1.0, 2.0]
check("empty exclusion returns the same object",
      brain._mask_scores(same, set(), 0.0) is same)

# Off switch and override are read at import; exercise them through the
# globals the helper consults, then restore.
_saved = (brain.DEFAULT_SCOPE_ENABLED, brain._DEFAULT_EXCLUDE_PREFIXES,
          brain._DEFAULT_EXCLUDE_TAGS)
try:
    brain.DEFAULT_SCOPE_ENABLED = False
    check("BRAIN_DEFAULT_SCOPE=0 disables the mask (folders and tags)",
          brain._default_scope_excluded(CHUNKS, Q, "default") == set())
    brain.DEFAULT_SCOPE_ENABLED = True
    brain._DEFAULT_EXCLUDE_PREFIXES = ("Research/",)
    check("BRAIN_DEFAULT_EXCLUDE replaces the folder list",
          brain._default_scope_excluded(CHUNKS, Q, "default") == {2, 6})
    brain._DEFAULT_EXCLUDE_TAGS = ("legal",)
    check("BRAIN_DEFAULT_EXCLUDE_TAGS replaces the tag list",
          brain._default_scope_excluded(CHUNKS, Q, "default") == {2, 8})
    brain._DEFAULT_EXCLUDE_TAGS = ("language-deprivation",)
    check("a tag whose words are in the query yields (LD question keeps LD-tagged notes)",
          brain._default_scope_excluded(CHUNKS, Q, "default") == {2})
    brain._DEFAULT_EXCLUDE_TAGS = ()
    check("empty tag list leaves the folder scope alone",
          brain._default_scope_excluded(CHUNKS, Q, "default") == {2})
    brain._DEFAULT_EXCLUDE_PREFIXES = ()
    check("both lists empty returns an empty set",
          brain._default_scope_excluded(CHUNKS, Q, "default") == set())
finally:
    (brain.DEFAULT_SCOPE_ENABLED, brain._DEFAULT_EXCLUDE_PREFIXES,
     brain._DEFAULT_EXCLUDE_TAGS) = _saved

# Integration: retrieve_context on a fake BM25 with hybrid off and the
# embedding rerank stubbed to identity. The Blockchain chunk scores highest
# on BM25 and must still not be retrieved for the default persona.
class _FakeBM25:
    def get_scores(self, toks):
        return [3.0, 9.0, 1.0, 2.0, 4.0, 0.5, 9.5, 3.5, 0.2]


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
    check("retrieve_context (default) drops the cv-tagged Sources chunk, keeps Hall 2017",
          6 not in got and 7 in got)
    check("mode tag carries +scope", "+scope" in d["mode"])
    d2 = brain.retrieve_context("What is language deprivation?", _FakeBM25(),
                                CHUNKS, {}, persona="scholar")
    got2 = [c["id"] for c in d2["retrieved"]]
    check("retrieve_context (scholar) still sees it",
          1 in got2 and 6 in got2)
finally:
    brain.HYBRID_ENABLED, brain._rerank_by_embedding, brain.HYDE_ENABLED = _saved_hy, _saved_rr, _saved_hyde

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
