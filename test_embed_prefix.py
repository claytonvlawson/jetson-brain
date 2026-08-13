#!/usr/bin/env python3
"""Checks for the nomic task-prefix scheme (build_index --embed-prefix).

The failure this guards against is silent. An index embedded with
"search_document: " but queried without "search_query: " does not raise, does
not warn, and does not produce an obviously wrong answer -- it just retrieves
measurably worse, forever. So the pairing is not something the two sides may
decide independently: the index RECORDS its prefix_version and brain.py reads
it back. These checks pin that contract.
"""

import os
import sys

# brain.py exits at import if it cannot find an index; point it at one that
# does not need to exist, since nothing here touches retrieval.
os.environ.setdefault("BRAIN_INDEX", os.path.join(os.path.dirname(__file__), "index-mac/bm25.pkl"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain          # noqa: E402
import build_index    # noqa: E402

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if not cond:
        FAILURES.append("%s%s" % (name, (" " + detail) if detail else ""))


# -- the two tables must agree, in both directions ---------------------------
# build_index owns (document, query) pairs; brain.py keeps a query-only mirror
# so it never has to import the builder at query time. A prefix added to one
# and forgotten in the other is exactly the drift this file exists to catch.
for version, (_doc, query) in build_index.EMBED_PREFIXES.items():
    check("query prefix for %r" % version, brain._QUERY_PREFIXES.get(version), query)

check("no extra versions in brain.py",
      sorted(brain._QUERY_PREFIXES), sorted(build_index.EMBED_PREFIXES))

# -- the default must stay backward-compatible -------------------------------
# Every index built before --embed-prefix existed has no prefix_version key at
# all, and was embedded verbatim. Those indexes must keep working untouched.
check("default is none", build_index.DEFAULT_EMBED_PREFIX, "none")
check("none means no document prefix", build_index.EMBED_PREFIXES["none"][0], "")
check("none means no query prefix", build_index.EMBED_PREFIXES["none"][1], "")

# -- _query_prefix resolution ------------------------------------------------
check("explicit none", brain._query_prefix({"prefix_version": "none"}), "")
check("explicit nomic-v1", brain._query_prefix({"prefix_version": "nomic-v1"}),
      "search_query: ")

# A legacy index (no key), an unrecognized value, an explicit null, and a
# malformed meta must all degrade to the verbatim scheme rather than raising:
# a retrieval path that throws takes the whole answer down, and "no prefix" is
# both the safe default and what a legacy artifact actually is.
check("legacy index, key absent", brain._query_prefix({}), "")
check("unrecognized version", brain._query_prefix({"prefix_version": "future-v9"}), "")
check("explicit null version", brain._query_prefix({"prefix_version": None}), "")
for bad in (None, [], "string", 0):
    try:
        check_true("_query_prefix(%r) returns a str" % (bad,),
                   isinstance(brain._query_prefix(bad), str))
    except Exception as exc:
        FAILURES.append("_query_prefix(%r) raised %r" % (bad, exc))

# -- the nomic scheme is the one nomic-embed-text was trained on -------------
check("nomic document prefix", build_index.EMBED_PREFIXES["nomic-v1"][0],
      "search_document: ")
check("nomic query prefix", build_index.EMBED_PREFIXES["nomic-v1"][1],
      "search_query: ")

# -- build_dense stamps the version it actually used -------------------------
# Verified without touching Ollama by driving build_dense with a stub embedder,
# which also proves the document prefix reaches the embed call and that
# truncation happens BEFORE the prefix is added (so a long chunk can never eat
# the prefix and silently become an unprefixed row).
_seen = []


def _fake_embed_fns():
    def _embed(text, host, model, timeout):
        _seen.append(text)
        return [1.0, 0.0]

    def _normalize(vec):
        return list(vec)

    return _embed, _normalize


_real = build_index._load_embed_fns
build_index._load_embed_fns = _fake_embed_fns
try:
    long_text = "x" * (build_index.EMBED_DOC_CHARS + 500)
    chunks = [{"id": 0, "text": "short chunk"}, {"id": 1, "text": long_text}]

    _seen.clear()
    block = build_index.build_dense(chunks, "nomic-embed-text", "http://h", "nomic-v1")
    check("stamps prefix_version", block.get("prefix_version"), "nomic-v1")
    check_true("document prefix is applied",
               all(t.startswith("search_document: ") for t in _seen),
               "saw %r" % ([t[:24] for t in _seen],))
    check("long chunk keeps full doc budget after prefixing",
          len(_seen[1]), len("search_document: ") + build_index.EMBED_DOC_CHARS)

    _seen.clear()
    block = build_index.build_dense(chunks, "nomic-embed-text", "http://h")
    check("default stamps none", block.get("prefix_version"), "none")
    check_true("no prefix by default",
               not any(t.startswith("search_document: ") for t in _seen))

    # An unknown version must fall back to verbatim rather than crashing a
    # 1500-chunk build 40 minutes in.
    _seen.clear()
    block = build_index.build_dense(chunks, "nomic-embed-text", "http://h", "future-v9")
    check_true("unknown version embeds verbatim",
               not any(t.startswith("search_document: ") for t in _seen))
finally:
    build_index._load_embed_fns = _real

# -- report ------------------------------------------------------------------
if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_embed_prefix: all checks passed")
