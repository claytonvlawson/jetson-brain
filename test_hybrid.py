#!/usr/bin/env python3
"""Offline correctness test for the hybrid RRF retrieval plumbing.

Exercises brain._prep_dense / _dense_scores / _rrf_fuse and build_index.build_dense
with a fake embedder, so no Ollama or Jetson index is needed. Verifies the one
thing that would silently corrupt retrieval if wrong: dense-row -> chunk-index
alignment, including a chunk that has NO dense vector (must be -inf, never a wrong
neighbor's score).
"""
import sys, math, array
sys.path.insert(0, "/Users/you.lawson/jetson-brain")

import semantic_cache
import brain
import build_index

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)

# ---- fixtures: 4 chunks (indices 0..3), dense vectors for ids 0,1,3 only -----
DIM = 4
chunks = [{"id": 0, "text": "a"}, {"id": 1, "text": "b"},
          {"id": 2, "text": "c"}, {"id": 3, "text": "d"}]  # id 2 = dense-invisible
def norm(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
rows = {0: norm([1,0,0,0]), 1: norm([0,1,0,0]), 3: norm([0,0,1,0])}
buf = array.array("f")
ids = [0, 1, 3]
for cid in ids:
    buf.extend(rows[cid])
dense_block = {"model": "fake-embed", "dim": DIM, "ids": ids, "buf": buf}

# ---- _prep_dense -------------------------------------------------------------
dm = brain._prep_dense(dense_block, chunks)
check("_prep_dense returns a block", dm is not None)
check("_prep_dense n == 3", dm and dm["n"] == 3)
# rows_idx maps dense row -> chunk index; here id==index so [0,1,3]
check("_prep_dense rows_idx aligns id->index", dm and dm["rows_idx"] == [0, 1, 3])
check("_prep_dense built numpy matrix", dm and dm["matrix"] is not None)
check("_prep_dense rejects short buffer",
      brain._prep_dense({**dense_block, "buf": array.array("f", [1,2,3])}, chunks) is None)
check("_prep_dense rejects non-dict", brain._prep_dense(None, chunks) is None)

# ---- _dense_scores with a fake query embedder (query ~ closest to id0) --------
# query [0.9,0.1,0,0] -> cos: id0 highest, id1 next, id3 ~0, id2 = -inf (no vec)
def fake_embed(text, host, model, timeout):
    return [0.9, 0.1, 0.0, 0.0]
semantic_cache._embed = fake_embed  # _dense_scores does `from semantic_cache import _embed` at call time

meta_np = {"dense": dm}
s = brain._dense_scores("does not matter", chunks, meta_np)
check("_dense_scores returns list len==4", isinstance(s, list) and len(s) == 4)
check("_dense_scores id2 is -inf (dense-invisible, not a wrong score)", s[2] == float("-inf"))
check("_dense_scores ranks id0 > id1 > id3", s[0] > s[1] > s[3])

# ---- pure-Python path must agree with numpy path -----------------------------
dm_py = dict(dm); dm_py["matrix"] = None; dm_py["buf"] = buf
s_py = brain._dense_scores("q", chunks, {"dense": dm_py})
agree = all(
    (a == b == float("-inf")) or (a != float("-inf") and b != float("-inf") and abs(a-b) < 1e-5)
    for a, b in zip(s, s_py))
check("pure-python dense scores match numpy path", agree)

# ---- _dense_scores with a NaN/inf query vector -------------------------------
# Ollama can return a non-finite embedding (nomic-embed-text under concurrent
# load). Fed to the matmul that ranks the dense arm by NaN with only a numpy
# RuntimeWarning to show for it. The scorer must treat it as a failed embed
# (None -> BM25 carries the query), and must not emit the warning.
import warnings
import semantic_cache
_real_embed = semantic_cache._embed
try:
    semantic_cache._embed = lambda text, host, model, timeout: [float("nan")] * DIM
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            s_nan = brain._dense_scores("q", chunks, {"dense": dm})
            warned = False
        except RuntimeWarning:
            s_nan, warned = "warned", True
    check("_dense_scores NaN query -> None (BM25 carries it)", s_nan is None)
    check("_dense_scores NaN query raises no RuntimeWarning", not warned)
    semantic_cache._embed = lambda text, host, model, timeout: [float("inf")] + [0.0] * (DIM - 1)
    check("_dense_scores inf query -> None", brain._dense_scores("q", chunks, {"dense": dm}) is None)
finally:
    semantic_cache._embed = _real_embed

# ---- _rrf_fuse ---------------------------------------------------------------
# bm25 likes 3 then 1; dense likes 0 then 1. id1 agreed by both -> should win.
fused = brain._rrf_fuse([[3, 1], [0, 1]])
check("_rrf_fuse rewards agreement (id1 ranked first)", fused[0] == 1)
check("_rrf_fuse includes all union docs", set(fused) == {0, 1, 3})
# a doc only dense (0) and a doc only bm25 (3) both survive -> recall win
check("_rrf_fuse keeps dense-only doc 0", 0 in fused)
check("_rrf_fuse keeps bm25-only doc 3", 3 in fused)

# ---- build_index.build_dense assembly (fake embedder, one failing chunk) -----
def fake_embed2(text, host, model, timeout):
    return None if text == "c" else norm([1.0, float(len(text)), 0.0, 0.0])
build_index._load_embed_fns = lambda: (fake_embed2, norm)
db = build_index.build_dense(chunks, "fake-embed", "http://x")
check("build_dense returns a block", db is not None)
check("build_dense skips the failing chunk (3 of 4 embedded)", db and db["ids"] == [0, 1, 3])
check("build_dense buffer length == n*dim", db and len(db["buf"]) == 3 * DIM)
check("build_dense dim detected", db and db["dim"] == DIM)
# round-trip: prep the freshly built block and confirm id2 still dense-invisible
dm2 = brain._prep_dense(db, chunks)
check("round-trip rows_idx == [0,1,3]", dm2 and dm2["rows_idx"] == [0, 1, 3])

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
