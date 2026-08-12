#!/usr/bin/env python3
"""Offline correctness test for expand_with_obsidian_signal (BRAIN_OBSIDIAN_RANK)
-- brain.py's integration point for obsidian_rank.py. No Ollama, network, or
real vault needed: obsidian_rank.load_note_vectors_from_export is monkey-
patched to return a small synthetic vector set, matching test_hybrid.py's
existing mocking convention in this repo, but the REAL obsidian_rank ranking
logic (expand_from_seeds/nearest_neighbors/cosine) runs unmocked.
"""
import sys
sys.path.insert(0, "/Users/you.lawson/jetson-brain")

import brain
import obsidian_rank

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)

VECTORS = {
    "seed.md": [1.0, 0.0],
    "related.md": [0.9, 0.1],
    "unrelated.md": [0.0, 1.0],
}

def make_meta():
    seed_chunk = {"id": 1, "source": "seed.md", "text": "seed text"}
    related_chunk = {"id": 2, "source": "related.md", "text": "related text"}
    unrelated_chunk = {"id": 3, "source": "unrelated.md", "text": "unrelated text"}
    return {
        "by_source": {"seed.md": [1], "related.md": [2], "unrelated.md": [3]},
        "by_id": {1: seed_chunk, 2: related_chunk, 3: unrelated_chunk},
    }, seed_chunk, related_chunk, unrelated_chunk

# ---- disabled: returns [] without touching obsidian_rank at all -------------
brain.OBSIDIAN_RANK_ENABLED = False
obsidian_rank.load_note_vectors_from_export = lambda: (_ for _ in ()).throw(
    RuntimeError("should not be called when disabled"))
meta, seed_chunk, related_chunk, unrelated_chunk = make_meta()
result = brain.expand_with_obsidian_signal([seed_chunk], [seed_chunk], meta)
check("disabled: returns [] and never calls obsidian_rank", result == [])

# ---- enabled: pulls in related notes ranked by similarity, excludes seed ----
brain.OBSIDIAN_RANK_ENABLED = True
obsidian_rank.load_note_vectors_from_export = lambda: VECTORS
meta, seed_chunk, related_chunk, unrelated_chunk = make_meta()
result = brain.expand_with_obsidian_signal([seed_chunk], [seed_chunk], meta)
result_srcs = [c["source"] for c in result]
check("enabled: pulls in both non-seed notes (small fixture, under max_expand)",
      set(result_srcs) == {"related.md", "unrelated.md"})
check("enabled: orders by similarity (related.md before unrelated.md)",
      result_srcs[0] == "related.md")
check("enabled: never re-adds the seed note itself",
      "seed.md" not in result_srcs)

# ---- dedup: a chunk already present in context_chunks is not re-added ------
context_chunks = [seed_chunk, related_chunk]  # related.md already pulled in (e.g. via wikilinks)
result = brain.expand_with_obsidian_signal([seed_chunk], context_chunks, meta)
result_srcs = [c["source"] for c in result]
check("dedup: skips a note already present in context_chunks",
      "related.md" not in result_srcs)
check("dedup: still returns the note that wasn't already present",
      result_srcs == ["unrelated.md"])

# ---- empty vector export degrades to [] without raising --------------------
obsidian_rank.load_note_vectors_from_export = lambda: {}
result = brain.expand_with_obsidian_signal([seed_chunk], [seed_chunk], meta)
check("empty export: returns [] rather than raising", result == [])

# ---- no seed sources (e.g. retrieved chunks with no 'source' field) --------
obsidian_rank.load_note_vectors_from_export = lambda: VECTORS
result = brain.expand_with_obsidian_signal([{"id": 9, "text": "no source"}],
                                            [{"id": 9, "text": "no source"}], meta)
check("no seed sources: returns [] rather than raising", result == [])

# ---- max_expand cap is respected (OBSIDIAN_MAX_EXPAND=1) -------------------
orig_max = brain.OBSIDIAN_MAX_EXPAND
brain.OBSIDIAN_MAX_EXPAND = 1
obsidian_rank.load_note_vectors_from_export = lambda: VECTORS
result = brain.expand_with_obsidian_signal([seed_chunk], [seed_chunk], meta)
check("respects OBSIDIAN_MAX_EXPAND cap", len(result) == 1)
brain.OBSIDIAN_MAX_EXPAND = orig_max

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
