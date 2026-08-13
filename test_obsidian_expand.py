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

# ---- block-aware chunk pick (BRAIN_OBSIDIAN_BLOCKS) ------------------------
# The note-level test above only proves WHICH notes get expanded. This proves
# WHICH CHUNK of an expanded note is taken: ids[0] by default, and the
# best-matching block's chunk when the block flag is on. Same monkeypatch
# convention; the real rank_note_blocks/best_chunk_for_block logic runs.
def make_multichunk_meta():
    """related.md has two chunks. Chunk 20 is the note's first (what ids[0]
    picks); chunk 21 is the one a query about funding actually wants."""
    seed_chunk = {"id": 10, "source": "seed.md", "text": "seed text"}
    intro = {"id": 20, "source": "related.md", "text": "Overview\nrelated text"}
    funding = {"id": 21, "source": "related.md",
               "text": "Funding history\nthe money details"}
    return {
        "by_source": {"seed.md": [10], "related.md": [20, 21]},
        "by_id": {10: seed_chunk, 20: intro, 21: funding},
    }, seed_chunk

BLOCK_VECTORS = {
    # Near-orthogonal to the seed vector [1.0, 0.0], so it loses.
    "related.md#Overview#{1}": [0.0, 1.0],
    # Aligned with the seed vector, so this block wins and its heading tokens
    # map onto chunk 21.
    "related.md#Funding history#{1}": [1.0, 0.05],
}

obsidian_rank.load_note_vectors_from_export = lambda: VECTORS
obsidian_rank.load_block_vectors_from_export = lambda: (_ for _ in ()).throw(
    RuntimeError("should not be called when BRAIN_OBSIDIAN_BLOCKS is off"))
brain.OBSIDIAN_BLOCKS_ENABLED = False
meta2, seed_chunk2 = make_multichunk_meta()
result = brain.expand_with_obsidian_signal([seed_chunk2], [seed_chunk2], meta2)
check("blocks off: takes the note's first chunk (ids[0]) and never loads blocks",
      [c["id"] for c in result if c["source"] == "related.md"] == [20])

brain.OBSIDIAN_BLOCKS_ENABLED = True
obsidian_rank.load_block_vectors_from_export = lambda: BLOCK_VECTORS
meta2, seed_chunk2 = make_multichunk_meta()
result = brain.expand_with_obsidian_signal([seed_chunk2], [seed_chunk2], meta2)
check("blocks on: takes the best-matching block's chunk, not ids[0]",
      [c["id"] for c in result if c["source"] == "related.md"] == [21])

obsidian_rank.load_block_vectors_from_export = lambda: {}
meta2, seed_chunk2 = make_multichunk_meta()
result = brain.expand_with_obsidian_signal([seed_chunk2], [seed_chunk2], meta2)
check("blocks on but export has no blocks: falls back to ids[0]",
      [c["id"] for c in result if c["source"] == "related.md"] == [20])

obsidian_rank.load_block_vectors_from_export = lambda: (_ for _ in ()).throw(
    RuntimeError("block export unreadable"))
meta2, seed_chunk2 = make_multichunk_meta()
result = brain.expand_with_obsidian_signal([seed_chunk2], [seed_chunk2], meta2)
check("blocks on but the loader raises: degrades, never propagates",
      isinstance(result, list))
brain.OBSIDIAN_BLOCKS_ENABLED = False

# ---- skill loop wiring (BRAIN_SKILLS) --------------------------------------
# _run_skill_loop is the seam brain.py hands brain_skills; these checks prove
# the injected context is wired to the real functions and that a broken module
# returns None (the caller then falls through to the existing web path).
import brain_skills

ctx_seen = []
_orig_answer = brain_skills.answer_with_skills
brain_skills.answer_with_skills = lambda ctx, q: ctx_seen.append((ctx, q))
brain._run_skill_loop("llama3.2:3b", "who?", None, [], {"by_id": {}}, True,
                      "default")
check("skill loop: answer_with_skills is called with the question",
      len(ctx_seen) == 1 and ctx_seen[0][1] == "who?")
_ctx = ctx_seen[0][0]
check("skill loop: vault_system is brain's VAULT_FIRST_SYSTEM verbatim",
      _ctx.vault_system == brain.VAULT_FIRST_SYSTEM)
check("skill loop: is_insufficient is brain's own parser",
      _ctx.is_insufficient is brain._is_insufficient)
check("skill loop: web/doc prompts and sources helpers are brain's",
      _ctx.web_system == brain.WEB_SYSTEM
      and _ctx.doc_system == brain.DOC_SYSTEM
      and _ctx.vault_sources is brain.vault_sources)
ctx_seen[:] = []
brain._run_skill_loop("llama3.2:3b", "who?", None, [], {}, True, "parent")
check("skill loop: parent persona passes PARENT_SYSTEM",
      ctx_seen[0][0].vault_system == brain.PARENT_SYSTEM)
# The loop's retry has to be judged by the SAME text phase 1 used, per-intent
# directive included, or a multi-part question loses the cover-every-entity
# instruction on exactly the retry that was supposed to rescue it.
ctx_seen[:] = []
brain._run_skill_loop("llama3.2:3b", "which came first?", None, [], {}, True,
                      "default", "multi")
check("skill loop: the intent directive is appended, not substituted",
      ctx_seen[0][0].vault_system
      == brain._shape_for_intent(brain.VAULT_FIRST_SYSTEM, "multi")
      and ctx_seen[0][0].vault_system.startswith(brain.VAULT_FIRST_SYSTEM))
# The last exchange reaches the loop as a short pre-rendered blob, never as
# extra chat turns.
ctx_seen[:] = []
brain._run_skill_loop(
    "llama3.2:3b", "what about her other paper?", None, [], {}, True,
    "default", "analytic",
    [{"role": "user", "content": "who is Maartje De Meulder"},
     {"role": "assistant", "content": "A Deaf sign language researcher."}])
check("skill loop: the prior turn is passed as a capped string",
      "Maartje De Meulder" in ctx_seen[0][0].prev_turn
      and len(ctx_seen[0][0].prev_turn) < 500)
brain_skills.answer_with_skills = lambda ctx, q: (_ for _ in ()).throw(
    RuntimeError("boom"))
check("skill loop: a raising module returns None, not an exception",
      brain._run_skill_loop("llama3.2:3b", "who?", None, [], {}, True,
                            "default") is None)
brain_skills.answer_with_skills = _orig_answer

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
