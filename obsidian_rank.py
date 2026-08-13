"""Reads Obsidian's Smart Connections Pro plugin embeddings (.smart-env/multi/*.ajson)
as an independent, rank-fusable retrieval signal.

Standalone by design: no imports from brain.py / build_index.py, no side effects at
import time, no network calls. This module only knows how to parse the plugin's
on-disk vectors and rank notes by cosine similarity -- it does not do any inference
itself (embedding a live query requires the same TaylorAI/bge-micro-v2 model the
plugin used, which is a separate integration decision).

The vectors here (384-dim, TaylorAI/bge-micro-v2) live in a different space than
brain.py's own dense index (nomic-embed-text via Ollama), so they are NOT
comparable by raw cosine score across the two sources. Fuse by RANK POSITION only
(the same way brain.py's hybrid RRF already fuses BM25 rank with dense rank) --
see to_rank_positions() below and [[project-jetson-brain-hybrid-rrf]].
"""

import glob
import json
import math
import os
import re

DEFAULT_SMART_ENV_DIR = os.environ.get("BRAIN_OBSIDIAN_SMART_ENV_DIR") or os.path.join(
    os.path.expanduser("~"), "Desktop", "Dream", "ThesisVault", ".smart-env", "multi"
)
EMBED_KEY = "TaylorAI/bge-micro-v2"
SOURCE_PREFIX = "smart_sources:"
BLOCK_PREFIX = "smart_blocks:"

# Paragraph-level (smart_blocks:) vectors are OFF by default, matching brain.py's
# OBSIDIAN_RANK_ENABLED discipline: a new retrieval signal ships dark until an A/B
# pass against gold/heldout/parent says it earns its keep. Measured cost of turning
# it on: the compact export grows from ~1.3 MB (306 notes) to ~22 MB (306 notes +
# 5,024 blocks), which is a real deploy.sh payload change, so it is opt-in on the
# export side too (export_obsidian_vectors.py --blocks).
BLOCKS_ENABLED = os.environ.get("BRAIN_OBSIDIAN_BLOCKS", "0") not in (
    "0", "false", "False", "no")


def _parse_ajson_file(path):
    """Parse one .ajson file into a dict.

    The plugin's "AJSON" format is an append-only sequence of `"key": {...},`
    entries with no surrounding braces -- wrapping in `{...}` and stripping the
    trailing comma turns it into one valid JSON object (later-appended keys with
    the same name legitimately overwrite earlier ones via normal dict semantics).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    wrapped = "{" + raw.rstrip(",") + "}"
    return json.loads(wrapped)


def load_note_vectors(smart_env_dir=None, embed_key=EMBED_KEY):
    """Return {note_path: vec} for every note with a computed embedding.

    Only reads `smart_sources:` entries (one per note). Skips `smart_blocks:`
    entries (per-chunk, used internally by the plugin's own UI) and notes with
    no embedding yet (below the plugin's min_chars threshold, or not indexed).
    A malformed/unreadable .ajson file is skipped rather than raising, since this
    reads plugin-managed state outside this codebase's control.
    """
    smart_env_dir = smart_env_dir or DEFAULT_SMART_ENV_DIR
    vectors = {}
    for fpath in glob.glob(os.path.join(smart_env_dir, "*.ajson")):
        try:
            entries = _parse_ajson_file(fpath)
        except (json.JSONDecodeError, OSError):
            continue
        for key, entry in entries.items():
            if not key.startswith(SOURCE_PREFIX):
                continue
            note_path = entry.get("path")
            vec = entry.get("embeddings", {}).get(embed_key, {}).get("vec")
            if note_path and vec:
                vectors[note_path] = vec
    return vectors


DEFAULT_EXPORT_PATH = os.environ.get("BRAIN_OBSIDIAN_VECTORS_PATH") or os.path.join(
    os.path.dirname(__file__), "obsidian_vectors.json"
)


def load_note_vectors_from_export(export_path=None):
    """Load {note_path: vec} from the compact artifact export_obsidian_vectors.py
    produces, instead of re-parsing the raw (much larger) .smart-env/multi/
    directory. This is what a deployed copy (e.g. on the Jetson) should use at
    runtime -- the raw plugin directory only needs to exist on the machine
    running Obsidian itself.

    Returns {} if the export file doesn't exist yet, rather than raising, so
    callers can treat "no export" the same as "no signal available" and degrade
    gracefully.
    """
    export_path = export_path or DEFAULT_EXPORT_PATH
    if not os.path.isfile(export_path):
        return {}
    with open(export_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    return artifact.get("vectors", {})


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_query_vector(vectors, query_vec, top_k=10):
    """Rank notes by cosine similarity to an externally-supplied query vector.

    Caller must embed the query with the SAME model (TaylorAI/bge-micro-v2) --
    this module performs no inference. Use this if/when live query embedding via
    that model is confirmed feasible on the Jetson.
    """
    scored = [(path, cosine(vec, query_vec)) for path, vec in vectors.items()]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def nearest_neighbors(vectors, seed_path, top_k=5):
    """Rank other notes by similarity to a SEED note's own vector.

    No query embedding required -- use an already-retrieved top BM25/dense hit as
    the seed to expand into semantically related notes the plugin has connected,
    even if live query-time inference with bge-micro-v2 turns out to be too slow
    on the Jetson.
    """
    seed_vec = vectors.get(seed_path)
    if seed_vec is None:
        return []
    scored = [
        (path, cosine(vec, seed_vec))
        for path, vec in vectors.items()
        if path != seed_path
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def to_rank_positions(ranked_list):
    """Convert a [(path, score), ...] ranked list into {path: rank} (0-indexed).

    This is the form brain.py's existing RRF fusion consumes for its BM25/dense
    rank lists -- integration should add this as a third ranked list rather than
    blending raw scores across incompatible embedding spaces.
    """
    return {path: i for i, (path, _score) in enumerate(ranked_list)}


def expand_from_seeds(vectors, seed_paths, top_k_per_seed=5, max_results=10):
    """Combine nearest-neighbor expansions from MULTIPLE seed notes into one
    ranked list -- the integration entry point for brain.py.

    Intended call shape: seed with the top few paths from brain.py's existing
    BM25/dense hit lists (already retrieved, no extra cost), get back a ranked
    list of related-but-not-yet-retrieved notes, then feed that through
    to_rank_positions() into the existing RRF fusion as a third signal. Requires
    no query embedding, so it works on the Jetson without torch/transformers.

    Dedup: a note reachable from more than one seed keeps its BEST (highest)
    similarity score rather than being counted once per seed. Seeds themselves,
    and any duplicate seed paths, are excluded from the result -- they're already
    present in the caller's own BM25/dense rank lists.
    """
    seed_set = set(seed_paths)
    best_score = {}
    for seed in seed_paths:
        for path, score in nearest_neighbors(vectors, seed, top_k=top_k_per_seed):
            if path in seed_set:
                continue
            if path not in best_score or score > best_score[path]:
                best_score[path] = score
    ranked = sorted(best_score.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:max_results]


# ---------------------------------------------------------------------------
# Paragraph-level blocks (smart_blocks:) -- BRAIN_OBSIDIAN_BLOCKS
#
# Counted in the live vault: 309 smart_sources: keys (306 with a vector, what
# everything above uses) and 10,610 smart_blocks: keys, of which 5,024 carry a
# vector. Everything above this line throws those 5,024 away. They matter because
# they are PARAGRAPH granularity, which is the granularity brain.py retrieves at:
# brain.py's expand_with_obsidian_signal currently picks an expanded note's chunk
# with `cid = ids[0]` and admits in a comment it has "no BM25 score here to pick a
# better one by". A block vector is exactly that missing score.
#
# Key format (verified against the real .ajson files):
#     smart_blocks:<note path>.md#<H1>#<H2>#...#{<n>}
# The parent note path is everything before the FIRST "#", identical to the
# smart_sources: path for the same note. The remainder is the heading breadcrumb,
# "#"-separated, outermost first. Two special segments appear: "---frontmatter---"
# (the YAML block, never embedded) and "{n}" (an ordinal for an unheaded paragraph
# run under the preceding heading). The entry's own `path` field is null on blocks
# -- only the key carries the parent, so the key must be parsed, not the fields.
#
# Rejected: also carrying each block's `lines` [start, end]. It is present on disk
# and would be a tempting locator, but brain.py's chunks (build_index.py:860-864)
# store only id/text/source/title with NO line numbers, so line ranges would be
# unmatchable dead weight in the export. The heading breadcrumb IS matchable,
# because build_index prepends that same breadcrumb to every chunk's text
# (_crumb_prefix, build_index.py:328).
# ---------------------------------------------------------------------------

FRONTMATTER_SEGMENT = "---frontmatter---"


def _blocks_on(enabled):
    """Resolve the BRAIN_OBSIDIAN_BLOCKS gate. `enabled=None` means "use the flag";
    an explicit True/False lets a caller (or a test) override it without touching
    the process environment."""
    return BLOCKS_ENABLED if enabled is None else bool(enabled)


def parse_block_key(block_key):
    """Split a smart_blocks: key into (note_path, [heading, ...]).

    Returns (None, []) for anything that isn't a well-formed block key, so callers
    can filter plugin-managed junk without a try/except - this parses state owned
    by another program. The "smart_blocks:" prefix is optional on input so both the
    raw .ajson key and the export's already-stripped key work.
    """
    if not block_key:
        return None, []
    key = block_key[len(BLOCK_PREFIX):] if block_key.startswith(BLOCK_PREFIX) else block_key
    note_path, sep, rest = key.partition("#")
    if not note_path or not sep:
        return None, []
    headings = [seg for seg in rest.split("#") if seg]
    return note_path, headings


def load_block_vectors(smart_env_dir=None, embed_key=EMBED_KEY, enabled=None):
    """Return {block_key: vec} for every paragraph-level block with an embedding.

    Keys keep the "smart_blocks:" prefix stripped OFF so the same string round-trips
    through the export and back through parse_block_key(). Blocks with no vector
    (below the plugin's min_chars, frontmatter, unembedded tails) are skipped, as
    are malformed files - same tolerance as load_note_vectors, for the same reason.

    Returns {} when the flag is off, so an accidental call on the default path costs
    one dict allocation instead of a 31 MB directory walk.
    """
    if not _blocks_on(enabled):
        return {}
    smart_env_dir = smart_env_dir or DEFAULT_SMART_ENV_DIR
    vectors = {}
    for fpath in glob.glob(os.path.join(smart_env_dir, "*.ajson")):
        try:
            entries = _parse_ajson_file(fpath)
        except (json.JSONDecodeError, OSError):
            continue
        for key, entry in entries.items():
            if not key.startswith(BLOCK_PREFIX):
                continue
            if not isinstance(entry, dict):
                continue
            vec = entry.get("embeddings", {}).get(embed_key, {}).get("vec")
            if not vec:
                continue
            note_path, _headings = parse_block_key(key)
            if not note_path:
                continue
            vectors[key[len(BLOCK_PREFIX):]] = vec
    return vectors


# Module state for the blocks memoization below. A plain dict rather than
# functools.lru_cache: the cache key is a stat of the file, not the arguments,
# and it has to be droppable (clear_block_vector_cache) on an 8 GB box.
_BLOCK_CACHE = {"key": None, "blocks": {}}


def load_block_vectors_from_export(export_path=None, enabled=None):
    """Load {block_key: vec} from the compact artifact, mirroring
    load_note_vectors_from_export.

    Tolerates an OLD export written before blocks existed: a missing "blocks" key
    reads as {} rather than raising, so a Jetson still running the 1.3 MB
    notes-only artifact degrades to today's behaviour instead of crashing.

    MEMOIZED, because brain.py calls this once per QUESTION. Measured on the 40
    retrieval_gold questions against the real 24.1 MB / 5,024-vector export on a
    Mac (faster than the Jetson this ships to), expand_with_obsidian_signal cost
    0.373 s/question with blocks off and 0.666 s/question with blocks on: ~0.29 s
    of that was re-parsing this one file every turn. The key is
    (path, mtime, size) so a fresh export_obsidian_vectors.py run is picked up
    without a restart, and the stat is wrapped so any OSError just falls through
    to the uncached read. TRADEOFF: the cached dict is ~63 MB resident on an
    8 GB Jetson, turning a per-turn transient into steady-state RSS in a
    long-running shim/advisor process - clear_block_vector_cache() exists so
    that memory can be given back when the flag is turned off at runtime.
    Note-level vectors are NOT worth the same treatment on their own (the 1.35 MB
    export loads in ~16 ms); this is the expensive one.
    """
    if not _blocks_on(enabled):
        return {}
    export_path = export_path or DEFAULT_EXPORT_PATH
    if not os.path.isfile(export_path):
        return {}
    try:
        st = os.stat(export_path)
        key = (export_path, st.st_mtime, st.st_size)
    except OSError:
        key = None
    if key is not None and _BLOCK_CACHE.get("key") == key:
        return _BLOCK_CACHE["blocks"]
    with open(export_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    blocks = artifact.get("blocks", {})
    if key is not None:
        _BLOCK_CACHE["key"] = key
        _BLOCK_CACHE["blocks"] = blocks
    return blocks


def clear_block_vector_cache():
    """Drop the memoized blocks export (~63 MB). For a long-running process that
    turned BRAIN_OBSIDIAN_BLOCKS off and wants the memory back."""
    _BLOCK_CACHE["key"] = None
    _BLOCK_CACHE["blocks"] = {}


def blocks_by_note(block_vectors):
    """Invert {block_key: vec} into {note_path: [(block_key, vec), ...]}.

    Build this ONCE per process and reuse it: rank_note_blocks() over the raw dict
    would rescan all 5,024 blocks per note, and brain.py calls it once per expanded
    note. Insertion order is preserved, which for plugin-written files is document
    order, so ties fall back to "earliest block in the note" - the same choice
    `cid = ids[0]` makes today.
    """
    index = {}
    for block_key, vec in block_vectors.items():
        note_path, _headings = parse_block_key(block_key)
        if not note_path:
            continue
        index.setdefault(note_path, []).append((block_key, vec))
    return index


def rank_note_blocks(blocks, note_path, signal_vec, top_k=5):
    """Rank ONE note's blocks by cosine similarity to a query-side signal vector.

    `blocks` accepts either the flat {block_key: vec} map or a blocks_by_note()
    index, so callers that only touch one note don't have to build the index first.
    `signal_vec` must live in the SAME 384-dim bge-micro-v2 space as the blocks -
    either a live query embedding, or (the no-inference route brain.py can use
    today) an already-loaded note vector from load_note_vectors_from_export, since
    the plugin embeds sources and blocks with one model.

    Returns [(block_key, score), ...] descending. Scores are comparable WITHIN this
    list only - never across to brain.py's 768-dim nomic space. Cross-source fusion
    stays rank-position-only (to_rank_positions), exactly as the module docstring
    requires.
    """
    if not blocks or not signal_vec:
        return []
    first = next(iter(blocks.values()))
    if isinstance(first, list) and first and isinstance(first[0], tuple):
        candidates = blocks.get(note_path) or []
    else:
        candidates = [
            (block_key, vec)
            for block_key, vec in blocks.items()
            if parse_block_key(block_key)[0] == note_path
        ]
    scored = [(block_key, cosine(vec, signal_vec)) for block_key, vec in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def best_block_for_note(blocks, note_path, signal_vec):
    """Top-ranked block key for a note, or None. Thin wrapper over rank_note_blocks
    for the common "I just need the one best paragraph" call."""
    ranked = rank_note_blocks(blocks, note_path, signal_vec, top_k=1)
    return ranked[0][0] if ranked else None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_TOKEN_RE.findall((text or "").lower()))


def block_heading_text(block_key):
    """Heading breadcrumb of a block as plain text, frontmatter and {n} ordinal
    segments dropped (they carry no words to match on)."""
    _note_path, headings = parse_block_key(block_key)
    keep = [
        h for h in headings
        if h != FRONTMATTER_SEGMENT and not (h.startswith("{") and h.endswith("}"))
    ]
    return " ".join(keep)


def best_chunk_for_block(block_key, chunk_texts):
    """Pick which of a note's brain.py chunks corresponds to a block.

    `chunk_texts` is [(chunk_id, text), ...] - deliberately a plain pair list, so
    this module still knows nothing about brain.py's chunk dicts or its meta index.

    Matching is by heading-token overlap, not embeddings: build_index prepends the
    same heading breadcrumb to each chunk's text that the plugin encodes in the
    block key, and the two vector spaces cannot be compared anyway. Jaccard-style
    overlap normalised by the heading's own token count, so a long chunk isn't
    penalised for containing many other words.

    Returns None when nothing overlaps (a block whose only heading is the note
    title, an unheaded {n} block), which tells the caller to keep its existing
    fallback rather than guess.
    """
    heading_tokens = _tokens(block_heading_text(block_key))
    if not heading_tokens or not chunk_texts:
        return None
    best_id, best_score = None, 0.0
    for chunk_id, text in chunk_texts:
        overlap = len(heading_tokens & _tokens(text)) / len(heading_tokens)
        if overlap > best_score:
            best_id, best_score = chunk_id, overlap
    return best_id


def best_chunk_for_note(blocks, note_path, signal_vec, chunk_texts, max_blocks=5):
    """The whole point of this section: given an expanded note, the query-side
    signal, and that note's brain.py chunks, return the chunk id of its BEST
    paragraph instead of its first one.

    Walks the note's top `max_blocks` blocks in rank order and returns the first
    one that maps onto a chunk - a top block may be an unheaded {n} run that
    best_chunk_for_block can't locate, and falling through to the next-best block
    beats falling all the way back to chunk zero. Returns None if nothing maps, so
    the caller's existing `ids[0]` fallback stays in charge.
    """
    for block_key, _score in rank_note_blocks(blocks, note_path, signal_vec,
                                              top_k=max_blocks):
        chunk_id = best_chunk_for_block(block_key, chunk_texts)
        if chunk_id is not None:
            return chunk_id
    return None


def _run_selftest():
    """Load vectors and sanity-check retrieval end-to-end. Prints PASS/FAIL and
    exits nonzero on failure, matching asl_server.py's --selftest convention
    elsewhere in this repo.

    Prefers the compact export (what a deployed copy, e.g. on the Jetson, will
    actually have on disk) and falls back to parsing the raw .smart-env/multi/
    directory (what a Mac dev checkout next to the live Obsidian vault has) --
    this makes the selftest representative of both environments.
    """
    vectors = load_note_vectors_from_export()
    source = f"export at {DEFAULT_EXPORT_PATH}"
    if not vectors:
        vectors = load_note_vectors()
        source = f"raw .smart-env at {DEFAULT_SMART_ENV_DIR}"
    if len(vectors) < 100:
        print(f"FAIL obsidian_rank selftest: only {len(vectors)} note vectors loaded "
              f"(expected 100+) from {source}")
        return 1
    sample_path = next(iter(vectors))
    dim = len(vectors[sample_path])
    if dim != 384:
        print(f"FAIL obsidian_rank selftest: unexpected vector dim {dim} (expected 384)")
        return 1
    neighbors = nearest_neighbors(vectors, sample_path, top_k=3)
    if not neighbors:
        print("FAIL obsidian_rank selftest: nearest_neighbors returned nothing for a real note")
        return 1
    expanded = expand_from_seeds(vectors, [sample_path], top_k_per_seed=3, max_results=5)
    if not expanded:
        print("FAIL obsidian_rank selftest: expand_from_seeds returned nothing for a real seed")
        return 1
    print(f"PASS obsidian_rank selftest ({source}): {len(vectors)} notes, dim={dim}, "
          f"sample neighbors of {sample_path!r}: {[p for p, _ in neighbors]}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_selftest())
