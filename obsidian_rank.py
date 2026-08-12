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

DEFAULT_SMART_ENV_DIR = os.environ.get("BRAIN_OBSIDIAN_SMART_ENV_DIR") or os.path.join(
    os.path.expanduser("~"), "Desktop", "Dream", "ThesisVault", ".smart-env", "multi"
)
EMBED_KEY = "TaylorAI/bge-micro-v2"
SOURCE_PREFIX = "smart_sources:"


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
