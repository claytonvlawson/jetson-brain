"""Export Obsidian Smart Connections Pro embeddings into one compact artifact.

The plugin's raw .smart-env/multi/ directory (~31MB across ~309 files) mixes
note-level vectors (what obsidian_rank.py actually uses) with per-chunk
smart_blocks: entries the plugin's own UI needs but this pipeline doesn't. This
script extracts ONLY the note-level {path: vec} data via
obsidian_rank.load_note_vectors() into a single small JSON file -- something
deploy.sh can ship to the Jetson instead of the raw plugin directory.

Usage:
    python3 export_obsidian_vectors.py [output_path]

Output JSON shape:
    {
      "embed_model": "TaylorAI/bge-micro-v2",
      "dim": 384,
      "note_count": 306,
      "vectors": {"<note path>": [<384 floats>], ...}
    }
"""

import json
import os
import sys

import obsidian_rank as ob

DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "obsidian_vectors.json")


def export(smart_env_dir=None, embed_model=ob.EMBED_KEY, output_path=None):
    output_path = output_path or DEFAULT_OUTPUT_PATH
    vectors = ob.load_note_vectors(smart_env_dir=smart_env_dir, embed_key=embed_model)
    dim = len(next(iter(vectors.values()))) if vectors else 0
    artifact = {
        "embed_model": embed_model,
        "dim": dim,
        "note_count": len(vectors),
        "vectors": vectors,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, separators=(",", ":"))
    return output_path, artifact


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    path, artifact = export(output_path=output_path)
    size_kb = os.path.getsize(path) / 1024
    print(
        f"Wrote {artifact['note_count']} note vectors "
        f"(dim={artifact['dim']}, model={artifact['embed_model']}) "
        f"to {path} ({size_kb:.0f} KB)"
    )
    if artifact["note_count"] == 0:
        print("WARNING: zero vectors exported -- check the smart_env_dir path", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
