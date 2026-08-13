"""Export Obsidian Smart Connections Pro embeddings into one compact artifact.

The plugin's raw .smart-env/multi/ directory (~31MB across ~309 files) stores
note-level vectors and per-chunk smart_blocks: vectors interleaved across 309
files. This script extracts them into a single JSON file -- something deploy.sh
can ship to the Jetson instead of the raw plugin directory.

Note vectors are always exported (306 of them, ~1.3 MB). Paragraph-level BLOCK
vectors are opt-in behind BRAIN_OBSIDIAN_BLOCKS=1 or --blocks, because the 5,024
of them take the artifact to ~22 MB -- a real deploy payload change that should
not happen by accident before the A/B pass says blocks help.

Usage:
    python3 export_obsidian_vectors.py [output_path] [--blocks]

Output JSON shape (the first four keys are unchanged from the pre-blocks export,
so an older reader keeps working against a newer file):
    {
      "embed_model": "TaylorAI/bge-micro-v2",
      "dim": 384,
      "note_count": 306,
      "vectors": {"<note path>": [<384 floats>], ...},
      "block_count": 5024,                        # only when blocks are included
      "blocks": {"<note path>#<H1>#<H2>": [...]}  # only when blocks are included
    }
"""

import json
import os
import sys

import obsidian_rank as ob

DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "obsidian_vectors.json")


def export(smart_env_dir=None, embed_model=ob.EMBED_KEY, output_path=None,
           include_blocks=None):
    """Write the artifact. `include_blocks=None` follows obsidian_rank's
    BRAIN_OBSIDIAN_BLOCKS flag; pass True/False to force it either way.

    The "blocks" key is OMITTED entirely rather than written as {} when blocks are
    off, so the default artifact stays byte-identical to what deploy.sh ships
    today and load_block_vectors_from_export's old-export tolerance is the one
    code path handling both cases.
    """
    output_path = output_path or DEFAULT_OUTPUT_PATH
    vectors = ob.load_note_vectors(smart_env_dir=smart_env_dir, embed_key=embed_model)
    dim = len(next(iter(vectors.values()))) if vectors else 0
    artifact = {
        "embed_model": embed_model,
        "dim": dim,
        "note_count": len(vectors),
        "vectors": vectors,
    }
    if ob._blocks_on(include_blocks):
        blocks = ob.load_block_vectors(smart_env_dir=smart_env_dir,
                                       embed_key=embed_model, enabled=True)
        artifact["block_count"] = len(blocks)
        artifact["blocks"] = blocks
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, separators=(",", ":"))
    return output_path, artifact


def main():
    args = [a for a in sys.argv[1:] if a != "--blocks"]
    include_blocks = True if "--blocks" in sys.argv[1:] else None
    output_path = args[0] if args else DEFAULT_OUTPUT_PATH
    path, artifact = export(output_path=output_path, include_blocks=include_blocks)
    size_kb = os.path.getsize(path) / 1024
    blocks_note = (f", {artifact['block_count']} block vectors"
                   if "blocks" in artifact else "")
    print(
        f"Wrote {artifact['note_count']} note vectors{blocks_note} "
        f"(dim={artifact['dim']}, model={artifact['embed_model']}) "
        f"to {path} ({size_kb:.0f} KB)"
    )
    if artifact["note_count"] == 0:
        print("WARNING: zero vectors exported -- check the smart_env_dir path", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
