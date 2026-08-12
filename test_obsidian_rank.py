"""Unit tests for obsidian_rank.py -- uses synthetic .ajson fixtures, not the real
vault, so these stay reproducible and fast regardless of vault content/state."""

import json
import math
import os
import tempfile
import unittest

import obsidian_rank as ob


def _ajson_entry(path, vec, model=ob.EMBED_KEY):
    """Build one smart_sources: entry as raw (unwrapped) AJSON text, matching the
    real plugin's on-disk format: `"key": {...},`"""
    key = f"smart_sources:{path}"
    body = {"path": path, "embeddings": {model: {"vec": vec}}}
    return f'"{key}": {json.dumps(body)},'


def _write_ajson_file(dirpath, filename, raw_lines):
    fpath = os.path.join(dirpath, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(raw_lines))
    return fpath


class TestParseAjsonFile(unittest.TestCase):
    def test_single_entry_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            fpath = _write_ajson_file(
                d, "a.ajson", [_ajson_entry("Notes/A.md", [1.0, 0.0, 0.0])]
            )
            parsed = ob._parse_ajson_file(fpath)
            self.assertEqual(len(parsed), 1)
            entry = parsed["smart_sources:Notes/A.md"]
            self.assertEqual(entry["path"], "Notes/A.md")

    def test_multiple_entries_including_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            fpath = _write_ajson_file(
                d,
                "a.ajson",
                [
                    _ajson_entry("Notes/A.md", [1.0, 0.0]),
                    '"smart_blocks:Notes/A.md#Heading": {"path": null, "embeddings": {}},',
                ],
            )
            parsed = ob._parse_ajson_file(fpath)
            self.assertEqual(len(parsed), 2)
            self.assertIn("smart_sources:Notes/A.md", parsed)
            self.assertIn("smart_blocks:Notes/A.md#Heading", parsed)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            fpath = _write_ajson_file(d, "empty.ajson", [""])
            self.assertEqual(ob._parse_ajson_file(fpath), {})


class TestLoadNoteVectors(unittest.TestCase):
    def test_loads_only_source_entries_with_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            _write_ajson_file(
                d,
                "a.ajson",
                [
                    _ajson_entry("Notes/A.md", [1.0, 0.0, 0.0]),
                    '"smart_blocks:Notes/A.md#H": {"path": null, "embeddings": {"TaylorAI/bge-micro-v2": {"vec": [0.1, 0.2, 0.3]}}},',
                ],
            )
            _write_ajson_file(
                d, "b.ajson", [_ajson_entry("Notes/B.md", [0.0, 1.0, 0.0])]
            )
            # Note with no embedding yet (below min_chars) -- should be skipped.
            _write_ajson_file(
                d,
                "c.ajson",
                ['"smart_sources:Notes/C.md": {"path": "Notes/C.md", "embeddings": {}},'],
            )
            vectors = ob.load_note_vectors(smart_env_dir=d)
            self.assertEqual(set(vectors.keys()), {"Notes/A.md", "Notes/B.md"})
            self.assertEqual(vectors["Notes/A.md"], [1.0, 0.0, 0.0])

    def test_skips_malformed_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_ajson_file(d, "good.ajson", [_ajson_entry("Notes/A.md", [1.0, 0.0])])
            _write_ajson_file(d, "bad.ajson", ["not valid ajson content {{{"])
            vectors = ob.load_note_vectors(smart_env_dir=d)
            self.assertEqual(set(vectors.keys()), {"Notes/A.md"})

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ob.load_note_vectors(smart_env_dir=d), {})


class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(ob.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(ob.cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(ob.cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(ob.cosine([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertEqual(ob.cosine([1.0, 1.0], [0.0, 0.0]), 0.0)

    def test_scale_invariant(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # same direction, different magnitude
        self.assertAlmostEqual(ob.cosine(a, b), 1.0)


class TestRankByQueryVector(unittest.TestCase):
    def test_orders_by_similarity_descending(self):
        vectors = {
            "close.md": [1.0, 0.0],
            "far.md": [0.0, 1.0],
            "mid.md": [0.7, 0.3],
        }
        query = [1.0, 0.0]
        ranked = ob.rank_by_query_vector(vectors, query, top_k=10)
        self.assertEqual([p for p, _ in ranked], ["close.md", "mid.md", "far.md"])

    def test_respects_top_k(self):
        vectors = {f"n{i}.md": [float(i), 1.0] for i in range(10)}
        ranked = ob.rank_by_query_vector(vectors, [5.0, 1.0], top_k=3)
        self.assertEqual(len(ranked), 3)

    def test_empty_vectors_returns_empty(self):
        self.assertEqual(ob.rank_by_query_vector({}, [1.0, 0.0]), [])


class TestNearestNeighbors(unittest.TestCase):
    def test_excludes_seed_itself(self):
        vectors = {
            "seed.md": [1.0, 0.0],
            "close.md": [0.9, 0.1],
            "far.md": [0.0, 1.0],
        }
        ranked = ob.nearest_neighbors(vectors, "seed.md", top_k=10)
        paths = [p for p, _ in ranked]
        self.assertNotIn("seed.md", paths)
        self.assertEqual(paths[0], "close.md")

    def test_unknown_seed_returns_empty(self):
        vectors = {"a.md": [1.0, 0.0]}
        self.assertEqual(ob.nearest_neighbors(vectors, "nope.md"), [])


class TestToRankPositions(unittest.TestCase):
    def test_zero_indexed_positions(self):
        ranked = [("a.md", 0.9), ("b.md", 0.5), ("c.md", 0.1)]
        self.assertEqual(
            ob.to_rank_positions(ranked), {"a.md": 0, "b.md": 1, "c.md": 2}
        )

    def test_empty_list(self):
        self.assertEqual(ob.to_rank_positions([]), {})


class TestLoadNoteVectorsFromExport(unittest.TestCase):
    def test_loads_vectors_from_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            export_path = os.path.join(d, "obsidian_vectors.json")
            artifact = {
                "embed_model": ob.EMBED_KEY,
                "dim": 2,
                "note_count": 2,
                "vectors": {"a.md": [1.0, 0.0], "b.md": [0.0, 1.0]},
            }
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f)
            vectors = ob.load_note_vectors_from_export(export_path)
            self.assertEqual(vectors, {"a.md": [1.0, 0.0], "b.md": [0.0, 1.0]})

    def test_missing_file_returns_empty_not_raise(self):
        vectors = ob.load_note_vectors_from_export("/nonexistent/path/vectors.json")
        self.assertEqual(vectors, {})


class TestExpandFromSeeds(unittest.TestCase):
    def test_single_seed_matches_nearest_neighbors(self):
        vectors = {
            "seed.md": [1.0, 0.0],
            "close.md": [0.9, 0.1],
            "far.md": [0.0, 1.0],
        }
        expanded = ob.expand_from_seeds(vectors, ["seed.md"], top_k_per_seed=10)
        direct = ob.nearest_neighbors(vectors, "seed.md", top_k=10)
        self.assertEqual(expanded, direct)

    def test_excludes_all_seeds_from_results(self):
        vectors = {
            "seed_a.md": [1.0, 0.0],
            "seed_b.md": [0.9, 0.1],
            "other.md": [0.0, 1.0],
        }
        expanded = ob.expand_from_seeds(vectors, ["seed_a.md", "seed_b.md"], top_k_per_seed=10)
        paths = [p for p, _ in expanded]
        self.assertNotIn("seed_a.md", paths)
        self.assertNotIn("seed_b.md", paths)
        self.assertIn("other.md", paths)

    def test_dedup_keeps_best_score_across_seeds(self):
        # "shared.md" is reachable from both seeds; the closer seed should win.
        vectors = {
            "seed_a.md": [1.0, 0.0],
            "seed_b.md": [0.0, 1.0],
            "shared.md": [0.99, 0.01],  # very close to seed_a, far from seed_b
        }
        expanded = ob.expand_from_seeds(vectors, ["seed_a.md", "seed_b.md"], top_k_per_seed=10)
        scores = dict(expanded)
        expected_best = ob.cosine(vectors["shared.md"], vectors["seed_a.md"])
        self.assertAlmostEqual(scores["shared.md"], expected_best)

    def test_respects_max_results(self):
        vectors = {f"n{i}.md": [float(i), 1.0] for i in range(20)}
        vectors["seed.md"] = [10.0, 1.0]
        expanded = ob.expand_from_seeds(vectors, ["seed.md"], top_k_per_seed=20, max_results=4)
        self.assertEqual(len(expanded), 4)

    def test_unknown_seeds_return_empty(self):
        vectors = {"a.md": [1.0, 0.0]}
        self.assertEqual(ob.expand_from_seeds(vectors, ["nope.md"]), [])

    def test_empty_seed_list_returns_empty(self):
        vectors = {"a.md": [1.0, 0.0], "b.md": [0.0, 1.0]}
        self.assertEqual(ob.expand_from_seeds(vectors, []), [])


if __name__ == "__main__":
    unittest.main()
