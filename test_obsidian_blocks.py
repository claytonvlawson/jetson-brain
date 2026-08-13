"""Unit tests for obsidian_rank.py's paragraph-level block support
(BRAIN_OBSIDIAN_BLOCKS).

Synthetic .ajson fixtures in a tmpdir, never the real vault: the plugin rewrites
.smart-env/multi/ whenever Obsidian runs, so any assertion against real counts
would be a time bomb. The real-vault numbers belong in the one-off sanity check,
not here.

Every test that needs blocks passes enabled=True explicitly, because the flag's
production default is OFF and one test below asserts exactly that.
"""

import json
import os
import tempfile
import unittest

import obsidian_rank as ob

NOTE_A = "Notes/A.md"
BLOCK_A_INTRO = f"{NOTE_A}#A Title#Intro"
BLOCK_A_METHODS = f"{NOTE_A}#A Title#Methods and design"


def _source_entry(path, vec, model=ob.EMBED_KEY):
    body = {"path": path, "embeddings": {model: {"vec": vec}}}
    return f'"{ob.SOURCE_PREFIX}{path}": {json.dumps(body)},'


def _block_entry(block_key, vec, model=ob.EMBED_KEY, lines=(1, 9)):
    """One smart_blocks: entry in the real on-disk shape: `path` is null (only the
    KEY carries the parent note), `text` is null, `lines` is a [start, end] pair."""
    body = {
        "path": None,
        "text": None,
        "key": block_key,
        "lines": list(lines),
        "class_name": "SmartBlock",
        "embeddings": {model: {"vec": vec}} if vec else {},
    }
    return f'"{ob.BLOCK_PREFIX}{block_key}": {json.dumps(body)},'


def _write_ajson_file(dirpath, filename, raw_lines):
    fpath = os.path.join(dirpath, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(raw_lines))
    return fpath


class TestParseBlockKey(unittest.TestCase):
    def test_splits_note_path_from_heading_path(self):
        note, headings = ob.parse_block_key(BLOCK_A_METHODS)
        self.assertEqual(note, NOTE_A)
        self.assertEqual(headings, ["A Title", "Methods and design"])

    def test_accepts_the_raw_prefixed_key_too(self):
        note, headings = ob.parse_block_key(ob.BLOCK_PREFIX + BLOCK_A_INTRO)
        self.assertEqual(note, NOTE_A)
        self.assertEqual(headings, ["A Title", "Intro"])

    def test_ordinal_and_frontmatter_segments_survive_parsing(self):
        note, headings = ob.parse_block_key(f"{NOTE_A}#A Title#{{1}}")
        self.assertEqual((note, headings), (NOTE_A, ["A Title", "{1}"]))
        note, headings = ob.parse_block_key(f"{NOTE_A}#---frontmatter---")
        self.assertEqual((note, headings), (NOTE_A, ["---frontmatter---"]))

    def test_malformed_keys_return_none(self):
        for bad in ["", None, "no-hash-at-all.md", "#leading-hash-only"]:
            self.assertEqual(ob.parse_block_key(bad), (None, []))


class TestLoadBlockVectors(unittest.TestCase):
    def _fixture_dir(self, d):
        _write_ajson_file(
            d,
            "a.ajson",
            [
                _source_entry(NOTE_A, [1.0, 0.0]),
                _block_entry(BLOCK_A_INTRO, [1.0, 0.0]),
                _block_entry(BLOCK_A_METHODS, [0.0, 1.0]),
                # No vector yet (below the plugin's min_chars) -- must be skipped.
                _block_entry(f"{NOTE_A}#---frontmatter---", None),
            ],
        )
        _write_ajson_file(
            d,
            "b.ajson",
            [
                _source_entry("Notes/B.md", [0.0, 1.0]),
                _block_entry("Notes/B.md#B Title#Only", [0.5, 0.5]),
            ],
        )
        return d

    def test_loads_only_blocks_with_vectors_keyed_without_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixture_dir(d)
            blocks = ob.load_block_vectors(smart_env_dir=d, enabled=True)
            self.assertEqual(
                set(blocks),
                {BLOCK_A_INTRO, BLOCK_A_METHODS, "Notes/B.md#B Title#Only"},
            )
            self.assertEqual(blocks[BLOCK_A_INTRO], [1.0, 0.0])

    def test_note_vectors_are_unaffected_by_blocks_in_the_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixture_dir(d)
            notes = ob.load_note_vectors(smart_env_dir=d)
            self.assertEqual(set(notes), {NOTE_A, "Notes/B.md"})

    def test_flag_off_returns_empty_without_reading_the_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixture_dir(d)
            self.assertEqual(ob.load_block_vectors(smart_env_dir=d, enabled=False), {})

    def test_skips_malformed_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_ajson_file(d, "good.ajson", [_block_entry(BLOCK_A_INTRO, [1.0, 0.0])])
            _write_ajson_file(d, "bad.ajson", ["not valid ajson content {{{"])
            blocks = ob.load_block_vectors(smart_env_dir=d, enabled=True)
            self.assertEqual(set(blocks), {BLOCK_A_INTRO})

    def test_tolerates_a_non_dict_entry(self):
        with tempfile.TemporaryDirectory() as d:
            _write_ajson_file(
                d,
                "weird.ajson",
                [
                    f'"{ob.BLOCK_PREFIX}{NOTE_A}#Dead": null,',
                    _block_entry(BLOCK_A_INTRO, [1.0, 0.0]),
                ],
            )
            blocks = ob.load_block_vectors(smart_env_dir=d, enabled=True)
            self.assertEqual(set(blocks), {BLOCK_A_INTRO})

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ob.load_block_vectors(smart_env_dir=d, enabled=True), {})


class TestFlagDefault(unittest.TestCase):
    def test_blocks_are_off_by_default(self):
        # The production default must stay OFF until the A/B pass. If someone
        # exports BRAIN_OBSIDIAN_BLOCKS in their shell this legitimately flips,
        # so assert against the env var, not blindly against False.
        expected = os.environ.get("BRAIN_OBSIDIAN_BLOCKS", "0") not in (
            "0", "false", "False", "no")
        self.assertEqual(ob.BLOCKS_ENABLED, expected)
        self.assertFalse(ob._blocks_on(False))
        self.assertTrue(ob._blocks_on(True))
        self.assertEqual(ob._blocks_on(None), ob.BLOCKS_ENABLED)


class TestBlocksByNote(unittest.TestCase):
    def test_maps_each_block_back_to_its_parent_note(self):
        blocks = {
            BLOCK_A_INTRO: [1.0, 0.0],
            BLOCK_A_METHODS: [0.0, 1.0],
            "Notes/B.md#B Title#Only": [0.5, 0.5],
        }
        index = ob.blocks_by_note(blocks)
        self.assertEqual(set(index), {NOTE_A, "Notes/B.md"})
        self.assertEqual([k for k, _ in index[NOTE_A]], [BLOCK_A_INTRO, BLOCK_A_METHODS])

    def test_drops_unparseable_keys(self):
        index = ob.blocks_by_note({"garbage-no-hash": [1.0], BLOCK_A_INTRO: [1.0, 0.0]})
        self.assertEqual(set(index), {NOTE_A})


class TestRankNoteBlocks(unittest.TestCase):
    BLOCKS = {
        BLOCK_A_INTRO: [1.0, 0.0],
        BLOCK_A_METHODS: [0.0, 1.0],
        "Notes/B.md#B Title#Only": [0.9, 0.1],
    }

    def test_ranks_within_one_note_only(self):
        ranked = ob.rank_note_blocks(self.BLOCKS, NOTE_A, [0.0, 1.0], top_k=10)
        self.assertEqual([k for k, _ in ranked], [BLOCK_A_METHODS, BLOCK_A_INTRO])

    def test_accepts_a_prebuilt_index_identically(self):
        index = ob.blocks_by_note(self.BLOCKS)
        self.assertEqual(
            ob.rank_note_blocks(index, NOTE_A, [0.0, 1.0], top_k=10),
            ob.rank_note_blocks(self.BLOCKS, NOTE_A, [0.0, 1.0], top_k=10),
        )

    def test_respects_top_k(self):
        self.assertEqual(len(ob.rank_note_blocks(self.BLOCKS, NOTE_A, [1.0, 1.0], top_k=1)), 1)

    def test_unknown_note_or_empty_inputs_return_empty(self):
        self.assertEqual(ob.rank_note_blocks(self.BLOCKS, "Notes/Nope.md", [1.0, 0.0]), [])
        self.assertEqual(ob.rank_note_blocks({}, NOTE_A, [1.0, 0.0]), [])
        self.assertEqual(ob.rank_note_blocks(self.BLOCKS, NOTE_A, None), [])

    def test_best_block_for_note(self):
        self.assertEqual(ob.best_block_for_note(self.BLOCKS, NOTE_A, [0.0, 1.0]),
                         BLOCK_A_METHODS)
        self.assertIsNone(ob.best_block_for_note(self.BLOCKS, "Notes/Nope.md", [0.0, 1.0]))


class TestBlockToChunkMatching(unittest.TestCase):
    # Shaped like build_index.py output: every chunk's text starts with the
    # heading breadcrumb _crumb_prefix() prepends.
    CHUNKS = [
        (1, "A / A Title\nopening paragraph about nothing in particular"),
        (2, "A / A Title / Methods and design\nwe sampled twelve participants"),
    ]

    def test_heading_text_drops_frontmatter_and_ordinals(self):
        self.assertEqual(ob.block_heading_text(BLOCK_A_METHODS), "A Title Methods and design")
        self.assertEqual(ob.block_heading_text(f"{NOTE_A}#---frontmatter---"), "")
        self.assertEqual(ob.block_heading_text(f"{NOTE_A}#A Title#{{1}}"), "A Title")

    def test_picks_the_chunk_whose_breadcrumb_matches(self):
        self.assertEqual(ob.best_chunk_for_block(BLOCK_A_METHODS, self.CHUNKS), 2)

    def test_no_overlap_returns_none_so_caller_keeps_its_fallback(self):
        self.assertIsNone(ob.best_chunk_for_block(f"{NOTE_A}#---frontmatter---", self.CHUNKS))
        self.assertIsNone(ob.best_chunk_for_block(BLOCK_A_METHODS, []))

    def test_best_chunk_for_note_beats_picking_chunk_zero(self):
        blocks = {BLOCK_A_INTRO: [1.0, 0.0], BLOCK_A_METHODS: [0.0, 1.0]}
        # Signal points at the Methods block, so we must NOT get chunk 1.
        self.assertEqual(
            ob.best_chunk_for_note(blocks, NOTE_A, [0.0, 1.0], self.CHUNKS), 2)

    def test_best_chunk_for_note_falls_through_an_unlocatable_top_block(self):
        # Top-ranked block is an unheaded {n} run that maps to no chunk; the
        # next-best block must still be used rather than giving up.
        blocks = {f"{NOTE_A}#{{1}}": [0.0, 1.0], BLOCK_A_METHODS: [0.0, 0.99]}
        self.assertEqual(
            ob.best_chunk_for_note(blocks, NOTE_A, [0.0, 1.0], self.CHUNKS), 2)

    def test_best_chunk_for_note_returns_none_when_nothing_maps(self):
        blocks = {f"{NOTE_A}#{{1}}": [0.0, 1.0]}
        self.assertIsNone(ob.best_chunk_for_note(blocks, NOTE_A, [0.0, 1.0], self.CHUNKS))


class TestExportRoundTripAndBackwardsCompat(unittest.TestCase):
    def _write_export(self, path, artifact):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f)

    def test_old_export_without_blocks_key_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "obsidian_vectors.json")
            self._write_export(p, {"embed_model": ob.EMBED_KEY, "dim": 2,
                                   "note_count": 1, "vectors": {NOTE_A: [1.0, 0.0]}})
            self.assertEqual(ob.load_block_vectors_from_export(p, enabled=True), {})
            # ...and the note-level path is untouched by the new code.
            self.assertEqual(ob.load_note_vectors_from_export(p), {NOTE_A: [1.0, 0.0]})

    def test_new_export_blocks_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "obsidian_vectors.json")
            self._write_export(p, {"embed_model": ob.EMBED_KEY, "dim": 2,
                                   "note_count": 1, "vectors": {NOTE_A: [1.0, 0.0]},
                                   "block_count": 1,
                                   "blocks": {BLOCK_A_INTRO: [1.0, 0.0]}})
            self.assertEqual(ob.load_block_vectors_from_export(p, enabled=True),
                             {BLOCK_A_INTRO: [1.0, 0.0]})

    def test_blocks_export_is_memoized_and_invalidated_by_mtime(self):
        # brain.py calls this once per QUESTION and the real export is 24 MB;
        # re-parsing it every turn cost ~0.29 s/question measured on a Mac.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "obsidian_vectors.json")
            self._write_export(p, {"blocks": {BLOCK_A_INTRO: [1.0, 0.0]}})
            ob.clear_block_vector_cache()
            first = ob.load_block_vectors_from_export(p, enabled=True)
            second = ob.load_block_vectors_from_export(p, enabled=True)
            self.assertIs(first, second)      # not re-parsed
            # A fresh export must be picked up without restarting the process.
            self._write_export(p, {"blocks": {BLOCK_A_INTRO: [0.0, 1.0]}})
            os.utime(p, (0, 0))
            third = ob.load_block_vectors_from_export(p, enabled=True)
            self.assertEqual(third, {BLOCK_A_INTRO: [0.0, 1.0]})
            ob.clear_block_vector_cache()
            self.assertEqual(ob._BLOCK_CACHE["blocks"], {})

    def test_missing_export_and_flag_off_return_empty(self):
        self.assertEqual(
            ob.load_block_vectors_from_export("/nonexistent/vectors.json", enabled=True), {})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "obsidian_vectors.json")
            self._write_export(p, {"blocks": {BLOCK_A_INTRO: [1.0, 0.0]}})
            self.assertEqual(ob.load_block_vectors_from_export(p, enabled=False), {})

    def test_exporter_omits_blocks_by_default_and_includes_them_on_request(self):
        import export_obsidian_vectors as ex

        with tempfile.TemporaryDirectory() as d:
            _write_ajson_file(
                d,
                "a.ajson",
                [_source_entry(NOTE_A, [1.0, 0.0]), _block_entry(BLOCK_A_INTRO, [1.0, 0.0])],
            )
            out = os.path.join(d, "out.json")

            _p, artifact = ex.export(smart_env_dir=d, output_path=out, include_blocks=False)
            self.assertNotIn("blocks", artifact)
            self.assertEqual(artifact["note_count"], 1)

            _p, artifact = ex.export(smart_env_dir=d, output_path=out, include_blocks=True)
            self.assertEqual(artifact["block_count"], 1)
            self.assertEqual(set(artifact["blocks"]), {BLOCK_A_INTRO})
            self.assertEqual(artifact["note_count"], 1)
            with open(out, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["blocks"], {BLOCK_A_INTRO: [1.0, 0.0]})


if __name__ == "__main__":
    unittest.main()
