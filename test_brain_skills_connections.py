#!/usr/bin/env python3
"""Tests for the "connections" skill added to brain_skills.py.

Mirrors test_brain_skills.py's structure (FakeModel, chunk(), META, make_ctx,
SkillsOn). obsidian_rank is a SEPARATE standalone module that _do_connections
only ever reaches via a LOCAL "import obsidian_rank" inside the function body,
so the only way to fake it from a test is sys.modules injection BEFORE that
import executes - there is no module-level name in brain_skills.py to
monkeypatch.

Run: python3 test_brain_skills_connections.py
"""

import sys
import types
import unittest

import brain_skills as bs


# ---- Fakes shared with test_brain_skills.py's shape ------------------------


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, model, messages):
        self.calls.append(messages)
        if not self.replies:
            return "INSUFFICIENT"
        return self.replies.pop(0)


def chunk(source, text, cid=0):
    return {"id": cid, "source": source, "text": text}


META = {
    "by_source": {
        "History/Deaf President Now 1988.md": [0, 1],
        "CRPD/Article 2 Definitions.md": [2],
        "History/I King Jordan.md": [3],
    },
    "by_id": {
        0: chunk("History/Deaf President Now 1988.md", "DPN began in March 1988.", 0),
        1: chunk("History/Deaf President Now 1988.md", "I. King Jordan was named president.", 1),
        2: chunk("CRPD/Article 2 Definitions.md", "Article 2 defines communication.", 2),
        3: chunk("History/I King Jordan.md", "I. King Jordan became Gallaudet's first deaf president.", 3),
    },
    "link_map": {
        "deaf president now 1988": "History/Deaf President Now 1988.md",
        "dpn": "History/Deaf President Now 1988.md",
        "article 2 definitions": "CRPD/Article 2 Definitions.md",
        "i king jordan": "History/I King Jordan.md",
    },
}


def make_ctx(model, **kw):
    ctx = bs.SkillContext(
        model="fake",
        chat=model,
        meta=META,
        web_enabled=kw.pop("web_enabled", True),
        retrieve=kw.pop("retrieve", None),
        gather_web=kw.pop("gather_web", None),
        fetch_page=kw.pop("fetch_page", None),
        clock=kw.pop("clock", None),
        max_steps=kw.pop("max_steps", 0),
        deadline_s=kw.pop("deadline_s", 0.0),
    )
    for key, value in kw.items():
        setattr(ctx, key, value)
    return ctx


class SkillsOn:
    def __enter__(self):
        self.prev = bs.SKILLS_ENABLED
        bs.SKILLS_ENABLED = True

    def __exit__(self, *exc):
        bs.SKILLS_ENABLED = self.prev
        return False


class FakeObsidianRank:
    """Stands in for the real obsidian_rank module via sys.modules injection.

    `vectors` maps note path -> anything truthy (the real values are 384-dim
    embeddings; these tests only need presence/absence, since cosine math
    itself is obsidian_rank's own concern, already covered by its own tests).
    `neighbors` is the canned [(path, score), ...] nearest_neighbors() should
    return for ANY seed present in `vectors`. `boom` makes nearest_neighbors
    raise, to prove the handler survives a misbehaving dependency.
    """

    def __init__(self, vectors=None, neighbors=None, boom=False):
        self.vectors = vectors if vectors is not None else {}
        self.neighbors = neighbors if neighbors is not None else []
        self.boom = boom
        self.load_calls = 0
        self.neighbor_calls = []

    def load_note_vectors_from_export(self):
        self.load_calls += 1
        return self.vectors

    def nearest_neighbors(self, vectors, seed_path, top_k=5):
        self.neighbor_calls.append((seed_path, top_k))
        if self.boom:
            raise RuntimeError("smart-env export is corrupt")
        return self.neighbors[:top_k]


class ObsidianRankInjected:
    """Context manager: installs a fake module at sys.modules['obsidian_rank']
    for the duration of the block, then restores whatever was there before
    (nothing, in the normal case - this repo never imports it eagerly)."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self._had_prev = "obsidian_rank" in sys.modules
        self._prev = sys.modules.get("obsidian_rank")
        sys.modules["obsidian_rank"] = self.fake
        return self.fake

    def __exit__(self, *exc):
        if self._had_prev:
            sys.modules["obsidian_rank"] = self._prev
        else:
            del sys.modules["obsidian_rank"]
        return False


class ObsidianRankMissing:
    """Forces the local `import obsidian_rank` inside the handler to raise
    ImportError, by installing None at sys.modules (Python's documented way to
    make an import fail deliberately) rather than relying on it simply not
    being installed, which would be environment-dependent in this repo."""

    def __enter__(self):
        self._had_prev = "obsidian_rank" in sys.modules
        self._prev = sys.modules.get("obsidian_rank")
        sys.modules["obsidian_rank"] = None
        return None

    def __exit__(self, *exc):
        if self._had_prev:
            sys.modules["obsidian_rank"] = self._prev
        else:
            del sys.modules["obsidian_rank"]
        return False


# ---- Protocol / menu / ordering --------------------------------------------


class TestConnectionsProtocol(unittest.TestCase):

    def test_connections_is_a_known_skill_name(self):
        self.assertIn("connections", bs.SKILL_NAMES)

    def test_connections_is_vault_provenance(self):
        self.assertEqual(bs.PROVENANCE_LABELS["connections"], bs.LABEL_VAULT)
        self.assertEqual(bs.PROVENANCE_LABELS["connections"], "")

    def test_connections_sits_between_search_vault_and_web_search(self):
        order = bs.ESCALATION_ORDER
        self.assertIn("connections", order)
        self.assertLess(order.index("search_vault"), order.index("connections"))
        self.assertLess(order.index("connections"), order.index("web_search"))

    def test_connections_is_dispatched(self):
        self.assertIn("connections", bs._DISPATCH)

    def test_menu_advertises_connections(self):
        self.assertIn("SKILL: connections |", bs.build_skill_system(web_enabled=True))
        self.assertIn("SKILL: connections |", bs.build_skill_system(web_enabled=False))

    def test_connections_parses(self):
        skill, arg, ok = bs.parse_skill(
            "SKILL: connections | Deaf President Now 1988", "q", True)
        self.assertEqual((skill, arg, ok),
                         ("connections", "Deaf President Now 1988", True))


# ---- Behaviour ---------------------------------------------------------


class TestConnectionsHandler(unittest.TestCase):
    Q = "how does DPN relate to other events"

    def test_seed_text_alone_when_no_export(self):
        # obsidian_rank imports fine but the export is empty - degrade to the
        # seed note's own text, same permissive philosophy as
        # expand_with_obsidian_signal.
        fake = FakeObsidianRank(vectors={})
        model = FakeModel(["SKILL: connections | dpn",
                           "DPN began in March 1988 (Deaf President Now 1988)."])
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "connections")
        self.assertEqual(r.sources, ["History/Deaf President Now 1988.md"])
        self.assertEqual(r.provenance, "")
        self.assertNotIn("[provenance", r.answer)
        self.assertEqual(fake.neighbor_calls, [])   # never called: seed absent from {}

    def test_related_note_excerpt_reaches_the_answer_call(self):
        fake = FakeObsidianRank(
            vectors={"History/Deaf President Now 1988.md": [1, 0],
                     "History/I King Jordan.md": [0.9, 0.1]},
            neighbors=[("History/I King Jordan.md", 0.97)],
        )
        model = FakeModel(["SKILL: connections | dpn",
                           "DPN led to I. King Jordan becoming Gallaudet's "
                           "first deaf president (Deaf President Now 1988, "
                           "I King Jordan)."])
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "connections")
        self.assertIn("History/Deaf President Now 1988.md", r.sources)
        self.assertIn("History/I King Jordan.md", r.sources)
        # The answer call's user content carried both excerpts.
        answer_user = model.calls[1][1]["content"]
        self.assertIn("I. King Jordan became Gallaudet's first deaf president",
                      answer_user)
        self.assertEqual(fake.neighbor_calls,
                         [("History/Deaf President Now 1988.md", bs.CONN_TOP_K)])

    def test_unresolvable_note_name_does_not_crash(self):
        fake = FakeObsidianRank()
        model = FakeModel(["SKILL: connections | not a real note anywhere",
                           "SKILL: reason", "fallback answer"])
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        # Never raised: the loop continued past the failed connections step
        # and still produced a result (via the scripted reason fallback).
        self.assertTrue(r.ok)
        self.assertEqual(r.trace[0]["skill"], "connections")
        self.assertIn("no vault note by that name", r.trace[0]["observation"])

    def test_empty_argument_is_a_clean_observation_not_a_crash(self):
        result = bs._do_connections(make_ctx(FakeModel([])), self.Q, "")
        self.assertEqual(result, (None, "connections: no note or topic given"))

    def test_never_raises_when_obsidian_rank_import_fails(self):
        model = FakeModel(["SKILL: connections | dpn",
                           "DPN began in March 1988 (Deaf President Now 1988)."])
        with ObsidianRankMissing(), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        # Import failure degrades to seed-text-only, exactly like an empty
        # export - it must not blow up the whole skill.
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "connections")

    def test_never_raises_when_nearest_neighbors_itself_raises(self):
        fake = FakeObsidianRank(
            vectors={"History/Deaf President Now 1988.md": [1, 0]}, boom=True)
        model = FakeModel(["SKILL: connections | dpn",
                           "DPN began in March 1988 (Deaf President Now 1988)."])
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)   # survived, fell back to seed text alone
        self.assertEqual(r.skill, "connections")
        self.assertEqual(r.sources, ["History/Deaf President Now 1988.md"])

    def test_no_text_anywhere_is_a_clean_observation(self):
        # A note that resolves but has zero chunks, with no related notes
        # either: the handler must report, not crash.
        empty_meta = {"by_source": {"Empty/Note.md": []}, "by_id": {},
                      "link_map": {"empty note": "Empty/Note.md"}}
        fake = FakeObsidianRank(vectors={})
        model = FakeModel(["SKILL: connections | empty note",
                           "SKILL: reason", "fallback"])
        ctx = make_ctx(model, meta=empty_meta)
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(ctx, self.Q)
        self.assertTrue(r.ok)   # fell through to reason
        self.assertEqual(r.trace[0]["skill"], "connections")
        self.assertIn("has no text", r.trace[0]["observation"])

    def test_insufficient_answer_reports_related_count(self):
        fake = FakeObsidianRank(
            vectors={"History/Deaf President Now 1988.md": [1, 0],
                     "History/I King Jordan.md": [0.9, 0.1]},
            neighbors=[("History/I King Jordan.md", 0.97)],
        )
        model = FakeModel(["SKILL: connections | dpn", "INSUFFICIENT",
                           "SKILL: reason", "fallback"])
        with ObsidianRankInjected(fake), SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)
        self.assertIn("plus 1 related notes", r.trace[0]["observation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
