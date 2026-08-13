#!/usr/bin/env python3
"""Tests for brain_skills.py - the bounded skill loop.

No Ollama, no network, no BM25 index: every dependency arrives through
SkillContext, which is the whole point of the module being standalone. The
model is a scripted list of replies, so a test asserts on the loop's control
flow rather than on a 3B model's mood.

Run: python3 test_brain_skills.py
"""

import unittest

import brain_skills as bs


# ---- Fakes -----------------------------------------------------------------


class FakeModel:
    """Scripted replies. Selection calls and answer calls both land here; the
    test scripts them in the exact order the loop will make them."""

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
    },
    "by_id": {
        0: chunk("History/Deaf President Now 1988.md", "DPN began in March 1988.", 0),
        1: chunk("History/Deaf President Now 1988.md", "I. King Jordan was named president.", 1),
        2: chunk("CRPD/Article 2 Definitions.md", "Article 2 defines communication.", 2),
    },
    "link_map": {
        "deaf president now 1988": "History/Deaf President Now 1988.md",
        "dpn": "History/Deaf President Now 1988.md",
        "article 2 definitions": "CRPD/Article 2 Definitions.md",
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
    """The module flag is default OFF; every behavioural test turns it on."""

    def __enter__(self):
        self.prev = bs.SKILLS_ENABLED
        bs.SKILLS_ENABLED = True

    def __exit__(self, *exc):
        bs.SKILLS_ENABLED = self.prev
        return False


# ---- Parsing ---------------------------------------------------------------


class TestParse(unittest.TestCase):
    Q = "who led deaf president now"

    def parse(self, text, web=True):
        return bs.parse_skill(text, self.Q, web)

    def test_canonical(self):
        self.assertEqual(self.parse("SKILL: search_vault | dpn leaders"),
                         ("search_vault", "dpn leaders", True))

    def test_missing_skill_label(self):
        self.assertEqual(self.parse("search_vault | dpn leaders"),
                         ("search_vault", "dpn leaders", True))

    def test_wrong_pipe_spacing(self):
        self.assertEqual(self.parse("SKILL:search_vault|dpn leaders"),
                         ("search_vault", "dpn leaders", True))
        self.assertEqual(self.parse("SKILL: search_vault    |    dpn leaders"),
                         ("search_vault", "dpn leaders", True))

    def test_markdown_bold(self):
        self.assertEqual(self.parse("**SKILL: web_search | dpn 1988**"),
                         ("web_search", "dpn 1988", True))

    def test_code_fence(self):
        text = "```\nSKILL: get_note | Deaf President Now 1988\n```"
        self.assertEqual(self.parse(text),
                         ("get_note", "Deaf President Now 1988", True))

    def test_smart_quotes(self):
        text = "SKILL: get_note | ‘Deaf President Now 1988’"
        self.assertEqual(self.parse(text),
                         ("get_note", "Deaf President Now 1988", True))

    def test_trailing_prose(self):
        skill, arg, ok = self.parse(
            "SKILL: web_search | dpn 1988 protest\nI picked this because the "
            "vault had nothing.")
        self.assertEqual(skill, "web_search")
        self.assertTrue(ok)
        self.assertTrue(arg.startswith("dpn 1988 protest"))

    def test_bullet_and_numbering(self):
        self.assertEqual(self.parse("- SKILL: reason")[0], "reason")
        self.assertEqual(self.parse("1. SKILL: reason")[0], "reason")

    def test_hyphen_and_case_variants(self):
        self.assertEqual(self.parse("Search-Vault | x")[0], "search_vault")
        self.assertEqual(self.parse("SEARCH VAULT | x")[0], "search_vault")

    def test_bare_skill_name_gets_the_question(self):
        self.assertEqual(self.parse("SKILL: search_vault"),
                         ("search_vault", self.Q, True))

    def test_placeholder_argument_is_not_an_argument(self):
        self.assertEqual(self.parse("SKILL: web_search | <a web search query>"),
                         ("web_search", self.Q, True))

    def test_reason_needs_no_argument(self):
        self.assertEqual(self.parse("SKILL: reason"), ("reason", "", True))

    def test_menu_restatement_falls_back(self):
        menu = ("SKILL: search_vault | <q>\nSKILL: get_note | <t>\n"
                "SKILL: web_search | <q>\nSKILL: fetch_url | <url>\n"
                "SKILL: reason")
        self.assertEqual(self.parse(menu), ("web_search", self.Q, False))

    def test_unparseable_falls_back_to_web_search(self):
        self.assertEqual(self.parse("Sure! Let me help you with that."),
                         ("web_search", self.Q, False))

    def test_empty_falls_back(self):
        self.assertEqual(self.parse(""), ("web_search", self.Q, False))
        self.assertEqual(self.parse(None), ("web_search", self.Q, False))

    def test_fallback_is_search_vault_when_web_is_off(self):
        # search_vault, not reason: ESCALATION_ORDER[0], and a non-decision on
        # an offline turn must not land on the one skill with no sources.
        self.assertEqual(self.parse("gibberish", web=False),
                         ("search_vault", self.Q, False))

    def test_web_skill_refused_when_web_is_off(self):
        self.assertEqual(self.parse("SKILL: web_search | anything", web=False),
                         ("search_vault", self.Q, False))

    # -- llama3.2:3b failure shapes, each one measured off a real reply ------

    def test_prose_preamble_does_not_become_the_argument(self):
        # "I will use the X skill to ..." is the single most likely 3B reply
        # shape. The tail is narration, so the original question must be used.
        self.assertEqual(
            self.parse("I will use the search_vault skill to find information "
                       "about this."),
            ("search_vault", self.Q, True))
        self.assertEqual(
            self.parse("I'll use the web_search skill to look up the Deaf "
                       "President Now protest."),
            ("web_search", self.Q, True))
        self.assertEqual(self.parse("Let me use the get_note skill for that."),
                         ("get_note", self.Q, True))

    def test_bare_command_forms_keep_their_argument(self):
        # The flip side: no pipe but the line STARTS with the name, so the tail
        # really is the argument and must survive.
        self.assertEqual(self.parse("search_vault deaf president now 1988"),
                         ("search_vault", "deaf president now 1988", True))
        self.assertEqual(self.parse("fetch_url https://example.com/page"),
                         ("fetch_url", "https://example.com/page", True))
        self.assertEqual(self.parse("get_note Deaf President Now"),
                         ("get_note", "Deaf President Now", True))

    def test_bare_word_reason_in_prose_is_not_a_skill_pick(self):
        # \breason\b over prose used to route a direct answer to unsourced
        # recall, which ends the loop and skips the web route entirely.
        self.assertEqual(
            self.parse("Deaf President Now was a 1988 protest. The main reason "
                       "was that Gallaudet had never had a deaf president."),
            ("web_search", self.Q, False))
        self.assertEqual(
            self.parse("The reason I would search the vault is that it is a "
                       "personal research topic."),
            ("web_search", self.Q, False))

    def test_reason_is_still_reachable_by_an_exact_pick(self):
        for text in ("SKILL: reason", "reason", "Reason.", "- SKILL: reason",
                     "1. SKILL: reason"):
            self.assertEqual(self.parse(text)[0], "reason", text)

    def test_prose_first_then_a_good_pipe_line_keeps_the_argument(self):
        # The narrate-then-comply shape: line 1 parses but carries no argument,
        # line 2 is well formed. The piped line wins.
        self.assertEqual(
            self.parse("I will use the search_vault skill.\n"
                       "SKILL: search_vault | deaf president now 1988"),
            ("search_vault", "deaf president now 1988", True))

    def test_two_skills_on_one_line_is_ambiguous(self):
        self.assertEqual(self.parse("use search_vault or web_search"),
                         ("web_search", self.Q, False))

    def test_menu_is_shorter_without_web(self):
        self.assertNotIn("web_search", bs.build_skill_system(web_enabled=False))
        self.assertIn("web_search", bs.build_skill_system(web_enabled=True))


# ---- Note resolution -------------------------------------------------------


class TestGetNoteLookup(unittest.TestCase):

    def test_link_map_hit(self):
        self.assertEqual(bs.resolve_note(META, "Deaf President Now 1988"),
                         "History/Deaf President Now 1988.md")

    def test_alias_hit(self):
        self.assertEqual(bs.resolve_note(META, "dpn"),
                         "History/Deaf President Now 1988.md")

    def test_case_insensitive_stem(self):
        self.assertEqual(bs.resolve_note(META, "article 2 DEFINITIONS.md"),
                         "CRPD/Article 2 Definitions.md")

    def test_wikilink_brackets_stripped(self):
        self.assertEqual(bs.resolve_note(META, "[[dpn]]"),
                         "History/Deaf President Now 1988.md")

    def test_miss(self):
        self.assertEqual(bs.resolve_note(META, "no such note"), "")

    def test_never_raises_on_junk_meta(self):
        self.assertEqual(bs.resolve_note({}, "dpn"), "")
        self.assertEqual(bs.resolve_note(META, ""), "")

    def test_note_text_joins_and_caps(self):
        text = bs.note_text(META, "History/Deaf President Now 1988.md")
        self.assertIn("March 1988", text)
        self.assertIn("I. King Jordan", text)
        capped = bs.note_text(META, "History/Deaf President Now 1988.md", cap=10)
        self.assertEqual(len(capped), 10)


# ---- Disabled flag ---------------------------------------------------------


class TestDisabled(unittest.TestCase):

    def test_disabled_is_a_no_op(self):
        model = FakeModel(["SKILL: reason"])
        prev, bs.SKILLS_ENABLED = bs.SKILLS_ENABLED, False
        try:
            result = bs.answer_with_skills(make_ctx(model), "anything")
        finally:
            bs.SKILLS_ENABLED = prev
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "disabled")
        self.assertEqual(model.calls, [])   # not one model call


# ---- Happy paths, one per skill --------------------------------------------


class TestSkillHappyPaths(unittest.TestCase):
    Q = "when did deaf president now happen"

    def test_search_vault(self):
        def retrieve(query, keep_k):
            self.assertEqual(keep_k, bs.WIDE_KEEP_K)   # wider than phase 1
            return [META["by_id"][0]]
        model = FakeModel(["SKILL: search_vault | dpn protest date",
                           "It began in March 1988 (Deaf President Now 1988)."])
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model, retrieve=retrieve), self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "search_vault")
        self.assertEqual(r.sources, ["History/Deaf President Now 1988.md"])
        self.assertEqual(r.provenance, "")          # vault answers stay clean
        self.assertNotIn("[provenance", r.answer)

    def test_get_note(self):
        model = FakeModel(["SKILL: get_note | dpn",
                           "March 1988 (Deaf President Now 1988)."])
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "get_note")
        self.assertEqual(r.sources, ["History/Deaf President Now 1988.md"])
        self.assertEqual(r.provenance, "")

    def test_web_search(self):
        model = FakeModel(["SKILL: web_search | deaf president now 1988",
                           "The protest ran in March 1988."])
        ctx = make_ctx(model, gather_web=lambda q: [("https://ex.org/a", "March 1988")])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "web_search")
        self.assertEqual(r.sources, ["https://ex.org/a"])
        self.assertTrue(r.answer.endswith(bs.LABEL_WEB_SEARCH))

    def test_fetch_url(self):
        model = FakeModel(["SKILL: fetch_url | https://gallaudet.edu/dpn",
                           "The page says March 1988."])
        ctx = make_ctx(model, fetch_page=lambda u: "DPN, March 1988.")
        with SkillsOn():
            r = bs.answer_with_skills(ctx, self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "fetch_url")
        self.assertEqual(r.sources, ["https://gallaudet.edu/dpn"])
        self.assertTrue(r.answer.endswith(bs.LABEL_WEB_FETCH))

    def test_fetch_url_accepts_a_bare_domain(self):
        model = FakeModel(["SKILL: fetch_url | gallaudet.edu/dpn",
                           "March 1988."])
        seen = []

        def fetch(url):
            seen.append(url)
            return "text"
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model, fetch_page=fetch), self.Q)
        self.assertEqual(seen, ["https://gallaudet.edu/dpn"])
        self.assertTrue(r.ok)

    def test_reason(self):
        model = FakeModel(["SKILL: reason",
                           "From memory, the protest was in 1988."])
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), self.Q)
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "reason")
        self.assertEqual(r.sources, [])
        self.assertTrue(r.answer.endswith(bs.LABEL_REASONED))


# ---- Provenance ------------------------------------------------------------


class TestProvenance(unittest.TestCase):

    def test_every_non_vault_skill_is_labelled(self):
        for skill in ("web_search", "fetch_url", "reason"):
            self.assertTrue(bs.PROVENANCE_LABELS[skill].startswith(
                "\n\n[provenance - not from the vault"))
        for skill in ("search_vault", "get_note"):
            self.assertEqual(bs.PROVENANCE_LABELS[skill], "")

    def test_reasoned_answer_can_never_look_vault_grounded(self):
        model = FakeModel(["SKILL: reason", "Some recalled fact."])
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), "q")
        self.assertIn("model's own general knowledge", r.answer)
        self.assertEqual(r.provenance, bs.LABEL_REASONED)

    def test_label_survives_trailing_whitespace_in_the_answer(self):
        model = FakeModel(["SKILL: reason", "Some recalled fact.\n\n  "])
        with SkillsOn():
            r = bs.answer_with_skills(make_ctx(model), "q")
        self.assertTrue(r.answer.endswith(bs.LABEL_REASONED))


# ---- Budgets ---------------------------------------------------------------


class TestBudgets(unittest.TestCase):
    Q = "unanswerable question"

    def test_step_cap(self):
        # Every selection picks a DIFFERENT vault query (so repeat suppression
        # never fires) and every answer attempt refuses.
        replies = []
        for i in range(10):
            replies += ["SKILL: search_vault | query %d" % i, "INSUFFICIENT"]
        model = FakeModel(replies)
        ctx = make_ctx(model, retrieve=lambda q, k: [META["by_id"][0]])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, self.Q)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "step_cap")
        self.assertEqual(r.steps, bs.MAX_STEPS)
        self.assertEqual(len(r.trace), bs.MAX_STEPS)

    def test_custom_step_cap(self):
        replies = []
        for i in range(10):
            replies += ["SKILL: search_vault | query %d" % i, "INSUFFICIENT"]
        ctx = make_ctx(FakeModel(replies), max_steps=1,
                       retrieve=lambda q, k: [META["by_id"][0]])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, self.Q)
        self.assertEqual(r.steps, 1)

    def test_deadline_before_the_first_call(self):
        ticks = iter([0.0, 999.0])
        model = FakeModel(["SKILL: reason", "text"])
        ctx = make_ctx(model, clock=lambda: next(ticks), deadline_s=45.0)
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "deadline")
        self.assertEqual(r.steps, 0)
        self.assertEqual(model.calls, [])

    def test_deadline_between_selection_and_execution(self):
        # started=0, pre-select check=1, post-select check=999 -> stop before
        # spending the skill's own I/O budget.
        ticks = iter([0.0, 1.0, 999.0])
        model = FakeModel(["SKILL: reason", "text"])
        ctx = make_ctx(model, clock=lambda: next(ticks), deadline_s=45.0)
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "deadline")
        self.assertEqual(r.steps, 1)
        self.assertEqual(len(model.calls), 1)   # selection only, no answer call

    def test_observations_are_capped(self):
        block = bs._observations_block(["x" * 5000] * 20)
        self.assertLessEqual(len(block), bs.OBS_TOTAL_CAP + 4)
        # Newest observation survives, oldest is dropped.
        block = bs._observations_block(["oldest"] + ["pad" * 100] * 20 + ["newest"])
        self.assertIn("newest", block)
        self.assertNotIn("oldest", block)

    def test_selection_options_are_deterministic(self):
        self.assertEqual(bs.SELECT_GEN_OPTIONS["temperature"], 0.0)
        self.assertLessEqual(bs.SELECT_GEN_OPTIONS["num_predict"], 64)

    def test_chat_select_is_preferred_over_chat(self):
        select = FakeModel(["SKILL: reason"])
        answer = FakeModel(["recalled"])
        ctx = make_ctx(answer)
        ctx.chat_select = select
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertTrue(r.ok)
        self.assertEqual(len(select.calls), 1)   # selection went to the temp-0 fn
        self.assertEqual(len(answer.calls), 1)   # the answer did not

    def test_prev_turn_reaches_the_selection_call_only(self):
        # A follow-up ("her other paper") is unroutable without an antecedent,
        # but the answer call must stay a bare [system, user] pair so the
        # injected grounding contract can never be top-truncated away.
        model = FakeModel(["SKILL: reason", "text"])
        ctx = make_ctx(model, prev_turn="user: who is Maartje De Meulder\n"
                                        "assistant: a Deaf sign language "
                                        "researcher")
        with SkillsOn():
            bs.answer_with_skills(ctx, "what about her other paper")
        select_blob = " ".join(m["content"] for m in model.calls[0])
        self.assertIn("Maartje De Meulder", select_blob)
        self.assertIn("Earlier in this conversation", select_blob)
        answer_msgs = model.calls[1]
        self.assertEqual(len(answer_msgs), 2)
        self.assertNotIn("Maartje",
                         " ".join(m["content"] for m in answer_msgs))

    def test_selection_prompt_never_carries_vault_context(self):
        # The measured reason brain.py:2538-2541 drops it from its routing call.
        model = FakeModel(["SKILL: reason", "text"])
        with SkillsOn():
            bs.answer_with_skills(make_ctx(model), "why is the sky blue")
        select_msgs = model.calls[0]
        blob = " ".join(m["content"] for m in select_msgs)
        self.assertNotIn("Vault context", blob)
        self.assertNotIn("[source:", blob)


# ---- Repeat suppression ----------------------------------------------------


class TestRepeatSuppression(unittest.TestCase):

    def test_identical_repeat_is_substituted(self):
        # The model stubbornly re-picks the same vault query. Step 2 must not
        # re-run it; it escalates to the next untried skill instead - which is
        # now connections (ESCALATION_ORDER: search_vault, connections,
        # web_search, reason). "q" resolves to no vault note, so connections
        # fails without a model call of its own and step 3 escalates again,
        # to web_search.
        replies = ["SKILL: search_vault | same query", "INSUFFICIENT",
                   "SKILL: search_vault | same query",
                   "SKILL: search_vault | same query", "web answer"]
        model = FakeModel(replies)
        ctx = make_ctx(model, retrieve=lambda q, k: [META["by_id"][0]],
                       gather_web=lambda q: [("https://ex.org/a", "text")])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertEqual([t["skill"] for t in r.trace][:3],
                         ["search_vault", "connections", "web_search"])
        self.assertEqual(r.skill, "web_search")

    def test_exhausted_when_every_skill_is_spent(self):
        # Web off, so search_vault, connections, and reason are reachable
        # (web_search/fetch_url are not). All three fail, the model keeps
        # repeating, and the loop stops rather than spinning. "q" resolves to
        # no vault note, so connections fails without spending a model call.
        replies = ["SKILL: search_vault | q", "INSUFFICIENT",
                   "SKILL: search_vault | q",
                   "SKILL: search_vault | q", "",
                   "SKILL: search_vault | q"]
        ctx = make_ctx(FakeModel(replies), web_enabled=False, max_steps=5,
                       retrieve=lambda q, k: [META["by_id"][0]])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "exhausted")

    def test_case_and_spacing_differences_still_count_as_a_repeat(self):
        replies = ["SKILL: search_vault | Same  Query", "INSUFFICIENT",
                   "SKILL: search_vault | same query", "web answer"]
        model = FakeModel(replies)
        ctx = make_ctx(model, retrieve=lambda q, k: [META["by_id"][0]],
                       gather_web=lambda q: [("https://ex.org/a", "t")])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertEqual(r.skill, "web_search")


# ---- Degradation: never raise into the caller ------------------------------


class TestNeverRaises(unittest.TestCase):

    def test_missing_dependencies_become_observations(self):
        # No retrieve, no gather_web, no fetch_page: each skill reports its
        # absence and the loop moves on instead of exploding.
        replies = ["SKILL: search_vault | q", "SKILL: web_search | q",
                   "SKILL: fetch_url | https://ex.org", "unused"]
        ctx = make_ctx(FakeModel(replies))
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertFalse(r.ok)
        self.assertEqual(r.steps, bs.MAX_STEPS)
        self.assertTrue(all("observation" in t for t in r.trace))

    def test_a_crashing_skill_is_absorbed(self):
        def boom(query, keep_k):
            raise RuntimeError("index on fire")
        replies = ["SKILL: search_vault | q", "SKILL: reason", "recalled"]
        ctx = make_ctx(FakeModel(replies), retrieve=boom)
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertTrue(r.ok)
        self.assertEqual(r.skill, "reason")
        self.assertIn("index on fire", r.trace[0]["observation"])

    def test_a_crashing_model_returns_no_answer(self):
        def boom(model, messages):
            raise RuntimeError("ollama down")
        ctx = make_ctx(boom, retrieve=lambda q, k: [META["by_id"][0]],
                       gather_web=lambda q: [])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertFalse(r.ok)          # brain.py falls through to its web path

    def test_model_error_string_is_treated_as_unparseable(self):
        # chat_once's '[error] ...' convention (brain.py:1358) must not become
        # a get_note argument.
        replies = ["[error] could not reach the model: boom", "web answer"]
        ctx = make_ctx(FakeModel(replies),
                       gather_web=lambda q: [("https://ex.org/a", "t")])
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertEqual(r.skill, "web_search")
        self.assertFalse(r.trace[0]["parsed"])


# ---- The injected grounding contract ---------------------------------------


class TestGroundingContract(unittest.TestCase):

    def test_injected_vault_system_is_used_verbatim(self):
        contract = "PRETEND THIS IS VAULT_FIRST_SYSTEM, verbatim."
        model = FakeModel(["SKILL: get_note | dpn", "answer"])
        ctx = make_ctx(model)
        ctx.vault_system = contract
        with SkillsOn():
            bs.answer_with_skills(ctx, "q")
        self.assertEqual(model.calls[1][0]["content"], contract)

    def test_injected_is_insufficient_decides_the_retry(self):
        # brain._is_insufficient also catches free-form refusals; the loop must
        # honour the injected version, not just the bare sentinel.
        model = FakeModel(["SKILL: get_note | dpn", "I can't answer that.",
                           "SKILL: reason", "recalled"])
        ctx = make_ctx(model)
        ctx.is_insufficient = lambda t: "can't answer" in (t or "").lower()
        with SkillsOn():
            r = bs.answer_with_skills(ctx, "q")
        self.assertEqual(r.skill, "reason")


if __name__ == "__main__":
    unittest.main(verbosity=2)
