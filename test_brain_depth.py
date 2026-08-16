#!/usr/bin/env python3
"""Tests for brain_depth.py - the pure half of the two-pass DEPTH answer.

No Ollama, no index, no network, and above all no brain.py: this module exists
so the depth path can be checked on a laptop, and a hygiene test below asserts
that property in the source text rather than trusting it.

The citation tests are the load-bearing ones. build_index is stdlib-only, so the
real frontmatter parser runs here against real note shapes, including the Gulati
trap: a note with no frontmatter doi: whose body carries 18 other papers' DOIs.

Run: python3 -m unittest test_brain_depth -v
"""

import os
import unittest

import brain_depth as bd
import build_index as bi


HERE = os.path.dirname(os.path.abspath(__file__))


def read_frontmatter_from(notes):
    """Production-shaped reader over an in-memory {source: full note text} map:
    split the frontmatter, parse it, hand back the fields. The body is parsed
    and then dropped, exactly as the real reader does."""
    def _read(source):
        fm, _body = bi.split_frontmatter(notes.get(source, ""))
        return bi.parse_frontmatter_fields(fm)
    return _read


# A shortened but shape-accurate Gulati 2018 note: no doi: in the frontmatter,
# and Hall/Levin/Anderson 2017's DOI sitting in the body reference list.
GULATI = """---
name: Gulati 2018 - Language Deprivation Syndrome
type: source
year: 2018
authors: [Sanjay Gulati]
venue: "Chapter 1 in Glickman and Hall (Eds.), Language Deprivation and Deaf Mental Health, Routledge"
---

# Gulati 2018: Language Deprivation Syndrome

Gulati names Language Deprivation Syndrome.

## References
- Hall, W. C., Levin, L. L., & Anderson, M. L. (2017). Language deprivation
  syndrome. Social Psychiatry and Psychiatric Epidemiology. doi: 10.1007/s00127-017-1351-7
- Humphries et al. (2012). Reversing the historical error. doi: 10.1016/j.harmed.2012.01.001
"""

HALL2017 = """---
name: Hall, Levin, Anderson 2017, Language Deprivation Syndrome
type: source
authors: [Wyatte C. Hall, Leonard L. Levin, Melissa L. Anderson]
year: 2017
venue: Social Psychiatry and Psychiatric Epidemiology 52(6)
doi: 10.1007/s00127-017-1351-7
url: https://link.springer.com/article/10.1007/s00127-017-1351-7
---

# Hall, Levin, and Anderson 2017
"""

CORE = """---
name: Critical Period and Brain
type: core-topic
tags: [core, language-deprivation]
---

# Critical Period and Brain
"""

GULATI_SRC = "Sources/Language Deprivation/Gulati 2018 - Language Deprivation Syndrome.md"
HALL_SRC = "Sources/Language Deprivation/Hall Levin Anderson 2017 - Language Deprivation Syndrome.md"
CORE_SRC = "LanguageDeprivationAgent/Core/Critical Period and Brain.md"

NOTES = {GULATI_SRC: GULATI, HALL_SRC: HALL2017, CORE_SRC: CORE}

HUB = "LanguageDeprivationAgent/Core/What Is Language Deprivation.md"
LINKS_OUT = {
    HUB: [
        "LanguageDeprivationAgent/Core/Critical Period and Brain.md",
        "LanguageDeprivationAgent/Core/Mental Health and Public Health.md",
        "LanguageDeprivationAgent/Core/Cochlear Implant Pathways and Risk.md",
        GULATI_SRC,
    ],
    GULATI_SRC: [HUB, "Concepts/Milan 1880 Congress.md"],
}


class TestHygiene(unittest.TestCase):
    """The whole point of the module is that it imports on a machine that
    cannot import brain.py. Assert it in the source, not in a comment."""

    def test_no_brain_import(self):
        with open(os.path.join(HERE, "brain_depth.py"), encoding="utf-8") as fh:
            src = fh.read()
        for line in src.split("\n"):
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import brain"), stripped)
            self.assertFalse(stripped.startswith("from brain"), stripped)


class TestGraphFacets(unittest.TestCase):

    def test_hub_neighbours_become_facets(self):
        facets = bd.graph_facets(LINKS_OUT, HUB, [HUB])
        self.assertEqual([f["source"] for f in facets], LINKS_OUT[HUB])
        self.assertEqual(facets[0]["label"], "Critical Period and Brain")

    def test_covered_sources_are_excluded(self):
        facets = bd.graph_facets(LINKS_OUT, HUB, [HUB, CORE_SRC, GULATI_SRC])
        srcs = [f["source"] for f in facets]
        self.assertNotIn(CORE_SRC, srcs)
        self.assertNotIn(GULATI_SRC, srcs)
        self.assertIn("LanguageDeprivationAgent/Core/Mental Health and Public Health.md", srcs)

    def test_second_seed_contributes_and_dedups(self):
        facets = bd.graph_facets(LINKS_OUT, HUB, [HUB, GULATI_SRC])
        srcs = [f["source"] for f in facets]
        self.assertIn("Concepts/Milan 1880 Congress.md", srcs)
        self.assertEqual(len(srcs), len(set(srcs)))
        self.assertNotIn(HUB, srcs)

    def test_source_label_drops_the_author_year_prefix(self):
        self.assertEqual(bd.facet_label(GULATI_SRC), "Language Deprivation Syndrome")

    def test_degenerate_input_never_raises(self):
        self.assertEqual(bd.graph_facets(None, None, None), [])
        self.assertEqual(bd.graph_facets({}, HUB, []), [])


class TestRankFacets(unittest.TestCase):

    def setUp(self):
        self.cands = bd.graph_facets(LINKS_OUT, HUB, [HUB])

    def test_picks_what_the_draft_does_not_cover(self):
        draft = ("Language deprivation harms the critical period and the brain. "
                 "It is a mental health and public health problem.")
        picked = bd.rank_facets(self.cands, draft, "what is language deprivation", limit=1)
        self.assertEqual(picked[0]["label"], "Cochlear Implant Pathways and Risk")

    def test_limit_is_honoured(self):
        self.assertEqual(len(bd.rank_facets(self.cands, "draft", "q", limit=2)), 2)
        self.assertEqual(bd.rank_facets(self.cands, "draft", "q", limit=0), [])
        self.assertEqual(bd.rank_facets([], "draft", "q", limit=3), [])

    def test_question_overlap_breaks_ties(self):
        draft = "Nothing in this draft mentions any facet at all."
        picked = bd.rank_facets(self.cands, draft, "cochlear implant pathways and risk", limit=1)
        self.assertEqual(picked[0]["label"], "Cochlear Implant Pathways and Risk")

    def test_ranking_is_deterministic(self):
        a = bd.rank_facets(self.cands, "draft text", "q", limit=3)
        b = bd.rank_facets(self.cands, "draft text", "q", limit=3)
        self.assertEqual([f["source"] for f in a], [f["source"] for f in b])


class TestNewChunks(unittest.TestCase):

    def chunks(self, ids):
        return [{"id": i, "source": "N%d.md" % i, "text": "t"} for i in ids]

    def test_drops_seen_ids_and_caps(self):
        got = bd.new_chunks(self.chunks([1, 2, 3, 4]), {2}, 2)
        self.assertEqual([c["id"] for c in got], [1, 3])

    def test_dedups_within_the_batch(self):
        got = bd.new_chunks(self.chunks([5, 5, 6]), set(), 5)
        self.assertEqual([c["id"] for c in got], [5, 6])

    def test_idless_and_empty_inputs(self):
        self.assertEqual(bd.new_chunks([{"source": "x.md"}], set(), 3), [])
        self.assertEqual(bd.new_chunks(None, None, 3), [])
        self.assertEqual(bd.new_chunks(self.chunks([1]), set(), 0), [])


class TestMerge(unittest.TestCase):

    def test_joins_both_passes(self):
        self.assertEqual(bd.merge("A.", "B."), "A.\n\nB.")

    def test_empty_expansion_keeps_the_draft(self):
        self.assertEqual(bd.merge("A.", ""), "A.")
        self.assertEqual(bd.merge("A.", None), "A.")

    def test_empty_draft_keeps_the_expansion(self):
        self.assertEqual(bd.merge("", "B."), "B.")

    def test_repeated_paragraph_is_dropped(self):
        self.assertEqual(bd.merge("A.\n\nB.", "B.\n\nC."), "A.\n\nB.\n\nC.")
        self.assertEqual(bd.merge("A.", "A."), "A.")


class TestCitationFooter(unittest.TestCase):

    def setUp(self):
        self.read = read_frontmatter_from(NOTES)

    def test_gulati_does_not_inherit_hall_2017_doi(self):
        # The trap: 10.1007/s00127-017-1351-7 is in Gulati's BODY and belongs to
        # Hall/Levin/Anderson 2017. Frontmatter-only construction makes it
        # structurally unreachable.
        footer = bd.citation_footer([GULATI_SRC], self.read)
        self.assertIn("Sanjay Gulati", footer)
        self.assertIn("2018", footer)
        self.assertNotIn("10.1007/s00127-017-1351-7", footer)
        self.assertNotIn("doi:", footer)

    def test_a_real_frontmatter_doi_is_emitted(self):
        footer = bd.citation_footer([HALL_SRC], self.read)
        self.assertIn("doi:10.1007/s00127-017-1351-7", footer)

    def test_core_topic_degrades_to_a_synthesis_note(self):
        footer = bd.citation_footer([CORE_SRC], self.read)
        self.assertIn(bd.SYNTHESIS_LABEL, footer)
        self.assertIn("Critical Period and Brain", footer)
        self.assertNotIn("doi", footer)

    def test_no_urls_reach_the_footer(self):
        # Hall 2017 HAS a frontmatter url:. Emitting it would make
        # faithfulness_flags treat our own verified citation as a fabrication.
        footer = bd.citation_footer([HALL_SRC, GULATI_SRC, CORE_SRC], self.read)
        self.assertNotIn("http", footer)
        self.assertNotIn("link.springer.com", footer)

    def test_sentinel_leads_the_block_and_sources_dedup(self):
        footer = bd.citation_footer([HALL_SRC, HALL_SRC], self.read)
        self.assertTrue(footer.startswith(bd.CITE_SENTINEL))
        self.assertEqual(footer.count("Wyatte C. Hall"), 1)

    def test_empty_and_failing_reader_degrade_quietly(self):
        self.assertEqual(bd.citation_footer([], self.read), "")

        def boom(_src):
            raise RuntimeError("no such file")

        footer = bd.citation_footer([CORE_SRC], boom)
        # Unknown frontmatter still yields an honest, non-broken line.
        self.assertIn(bd.SYNTHESIS_LABEL, footer)

    def test_strip_citations_removes_the_footer(self):
        answer = "Body text." + bd.citation_footer([HALL_SRC], self.read)
        self.assertEqual(bd.strip_citations(answer), "Body text.")
        self.assertEqual(bd.strip_citations("no footer here"), "no footer here")
        self.assertEqual(bd.strip_citations(""), "")


class TestFacetCoverage(unittest.TestCase):

    def setUp(self):
        self.facets = bd.graph_facets(LINKS_OUT, HUB, [HUB])

    def test_uncovered_text_scores_zero(self):
        self.assertEqual(bd.facet_coverage("Deaf schools began in 1817.", self.facets), 0.0)

    def test_fully_covered_text_scores_one(self):
        text = ("The critical period and brain evidence, mental health and public "
                "health outcomes, cochlear implant pathways and risk, and language "
                "deprivation syndrome are all covered here.")
        self.assertEqual(bd.facet_coverage(text, self.facets), 1.0)

    def test_partial_coverage_is_a_fraction(self):
        cov = bd.facet_coverage("The critical period and brain matter most.", self.facets)
        self.assertGreater(cov, 0.0)
        self.assertLess(cov, 1.0)
        self.assertAlmostEqual(cov, 1.0 / len(self.facets))

    def test_footer_names_do_not_count_as_coverage(self):
        # A bare citation list must not score as discussion of those facets.
        read = read_frontmatter_from(NOTES)
        answer = "Short body." + bd.citation_footer([CORE_SRC], read)
        self.assertEqual(bd.facet_coverage(answer, self.facets), 0.0)

    def test_empty_inputs(self):
        self.assertEqual(bd.facet_coverage("anything", []), 0.0)
        self.assertEqual(bd.facet_coverage("", self.facets), 0.0)
        self.assertEqual(bd.facet_coverage(None, self.facets), 0.0)

    def test_plain_string_facets_are_accepted(self):
        # The scorer feeds source paths straight in, with no graph handy.
        cov = bd.facet_coverage("Cochlear implant pathways and risk are real.",
                                ["LanguageDeprivationAgent/Core/Cochlear Implant Pathways and Risk.md"])
        self.assertEqual(cov, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
