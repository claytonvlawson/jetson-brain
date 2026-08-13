#!/usr/bin/env python3
"""test_intent.py - question-intent classification, query expansion, and
multi-clause decomposition (brain.py).

These three make short factual questions answerable at all. Before them,
"Where is Gallaudet University?" and "What is Gallaudet University?" produced
byte-identical retrieval (both reduce to {gallaudet, university}) and both got
the base prompt's "about 5 to 8 sentences", so a request for a city returned an
essay. Run: python3 test_intent.py
"""

import os
import sys

# brain.py exits at import if it cannot find an index, and it does not need one
# for the pure functions under test.
os.environ.setdefault("BRAIN_INDEX", "/dev/null")

import brain  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append("%s\n    got:  %r\n    want: %r" % (label, got, want))


def check_true(label, cond, detail=""):
    if not cond:
        FAILURES.append("%s%s" % (label, ("\n    " + detail) if detail else ""))


# -- classify_intent ---------------------------------------------------------
# Short + concrete answer-type -> factual_short.
for q in ("Where is Gallaudet University?",
          "When was Gallaudet University founded?",
          "Who wrote the Deaf Tax paper?",
          "How many students attend MSSD?"):
    check("factual_short: " + q, brain.classify_intent(q)[0], "factual_short")

# "what"/"why"/"how" ask for an explanation, not a fact. They must NOT be
# shortened, or every conceptual question collapses to one sentence.
for q in ("What is language deprivation?",
          "Why does early language access matter?",
          "How does a cochlear implant affect language development?"):
    check("analytic: " + q, brain.classify_intent(q)[0], "analytic")

# A concrete answer-type carried by a long, discursive question is still
# analytic: the asker wants the reasoning, not just the noun.
long_where = ("Where in the research literature would I look to understand how "
              "deaf children of hearing parents end up without a first language "
              "during the years that matter most for acquisition?")
check("long 'where' stays analytic", brain.classify_intent(long_where)[0], "analytic")

# Several clauses -> multi, regardless of the leading question word.
multi_q = ("Who founded Gallaudet University and in what year, and what law "
           "made it possible?")
check("multi-clause", brain.classify_intent(multi_q)[0], "multi")

# Degenerate input must never raise.
for bad in (None, "", "   ", "???", 12345):
    try:
        intent, wh = brain.classify_intent(bad)
        check_true("classify_intent(%r) returns a str intent" % (bad,),
                   isinstance(intent, str) and isinstance(wh, str))
    except Exception as exc:
        FAILURES.append("classify_intent(%r) raised %r" % (bad, exc))


# -- intent_boost_terms ------------------------------------------------------
base = brain.query_tokens("Where is Gallaudet University?")
terms = brain.intent_boost_terms(base, "where")
check_true("'where' yields locative terms", "located" in terms and "city" in terms,
           "terms=%r" % (terms,))
check_true("boost terms are disjoint from the query",
           not (set(terms) & set(base)), "terms=%r base=%r" % (terms, base))

# A term already in the query must not be re-added: merged into the bag it
# would double that term's BM25 weight.
dup = brain.intent_boost_terms(brain.query_tokens("Where is the campus located?"),
                               "where")
check_true("already-present terms are excluded",
           "located" not in dup and "campus" not in dup, "dup=%r" % (dup,))

# An intent with no expansion table yields nothing.
check("unknown wh yields no terms", brain.intent_boost_terms(base, "why"), ())
check("empty wh yields no terms", brain.intent_boost_terms(base, ""), ())


# -- _apply_intent_boost is GATED --------------------------------------------
# The load-bearing property. Merging these terms into the query bag instead
# measurably dropped the Gallaudet note out of the top 6 for "Where is
# Gallaudet University?", because address-dense notes outscored it on the added
# terms alone. The boost must never promote a chunk the query did not match.
class _FakeBM25(object):
    def __init__(self, boost):
        self._boost = boost

    def get_scores(self, terms):
        return list(self._boost)


scores = [5.0, 0.0, 0.0, 2.0]
boosted = brain._apply_intent_boost(list(scores), _FakeBM25([1.0, 99.0, 0.0, 1.0]),
                                    ("located", "city"))
check_true("an unmatched chunk stays unmatched despite a huge boost",
           boosted[1] == 0.0, "boosted=%r" % (boosted,))
check_true("a matched chunk is lifted", boosted[0] > scores[0])
check_true("the lift is a fraction, not a replacement",
           boosted[0] < scores[0] + 99.0)
check("no terms is a no-op", brain._apply_intent_boost(list(scores),
                                                       _FakeBM25([9.0] * 4), ()),
      scores)
# A length-mismatched or raising scorer must degrade to the input, not crash.
check("mismatched boost length is ignored",
      brain._apply_intent_boost(list(scores), _FakeBM25([1.0]), ("located",)), scores)


# -- _apply_card_boost -------------------------------------------------------
card_chunks = [{"card": True}, {}, {"card": True}]
cs = brain._apply_card_boost([4.0, 4.0, 0.0], card_chunks, "factual_short")
check_true("a card is preferred for fact questions", cs[0] > cs[1])
check_true("an unmatched card is not promoted", cs[2] == 0.0)
check("analytic questions leave prose alone",
      brain._apply_card_boost([4.0, 4.0, 0.0], card_chunks, "analytic"),
      [4.0, 4.0, 0.0])


# -- entity_subqueries -------------------------------------------------------
ENT_CHUNKS = [{"title": "Gallaudet University"},
              {"title": "Maryland School for the Deaf"},
              {"title": "Black ASL"}]
cmp_q = ("Gallaudet University and the Maryland School for the Deaf were both "
         "founded in the 1860s. Which came first?")
ent = brain.entity_subqueries(brain.query_tokens(cmp_q), ENT_CHUNKS)
check("two named notes yield two sub-queries", len(ent), 2)
by_entity = {"gallaudet" in e: e for e in ent}
check_true("each sub-query names exactly one entity",
           all(("gallaudet" in e) != ("maryland" in e) for e in ent),
           "ent=%r" % (ent,))
check_true("each sub-query keeps the shared predicate",
           all("founded" in e for e in ent), "ent=%r" % (ent,))
check_true("an unnamed note contributes no sub-query",
           not any("asl" in e for e in ent), "ent=%r" % (ent,))

# A single named entity is ordinary retrieval, not a comparison.
check("one named note yields nothing",
      brain.entity_subqueries(brain.query_tokens(
          "When was Gallaudet University founded and why does it matter?"),
          ENT_CHUNKS), [])
# A partial title overlap must not count as naming the note.
check("partial title overlap does not name a note",
      brain.entity_subqueries(brain.query_tokens(
          "What school serves deaf children in Maryland and elsewhere?"),
          ENT_CHUNKS), [])
check("empty query yields nothing", brain.entity_subqueries([], ENT_CHUNKS), [])
check("empty corpus yields nothing",
      brain.entity_subqueries(brain.query_tokens(cmp_q), []), [])


# -- _apply_title_boost ------------------------------------------------------
# Symmetric overlap: a one-word title must not capture every query using that
# word, which is what makes this safe to apply to every chunk.
tb_chunks = [{"title": "Gallaudet University"},
             {"title": "Summer Camps for Deaf Children DC Maryland"}]
tb = brain._apply_title_boost([10.0, 10.0], tb_chunks,
                              brain.query_tokens("Where is Gallaudet University?"))
check_true("the note ABOUT the entity outranks one that merely mentions it",
           tb[0] > tb[1], "tb=%r" % (tb,))
check("a chunk with no score is never promoted",
      brain._apply_title_boost([0.0], [{"title": "Gallaudet University"}],
                               brain.query_tokens("Where is Gallaudet University?")),
      [0.0])


# -- decompose ---------------------------------------------------------------
# BRAIN_DECOMPOSE now defaults OFF (it measured worse: in_context 31/40 vs
# 29/40 on retrieval_gold, replicated on two indexes). These checks are about
# the CORRECTNESS of decompose() when it is asked to run, so force the flag on
# rather than depending on whichever way the default currently points -- the
# original version depended on it, and silently became a no-op assertion the
# moment the default flipped.
check_true("decompose is OFF by default", not brain.DECOMPOSE_ENABLED,
           "BRAIN_DECOMPOSE=%r" % (os.environ.get("BRAIN_DECOMPOSE"),))
_saved_decompose = brain.DECOMPOSE_ENABLED
brain.DECOMPOSE_ENABLED = True

subqs = brain.decompose(multi_q)
check_true("multi-clause decomposes", len(subqs) >= 2, "subqs=%r" % (subqs,))

# THE load-bearing property. Every sub-query must carry the first clause's
# entity terms. An unanchored trailing clause ("in what year" -> {year}) is a
# one-term query that retrieves by that term alone, which is precisely the
# decoy problem decomposition exists to fix.
anchor = set(brain.query_tokens("Who founded Gallaudet University"))
for sub in subqs:
    check_true("sub-query is anchored: %r" % (sub,), anchor.issubset(set(sub)))

# Each sub-query must add something past the anchor, or it is a duplicate pass.
for sub in subqs[1:]:
    check_true("sub-query adds terms: %r" % (sub,), len(set(sub)) > len(anchor))

check_true("sub-queries are capped", len(subqs) <= brain.DECOMPOSE_MAX_SUBQ)

# Single-intent questions must not pay for extra BM25 passes.
for q in ("Where is Gallaudet University?",
          "What is language deprivation?",
          "How does language deprivation affect reading outcomes for deaf children?"):
    check("no decomposition: " + q[:44], brain.decompose(q), [])

for bad in (None, "", "and and and", "?"):
    try:
        check_true("decompose(%r) returns a list" % (bad,),
                   isinstance(brain.decompose(bad), list))
    except Exception as exc:
        FAILURES.append("decompose(%r) raised %r" % (bad, exc))

# With the flag off, decompose must short-circuit to [] no matter the question.
brain.DECOMPOSE_ENABLED = False
check("decompose disabled returns []", brain.decompose(multi_q), [])
brain.DECOMPOSE_ENABLED = _saved_decompose


# -- _shape_for_intent -------------------------------------------------------
# The GROUNDING CONTRACT is load-bearing for three separate mechanisms
# (_is_insufficient, faithfulness_flags, chat_stream_holdback), so every clause
# they depend on must survive shaping verbatim.
#
# Shaping used to be append-only and this block asserted startswith(base). That
# was too strong a proxy for the real invariant: it also froze the base
# prompt's trailing LENGTH sentence, which is exactly the text that needed to
# change, and so forced factual_short to be phrased as "OVERRIDE the length
# guidance above" -- a direct self-contradiction handed to a 3B model in
# adjacent sentences. The length sentence is substituted now. What must not
# change is the contract, so assert the contract itself, not a prefix.
CONTRACT_CLAUSES = (
    "reply with EXACTLY the single word INSUFFICIENT",
    "Cite the sources by filename when you use them.",
    "Do not guess, do not use outside knowledge",
    "Use only the provided context.",
)
for base_name, base in (("vault", brain.VAULT_FIRST_SYSTEM),
                        ("parent", brain.PARENT_SYSTEM)):
    for intent in ("factual_short", "multi", "analytic", "nonsense"):
        shaped = brain._shape_for_intent(base, intent)
        for clause in CONTRACT_CLAUSES:
            check_true("%s/%s keeps: %s" % (base_name, intent, clause[:38]),
                       clause in shaped)

# The contradiction must be GONE: no shaped prompt may state a sentence range
# and then instruct the model to override it.
for base in (brain.VAULT_FIRST_SYSTEM, brain.PARENT_SYSTEM):
    for intent in ("factual_short", "multi", "analytic"):
        check_true("no OVERRIDE meta-instruction (%s)" % intent,
                   "OVERRIDE the length guidance"
                   not in brain._shape_for_intent(base, intent))
    short = brain._shape_for_intent(base, "factual_short")
    check_true("factual_short drops the long-answer target",
               "5 to 8 sentences" not in short and "7 to 12 sentences" not in short,
               repr(short[-160:]))
    check_true("factual_short states the short target exactly once",
               short.count("FIRST sentence") == 1)

# Non-length intents still append, so the base survives as a prefix there.
for intent in ("multi", "analytic", "nonsense"):
    check_true("%s still appends" % intent,
               brain._shape_for_intent(brain.VAULT_FIRST_SYSTEM, intent)
               .startswith(brain.VAULT_FIRST_SYSTEM))

# If the base length sentence is ever reworded, substitution cannot match and
# must DEGRADE TO APPENDING rather than silently dropping the instruction.
check_true("unmatched base still receives the directive",
           "FIRST sentence" in brain._shape_for_intent(
               "A prompt with no length sentence.", "factual_short"))
check("analytic is unchanged",
      brain._shape_for_intent(brain.VAULT_FIRST_SYSTEM, "analytic"),
      brain.VAULT_FIRST_SYSTEM)
check_true("parent persona also shapes",
           "EVERY part" in brain._shape_for_intent(brain.PARENT_SYSTEM, "multi"))
check_true("parent persona keeps its warmth under shaping",
           "warm" in brain._shape_for_intent(brain.PARENT_SYSTEM, "factual_short"))
# The parent rule most worth losing sleep over if shaping ever rewrites text.
check_true("parent persona never loses its no-blame rule",
           "Never shame or blame a parent"
           in brain._shape_for_intent(brain.PARENT_SYSTEM, "factual_short"))


if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_intent: all checks passed")
