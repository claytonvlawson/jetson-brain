#!/usr/bin/env python3
"""test_note_card.py - definition-card chunks, callout cleaning, and generalized
frontmatter capture (build_index.py).

The card exists because BM25 length-normalizes: a short answer-dense doc
outranks the same terms diluted across a long one. FLOOR_WORDS=90 merges a
25-word one-line summary into a 250-word prose blob, which is the exact chunk a
short factual question needs and the exact chunk merging destroys.

Run: python3 test_note_card.py
"""

import sys

import build_index as bi

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append("%s\n    got:  %r\n    want: %r" % (label, got, want))


def check_true(label, cond, detail=""):
    if not cond:
        FAILURES.append("%s%s" % (label, ("\n    " + detail) if detail else ""))


# A note in this vault's house style: identity in frontmatter, the answer in an
# [!abstract] callout, and the prose that would otherwise swallow it.
NOTE = """---
name: Gallaudet University
type: org
tags: [org, academic, us]
status: active
founded: 1864
---

# Gallaudet University

> [!abstract] One-line
> Federally chartered Deaf university in Washington DC, the principal US
> academic institution for Deaf studies and ASL linguistics.

## Role in the thesis

- **Research locus**: home of the Gallaudet Research Institute and multiple labs.
- **Dataset context**: signing data collected here carries institutional weight.
"""

fm, body = bi.split_frontmatter(NOTE)
fields = bi.parse_frontmatter_fields(fm)
title = bi.extract_title(body, "Gallaudet University")
card = bi.note_card(fields, title, "Gallaudet University", body)


# -- frontmatter capture -----------------------------------------------------
# The whole point: "founded: 1864" is not in _FM_SCALAR_KEYS, and before the
# extras pass it was silently dropped, leaving the vault's own founding year
# unreachable by any query.
check("founded reaches the parser", fields.get("extra", {}).get("founded"), "1864")
check("known scalars still parse", fields.get("type"), "org")
check("lists still parse", fields.get("tags"), ["org", "academic", "us"])
check_true("bookkeeping keys are dropped",
           "status" not in fields.get("extra", {}), "extra=%r" % (fields.get("extra"),))

# fm_meta_sentence must keep its original 7-field output. It is injected into
# real prose chunks, so widening it would rewrite every chunk in the vault.
sentence = bi.fm_meta_sentence(fields)
check_true("meta sentence ignores extras", "1864" not in sentence, sentence)


# -- the card ----------------------------------------------------------------
check_true("a card is produced", bool(card))
body_only = card.split("\n", 1)[1]
for term in ("Gallaudet University", "org", "1864", "Washington DC"):
    check_true("card carries %r" % term, term in card, card)
check_true("card is short enough to win on length",
           len(body_only.split()) <= bi.CARD_MAX_WORDS,
           "%d words" % len(body_only.split()))
check_true("card carries a breadcrumb", card.startswith("["))
check_true("card prefers the callout body over the callout TITLE",
           "One-line" not in card, card)
check_true("card drops the callout type marker",
           "[!abstract]" not in card and "abstract" not in card.lower(), card)

# Field order must be stable across rebuilds or every card churns in the
# unchanged-note comparison that incremental updates depend on.
check("card is deterministic", bi.note_card(fields, title, "Gallaudet University",
                                            body), card)


# -- definition extraction ---------------------------------------------------
check("callout body wins",
      bi.extract_definition(body, title),
      "Federally chartered Deaf university in Washington DC, the principal US "
      "academic institution for Deaf studies and ASL linguistics.")

no_callout = "# Thing\n\nA thing is a kind of object. It has properties.\n"
check("falls back to the first sentences after the H1",
      bi.extract_definition(no_callout, "Thing"),
      "A thing is a kind of object. It has properties.")

check("a [!warning] is not a definition",
      bi.extract_definition("# X\n\n> [!warning] Careful\n> Do not do this.\n", "X"),
      "Do not do this.")

check("open tasks are not definitions",
      bi.extract_definition("# X\n\n- [ ] todo item\n\nX is a real thing.\n", "X"),
      "X is a real thing.")
check("empty note yields no definition", bi.extract_definition("", "X"), "")

# Wikilinks must be unwrapped, or the card indexes bracket syntax.
check("wikilinks are cleaned in definitions",
      bi.extract_definition("# X\n\nSee [[Language Deprivation|LD]] for context.\n", "X"),
      "See LD for context.")


# -- cards must not be emitted when they would be noise ----------------------
check("a bare title alone is not a card", bi.note_card({}, "Daily Note", "2026-04-27", ""), "")
check("an empty note yields no card", bi.note_card({}, "", "", ""), "")

for bad_fm, bad_body in (({}, None), (None, "x"), ({"tags": None}, "# X\ntext")):
    try:
        bi.note_card(bad_fm or {}, "T", "T", bad_body or "")
    except Exception as exc:
        FAILURES.append("note_card(%r, %r) raised %r" % (bad_fm, bad_body, exc))


# -- chunk assembly ----------------------------------------------------------
chunks, injected, has_card = bi.build_note_chunks(body, fm, title, "Gallaudet University", "inline")
check_true("the card leads the chunk list", chunks and chunks[0] == card)
check_true("prose chunks survive alongside it", len(chunks) >= 2,
           "%d chunks" % len(chunks))
check_true("frontmatter meta still injects", injected)
check_true("the card is flagged for retrieval", has_card)

# The card must NOT absorb the meta sentence: apply_meta targets chunk_texts[0],
# so inserting the card first would put the meta on the card (which already
# carries those fields) and starve the first prose chunk that needs it.
check_true("meta sentence lands on prose, not the card",
           "Tags:" in chunks[1] and sentence in chunks[1],
           "chunks[1]=%r" % (chunks[1][:200],))

# Every chunk must tokenize, or build() silently drops it.
for i, c in enumerate(chunks):
    check_true("chunk %d tokenizes" % i, bool(bi.tokenize(c)))

# Parity with update_index: both callers go through build_note_chunks, so a full
# build and an incremental update produce identical chunks for the same note.
again, _, _ = bi.build_note_chunks(body, fm, title, "Gallaudet University", "inline")
check("build_note_chunks is deterministic", again, chunks)

# A frontmatter-only note still gets a card, so its identity stays retrievable.
fm2, body2 = bi.split_frontmatter("---\nname: VL2 Lab\ntype: org\nfounded: 2006\n---\n")
only, _, _ = bi.build_note_chunks(body2, fm2, "VL2 Lab", "VL2 Lab", "inline")
check_true("frontmatter-only note still indexes", bool(only))
check_true("...and carries its year", any("2006" in c for c in only), repr(only))


# -- callout cleaning does not damage ordinary prose --------------------------
check("plain prose is untouched", bi.clean_obsidian("A normal sentence."),
      "A normal sentence.")
check("blockquote markers are stripped", bi.clean_obsidian("> quoted text"), "quoted text")
check("wikilinks still unwrap", bi.clean_obsidian("see [[A Note]]"), "see A Note")
check("aliased wikilinks still unwrap", bi.clean_obsidian("see [[A Note|alias]]"),
      "see alias")
check("image embeds still vanish", bi.clean_obsidian("x ![[img.png]] y"), "x  y")
# A ">" used as a breadcrumb separator mid-line is not a blockquote.
check("mid-line > survives", bi.clean_obsidian("[A > B]"), "[A > B]")


if FAILURES:
    print("FAIL (%d)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("test_note_card: all checks passed")
