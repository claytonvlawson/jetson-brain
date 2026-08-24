#!/usr/bin/env python3
"""Smoke test for brain.PLAIN_VOICE and its placement in the system prompts.

Guards the shared voice rule that stops the answerer narrating its own plumbing
("Moreover, the snippets highlight...") and stops the stock AI-tell openers.

Two regressions matter here, and only one of them is obvious:

1. The voice rule silently falling out of a prompt. Cheap to check, and the
   reason this file exists at all.

2. The far more expensive one: PLAIN_VOICE being appended to the END of
   VAULT_FIRST_SYSTEM instead of before the attribution block. The attribution
   requirement ("name the scholar, the school, the year") was deliberately moved
   to the final position because a 3B follows the last thing it reads, and the
   judged failure taxonomy behind that move was 10 of 13 failures scoring 80-86
   on a single missing NAME. Appending anything after it demotes it and undoes
   that win invisibly: nothing crashes, the answers just quietly stop naming
   Bauman, Humphries, Stokoe. The ends-with assertion below is the only thing
   that catches it.

ROUTE_SYSTEM is deliberately excluded: it must emit exactly one FETCH/SEARCH
line, so prose-style rules are noise at best there.

Run:  python3 test_plain_voice.py
Exits non-zero on any failure.
"""

import sys

import brain


# Every answer-producing system prompt. ROUTE_SYSTEM is not one.
VOICED_PROMPTS = (
    "VAULT_FIRST_SYSTEM",
    "PARENT_SYSTEM",
    "NO_WEB_SYSTEM",
    "WEB_SYSTEM",
    "DOC_SYSTEM",
)

# The attribution sentence that must stay dead last in VAULT_FIRST_SYSTEM.
ATTRIBUTION_TAIL = ("Naming who or where a claim comes from is part of "
                    "answering, not optional detail.")

# The meta-reference phrases the rule exists to suppress, plus the substitute
# behaviour that is the reason it works at all. Asserted present so a
# well-meaning "tighten the prompt" edit cannot quietly drop them.
REQUIRED_IN_VOICE = (
    "the snippets",
    "the provided sources",
    "the excerpts",
    "cite its source",       # the substitute; a bare ban measurably does less
)

# Phrases that must NOT reappear here. This is not style policing, it is a
# measured result: enumerating forbidden connectives moved stock openers
# 17 -> 18 -> 14 across three arms, i.e. no reliable benefit, while costing 36
# words on a prompt whose length is a known hazard on this model. The padding
# tells were 0 in ALL THREE arms, so banning them bought nothing at all.
#
# Note for whoever reads the git history: an earlier version of this comment
# claimed the "not just X, but Y" ban CREATED that shape (0 -> 6). Arm C
# removed the ban and still produced 5, which disproves it. The cause is not
# established. See the PLAIN_VOICE comment in brain.py for the full table and
# the one untested hypothesis.
FORBIDDEN_IN_VOICE = (
    "Moreover",
    "Furthermore",
    "In conclusion",
    "not just X",
    "delve",
    "rich tapestry",
    "serves as a testament",
)

# PLAIN_VOICE is appended to prompts that are already long, and prompt LENGTH is
# a known hazard on this model (a ~700 word filler prompt measurably broke the
# grounding contract). This is a loose tripwire, not a style preference: it fires
# only if someone grows the rule into an essay.
MAX_VOICE_WORDS = 120


def main():
    failures = []

    def check(label, ok, detail=""):
        print("[%s] %s%s" % ("ok  " if ok else "FAIL", label,
                            "" if ok else "  <- " + detail))
        if not ok:
            failures.append(label)

    voice = brain.PLAIN_VOICE

    check("PLAIN_VOICE is a non-empty string",
          isinstance(voice, str) and voice.strip() != "")

    check("PLAIN_VOICE stays terse (<= %d words)" % MAX_VOICE_WORDS,
          len(voice.split()) <= MAX_VOICE_WORDS,
          "%d words" % len(voice.split()))

    for tell in REQUIRED_IN_VOICE:
        check("PLAIN_VOICE still names %r" % tell, tell in voice)

    for tell in FORBIDDEN_IN_VOICE:
        check("PLAIN_VOICE does NOT re-enumerate %r" % tell, tell not in voice,
              "enumerating this measurably did nothing or backfired")

    for name in VOICED_PROMPTS:
        prompt = getattr(brain, name, None)
        check("%s exists" % name, isinstance(prompt, str))
        if not isinstance(prompt, str):
            continue
        check("%s carries PLAIN_VOICE" % name, voice in prompt)

    # ROUTE_SYSTEM answers nothing; a voice rule there is dead weight on a
    # single-line protocol the parser depends on.
    check("ROUTE_SYSTEM is left alone", voice not in brain.ROUTE_SYSTEM)

    # The load-bearing one. See the module docstring.
    check("VAULT_FIRST_SYSTEM still ENDS with the attribution rule",
          brain.VAULT_FIRST_SYSTEM.rstrip().endswith(ATTRIBUTION_TAIL),
          "tail is ..." + repr(brain.VAULT_FIRST_SYSTEM.rstrip()[-90:]))

    # PLAIN_VOICE must precede it, not follow it.
    vf = brain.VAULT_FIRST_SYSTEM
    check("PLAIN_VOICE precedes the attribution rule in VAULT_FIRST_SYSTEM",
          voice in vf and vf.index(voice) < vf.index(ATTRIBUTION_TAIL))

    # The grounding contract itself must survive every voice edit. PLAIN_VOICE
    # constrains form only; if one of these ever goes missing the prompt was
    # rewritten, not restyled.
    for name, needle in (
        ("VAULT_FIRST_SYSTEM", "USING ONLY the vault snippets provided"),
        ("VAULT_FIRST_SYSTEM", "EXACTLY the single word INSUFFICIENT"),
        ("PARENT_SYSTEM", "USING ONLY the vault snippets provided"),
        ("PARENT_SYSTEM", "EXACTLY the single word INSUFFICIENT"),
    ):
        check("%s keeps the grounding clause %r" % (name, needle[:40]),
              needle in getattr(brain, name))

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for label in failures:
            print("  " + label)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
