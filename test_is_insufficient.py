#!/usr/bin/env python3
"""Smoke test for brain._is_insufficient.

Guards the refusal detector that decides whether the vault-first pass answered
or must fall out to the web. Regression target: a small model (llama3.2:3b)
paraphrasing its refusal as "I can't answer this question." instead of emitting
the exact INSUFFICIENT sentinel must be classified INSUFFICIENT so the web
fallback fires, rather than being shown to the user as a real vault answer.

Run:  python3 test_is_insufficient.py
Exits non-zero on any failure.
"""

import sys

from brain import _is_insufficient


# (text, expected_insufficient)
CASES = [
    # --- the exact reported failure and close variants: must be True --------
    ("I can't answer this question.", True),
    ("I cannot answer this question.", True),
    ("I can not answer that.", True),
    ("Sorry, I don't know.", True),
    ("I do not know the answer.", True),
    ("I'm unable to answer based on the snippets.", True),
    ("I am unable to help with that.", True),
    ("The vault does not contain enough information to answer.", True),
    ("These snippets don't contain enough to answer the question.", True),
    ("There is no information about this in the provided notes.", True),
    ("There is no relevant information in the vault.", True),
    ("I cannot provide an answer from these sources.", True),
    ("Not enough information is available.", True),
    # multi-line / whitespace-wrapped refusal
    ("  I can't answer\n   this question.  ", True),

    # --- the exact sentinel path: must stay True ---------------------------
    ("INSUFFICIENT", True),
    ("insufficient", True),
    ("Insufficient.", True),
    ("INSUFFICIENT - the snippets do not cover this.", True),
    ("", True),
    ("   ", True),
    (None, True),

    # --- genuine short vault answers: must be False (not kicked to web) ----
    ("No.", False),
    ("Yes, under the CRPD.", False),
    ("Wyatte Hall.", False),
    ("1994.", False),
    ("None recorded.", False),
    ("Maartje De Meulder wrote it.", False),
    # a long, real answer that only quotes a refusal-ish phrase deep in the
    # body must NOT be misread as a refusal (head-only matching)
    (
        "The Ballard 2024 source argues that ASL storytelling preserves Deaf "
        "cultural identity across generations. It documents specific narrative "
        "roots and their transmission. Note that one interviewee said they "
        "could not answer whether the tradition predates 1817, but the broader "
        "record is clear that community storytelling long predates formal "
        "schooling.",
        False,
    ),
]


def main():
    failures = []
    for text, expected in CASES:
        got = _is_insufficient(text)
        status = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures.append((text, expected, got))
        preview = ("<None>" if text is None else repr(text))[:60]
        print("[%s] expected=%-5s got=%-5s  %s" % (
            status, expected, got, preview))

    print()
    if failures:
        print("%d/%d FAILED:" % (len(failures), len(CASES)))
        for text, expected, got in failures:
            print("  expected %s got %s for %r" % (expected, got, text))
        sys.exit(1)
    print("all %d cases passed" % len(CASES))


if __name__ == "__main__":
    main()
