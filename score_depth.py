#!/usr/bin/env python3
"""Deterministic depth scorer for eval_brain.py --answers-out JSONL. No model.

eval_brain clamps its judge score to 100, so a 223-word canned paragraph and a
600-word cited answer carrying the same facts score IDENTICALLY -- and on judge
transport failure _iso_bool defaults a fact to NOT PRESENT, i.e. its noise
points the same way as the effect being measured. Depth therefore needs a scorer
that never calls a model: same input, same number, every run.

Usage:
  python3 score_depth.py baseline=out_a.jsonl depth=out_b.jsonl --vault PATH

Arm name comes from the LABEL= prefix, else the record's "arm" field, else the
file's basename. --vault enables the two metrics that need to know what the
model actually saw (verified DOIs, ungrounded specifics); without it they read
n/a rather than guessing.
"""

import argparse
import json
import os
import re
import sys

import brain_depth as bd
import build_index as bi

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]]+")
_URL_RE = re.compile(r"https?://\S+")
_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Za-z]?\b")


def load_vault(vault):
    """{source_rel_path: (frontmatter_fields, full_text)} for the whole vault."""
    notes = {}
    if not vault:
        return notes
    for root, _dirs, files in os.walk(vault):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            fm, _body = bi.split_frontmatter(text)
            notes[os.path.relpath(path, vault)] = (bi.parse_frontmatter_fields(fm), text)
    return notes


def specifics(text):
    """Citation-like tokens: the fabrication surface small models actually use."""
    return set(_DOI_RE.findall(text)) | set(_URL_RE.findall(text)) | \
        set(m.lower() for m in _ARTICLE_RE.findall(text))


def score_record(rec, notes):
    """Per-answer metrics. Everything scored runs on the body (footer stripped);
    the footer is read only for the DOIs it claims."""
    answer = rec.get("answer") or ""
    body = bd.strip_citations(answer)
    footer = answer[len(body):]
    sources = [s for s in (rec.get("sources") or []) if s]
    m = {
        "n_sources": len(sources),
        "chars": len(body),
        "named_sources": len({s for s in sources
                              if bd._stem_of(s).lower() in body.lower()}),
        "facet_coverage": bd.facet_coverage(body, sources),
        # Diagnostic only -- see the report footer.
        "words": len(body.split()),
    }
    if notes:
        dois = {(notes.get(s) or ({}, ""))[0].get("doi") for s in sources}
        m["verified_dois"] = len({d for d in _DOI_RE.findall(footer) if d in dois})
        ctx = "\n".join((notes.get(s) or ({}, ""))[1] for s in sources).lower()
        m["ungrounded_specifics"] = len([t for t in specifics(body) if t.lower() not in ctx])
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("answers", nargs="+", metavar="[LABEL=]PATH",
                   help="one or more --answers-out JSONL files, one per arm")
    p.add_argument("--vault", default=None,
                   help="vault root; enables verified_dois and ungrounded_specifics")
    args = p.parse_args()

    notes = load_vault(args.vault)
    if args.vault and not notes:
        sys.stderr.write("warning: no .md notes under %s\n" % args.vault)

    arms = {}
    for spec in args.answers:
        label, _, path = spec.partition("=")
        if not path:
            label, path = "", spec
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            arm = label or rec.get("arm") or os.path.splitext(os.path.basename(path))[0]
            arms.setdefault(arm, []).append(score_record(rec, notes))

    keys = ["n_sources", "chars", "named_sources", "verified_dois",
            "facet_coverage", "ungrounded_specifics"]
    print("%-16s %5s %s" % ("arm", "n", " ".join("%18s" % k for k in keys + ["words*"])))
    for arm, rows in arms.items():
        cells = []
        for k in keys + ["words"]:
            vals = [r[k] for r in rows if k in r]
            cells.append("%18s" % ("n/a" if not vals else
                                   "%.2f" % (sum(vals) / float(len(vals)))))
        print("%-16s %5d %s" % (arm, len(rows), " ".join(cells)))
    print("\n* words is DIAGNOSTIC ONLY and is never scored: a metric that "
          "rewards length\n  is gamed by a verbose model that says nothing. "
          "Same for any words-derived\n  rate such as grounded_per_100w. Judge "
          "depth on facet_coverage, named_sources\n  and verified_dois, against "
          "a flat or falling ungrounded_specifics.")
    if not notes:
        print("  verified_dois / ungrounded_specifics need --vault to be computable.")


if __name__ == "__main__":
    main()
