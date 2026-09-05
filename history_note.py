#!/usr/bin/env python3
"""history_note.py - renders history_review.py candidates as ThesisVault notes.

Three kinds of candidate, one shared schema:

  gap     - a transcript turn whose sources were empty/thin: the vault likely
            lacks material on this topic.
  novel   - a transcript turn whose answer said something not evidently
            covered by the titles of its own cited sources.
  summary - one rolling entry per calendar day: what was asked, what mode/
            model answered it, nothing inferred.

Deliberately NOT the research_note.py schema (name/authors/doi/citekey): these
notes have no paper behind them, so the source_ts field (unix timestamps into
cache/transcript.jsonl) is what stands in for a citation, letting a human trace
a note back to the exact Q&A turn(s) that produced it.

Reuses web_cache.py's hardened primitives (_atomic_write_text, sanitizers, the
no-em-dash rule). Never raises into the caller. Python 3.9 compatible.
"""
from __future__ import annotations

import os
import re
import datetime

from web_cache import (_atomic_write_text, _sanitize_frontmatter_value,
                       _no_dashes, _yaml_quote, _clean_body_text)

_SAFE_NAME_RE = re.compile(r'[\\/:#^|\[\]?*"<>]')


def safe_name(text, max_len=90):
    """Obsidian-safe filename fragment (no forbidden chars, no em dashes)."""
    try:
        name = _SAFE_NAME_RE.sub("", _no_dashes(str(text))).strip().rstrip(".")
        name = re.sub(r"\s+", " ", name)
        return name[:max_len].strip()
    except Exception:
        return "untitled"


def _date_str(ts):
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except Exception:
        return "unknown-date"


def _slug(text, max_len=50):
    """Underscore slug with no spaces/punctuation. Unlike safe_name (used for
    the human-readable frontmatter title), this becomes the queue filename
    stem AND gets embedded, unquoted, into the ssh --promote/--reject command
    lines the digest note and the note body print for copy-paste. A stem
    containing a space would word-split into multiple argv entries on the
    remote shell and break argparse; a stem containing a quote could break
    the quoting entirely. Only [a-z0-9_-] survives."""
    try:
        s = re.sub(r"[^a-z0-9]+", "_", _no_dashes(str(text)).lower()).strip("_")
        return (s or "untitled")[:max_len]
    except Exception:
        return "untitled"


def queue_stem(cand):
    """Deterministic stem: date + kind + a slug of the title, so the same
    candidate reprocessed twice (should not happen, state.json's cursor
    prevents it) overwrites in place rather than duplicating."""
    date = _date_str((cand.get("source_ts") or [0])[0])
    kind = str(cand.get("kind") or "note")
    slug = _slug(cand.get("title") or "untitled")
    return "%s_%s_%s" % (date, kind, slug)


def note_filename(cand):
    return queue_stem(cand) + ".md"


_KIND_LABEL = {
    "gap": "retrieval gap",
    "novel": "novel information",
    "summary": "daily session summary",
}


def render_note(cand, status="candidate"):
    """The full .md text for one candidate. Frontmatter is provenance; the
    body restates every load-bearing field so BM25 can retrieve the note once
    it is promoted."""
    kind = str(cand.get("kind") or "note")
    title = _no_dashes(str(cand.get("title") or "Untitled").strip())
    summary = _no_dashes(str(cand.get("summary") or "").strip())
    questions = [str(q) for q in (cand.get("questions") or []) if str(q).strip()]
    source_ts = [float(t) for t in (cand.get("source_ts") or [])
                 if isinstance(t, (int, float))]
    discovered = str(cand.get("discovered_at") or "")
    date = _date_str(source_ts[0] if source_ts else 0)

    fm = [
        "---",
        "name: %s" % _yaml_quote(_sanitize_frontmatter_value(title)),
        "type: history-review",
        "tags: [history-review, %s]" % kind,
        "status: %s" % status,
        "discovery_source: history_review",
        "source_ts: [%s]" % ", ".join(str(t) for t in source_ts),
        "review_date: %s" % date,
        "discovered: %s" % _yaml_quote(_sanitize_frontmatter_value(discovered)),
        "---",
    ]

    body = [
        "",
        "# " + title,
        "",
        "**Kind:** %s  " % _KIND_LABEL.get(kind, kind),
        "**Date:** %s  " % date,
    ]
    if summary:
        body.extend(["", _clean_body_text(summary)])
    if questions:
        body.extend(["", "## Question(s) that prompted this", ""])
        body.extend("- " + _clean_body_text(q) for q in questions)
    body.extend(["", "## Approve or reject", "",
                 "This note was drafted by history_review.py from your own "
                 "brain REPL transcript. It is NOT indexed until promoted:",
                 "```",
                 'ssh user@192.168.55.1 "cd /mnt/nvme/brain && '
                 'python3 history_review_approve.py --promote %s"' % queue_stem(cand),
                 'ssh user@192.168.55.1 "cd /mnt/nvme/brain && '
                 'python3 history_review_approve.py --reject %s"' % queue_stem(cand),
                 "```", ""])

    return "\n".join(fm) + "\n" + "\n".join(body)


def write_note(cand, dest_dir, status="candidate"):
    """Render and atomically write the note. Returns the path or '' on
    failure. Never raises."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, note_filename(cand))
        text = render_note(cand, status=status)
        return path if _atomic_write_text(path, text) else ""
    except Exception:
        return ""
