#!/usr/bin/env python3
"""web_review_digest.py - mirror the web-fetch review queue into Obsidian.

web_review.py already has everything a human needs to act on the queue
(--list/--show/--promote/--reject), but it only speaks over SSH: nothing
surfaces the pending pages inside Obsidian itself, since the queue lives at
web_paths.WEB_QUEUE_DIR, deliberately OUTSIDE the vault so build_index.py
never indexes an un-reviewed page.

This script closes that gap without touching the approval boundary. It reads
the same queue web_review.py reads (via its load_queue(), not a second
parser) and writes ONE rolling note into the vault, under _PendingReview/, so
opening Obsidian shows what is waiting. The note is documentation only: it
never promotes or rejects anything, and _PendingReview is pruned by
build_index.py's _JUNK_DIR_NAMES the same way Templates/ and _Fulltext are,
so the checklist itself can never be indexed or dilute BM25.

The note is OVERWRITTEN each run rather than appended, because the queue
itself is the state: an item promoted or rejected through web_review.py
simply stops appearing in load_queue() on the next run, so there is nothing
separate to reconcile against decisions.jsonl.

Run: python3 web_review_digest.py
Meant to run daily via web-review-digest.timer (see that file for the
install steps, same one-time sudo dance as brain-learn.timer).

Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_paths import VAULT_ROOT
from web_cache import _atomic_write_text
from web_review import (load_queue, _first_body_line, _age_days, _status,
                        _host_of, WEB_QUEUE_DIR)

PENDING_REVIEW_DIR = os.path.join(VAULT_ROOT, "_PendingReview")
DIGEST_PATH = os.path.join(PENDING_REVIEW_DIR, "pending-web-review.md")

SSH_HOST = "user@192.168.55.1"


def _age_label(fetched_at):
    days = _age_days(fetched_at)
    if days is None:
        return "unknown age"
    if days == 0:
        return "fetched today"
    if days == 1:
        return "1 day old"
    return "%d days old" % days


def _item_section(entry):
    stem = entry["stem"]
    title = str(entry.get("title") or stem)
    url = str(entry.get("url") or entry.get("final_url") or "")
    query = str(entry.get("query") or "")
    fetched_at = str(entry.get("fetched_at") or "")
    md_path = os.path.join(WEB_QUEUE_DIR, stem + ".md")
    excerpt = _first_body_line(md_path, entry)

    lines = []
    lines.append("### %s" % title)
    lines.append("")
    lines.append("- Source: %s" % url)
    lines.append("- Host: %s" % _host_of(url))
    lines.append("- Query: %s" % query)
    lines.append("- Fetched: %s (%s)" % (fetched_at, _age_label(fetched_at)))
    lines.append("- Status: %s" % _status(entry))
    if excerpt:
        lines.append("- Excerpt: %s" % excerpt)
    lines.append("")
    lines.append("Approve or reject over SSH:")
    lines.append("```")
    lines.append('ssh %s "cd /mnt/nvme/brain && python3 web_review.py --promote %s"'
                 % (SSH_HOST, stem))
    lines.append('ssh %s "cd /mnt/nvme/brain && python3 web_review.py --reject %s"'
                 % (SSH_HOST, stem))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_digest():
    """Return (markdown_text, item_count) for the current queue."""
    queue = load_queue()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    front = [
        "---",
        "title: Pending web review",
        "generator: web_review_digest.py",
        "tags: [web-review-digest]",
        "updated_at: %s" % now,
        "---",
        "",
        "# Pending web review",
        "",
        "Generated %s by web_review_digest.py. Reflects the current queue at "
        "queue/_WebCache/ -- items disappear from here as soon as they are "
        "promoted or rejected via web_review.py." % now,
        "",
        "## Pending (%d)" % len(queue),
        "",
    ]

    if not queue:
        front.append("Nothing pending review.")
        front.append("")
        body = []
    else:
        body = [_item_section(entry) for entry in queue]

    text = "\n".join(front) + "\n".join(body)
    return text, len(queue)


def main():
    text, count = build_digest()
    ok = _atomic_write_text(DIGEST_PATH, text)
    if not ok:
        print("FAILED to write %s" % DIGEST_PATH)
        return 1
    print("%d page(s) pending review -> %s" % (count, DIGEST_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
