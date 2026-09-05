#!/usr/bin/env python3
"""history_review_approve.py - the human gate for history_review.py's queue.

history_review.py DRAFTS notes into /mnt/nvme/brain/history_review/queue from
your own brain REPL transcript; nothing reaches retrieval until this tool
promotes it. Same curated-growth rule research_review.py and web_review.py
already enforce: automation proposes, a person disposes.

Commands:

    python3 history_review_approve.py --list             # queue at a glance
    python3 history_review_approve.py --show STEM        # full note text
    python3 history_review_approve.py --promote STEM      # -> vault/_History,
                                                           # reindex
    python3 history_review_approve.py --reject STEM       # discard the draft

--promote and --reject take --all instead of a STEM to act on the whole queue.
--dry-run prints what each command would do and touches nothing.

Unlike web_review.py there is no denylist: a rejected draft came from your
own transcript, not a re-fetchable url, so there is nothing to prevent from
recurring. Every decision still lands in decisions.jsonl (append-only audit
log), same convention as the other two review tools.

Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from history_review import QUEUE_DIR, HISTORY_DIR
from web_paths import VAULT_ROOT
from web_cache import _atomic_write_text

HISTORY_NOTES_DIR = os.path.join(VAULT_ROOT, "_History")
DECISIONS_LOG = os.path.join(HISTORY_DIR, "decisions.jsonl")


def _append_decision(record):
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        record["ts"] = time.time()
        with open(DECISIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _stems():
    try:
        return sorted(n[:-3] for n in os.listdir(QUEUE_DIR) if n.endswith(".md"))
    except Exception:
        return []


def _resolve(stem, directory):
    stems = sorted(n[:-3] for n in os.listdir(directory) if n.endswith(".md")) \
        if os.path.isdir(directory) else []
    if stem in stems:
        return stem
    matches = [s for s in stems if s.startswith(stem)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("Ambiguous stem %r matches %d notes:" % (stem, len(matches)))
        for s in matches:
            print("  " + s)
    return ""


def _reindex():
    try:
        import update_index
        return update_index.run(vault=VAULT_ROOT, quiet=False)
    except Exception as exc:
        print("[history_review_approve] reindex failed (%r); run "
              "update_index.py manually" % exc)
        return 1


def cmd_list():
    stems = _stems()
    print("%d draft(s) in %s:" % (len(stems), QUEUE_DIR))
    for stem in stems:
        print("  " + stem)
    return 0


def cmd_show(stem):
    resolved = _resolve(stem, QUEUE_DIR)
    if not resolved:
        print("No queued note matching %r (try --list)." % stem)
        return 1
    try:
        with open(os.path.join(QUEUE_DIR, resolved + ".md"), "r",
                  encoding="utf-8", errors="ignore") as fh:
            print(fh.read())
    except Exception as exc:
        print("(could not read the note: %r)" % exc)
        return 1
    return 0


def _promote_one(stem, dry_run):
    src = os.path.join(QUEUE_DIR, stem + ".md")
    dest = os.path.join(HISTORY_NOTES_DIR, stem + ".md")
    if dry_run:
        print("  would promote %s -> %s" % (src, dest))
        return True
    try:
        with open(src, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        print("  cannot read %s (%r); leaving it queued." % (src, exc))
        return False
    text = re.sub(r"(?m)^status:\s*candidate\s*$", "status: active",
                  text, count=1)
    os.makedirs(HISTORY_NOTES_DIR, exist_ok=True)
    if not _atomic_write_text(dest, text):
        print("  could not write %s; leaving it queued." % dest)
        return False
    try:
        os.remove(src)
    except OSError:
        pass
    _append_decision({"action": "promote", "stem": stem, "note_path": dest})
    print("  promoted -> " + dest)
    return True


def cmd_promote(stems, do_all, dry_run):
    targets = _stems() if do_all else []
    if not do_all:
        for raw in stems:
            resolved = _resolve(raw, QUEUE_DIR)
            if not resolved:
                print("No queued note matching %r (try --list)." % raw)
                return 1
            targets.append(resolved)
    if not targets:
        print("Nothing to promote; the queue is empty.")
        return 0
    ok_any = False
    for stem in targets:
        print(stem)
        if _promote_one(stem, dry_run):
            ok_any = True
    if dry_run:
        print("\n[dry run] %d note(s) would be promoted, then reindexed."
              % len(targets))
        return 0
    if not ok_any:
        return 1
    print("\nFolding promoted note(s) into the index...")
    _reindex()
    return 0


def _reject_one(stem, dry_run):
    path = os.path.join(QUEUE_DIR, stem + ".md")
    if dry_run:
        print("  would remove %s" % path)
        return True
    try:
        os.remove(path)
    except OSError as exc:
        print("  could not remove %s (%r)" % (path, exc))
        return False
    _append_decision({"action": "reject", "stem": stem})
    print("  rejected " + stem)
    return True


def cmd_reject(stems, do_all, dry_run):
    targets = _stems() if do_all else []
    if not do_all:
        for raw in stems:
            resolved = _resolve(raw, QUEUE_DIR)
            if not resolved:
                print("No queued note matching %r (try --list)." % raw)
                return 1
            targets.append(resolved)
    if not targets:
        print("Nothing to reject; the queue is empty.")
        return 0
    ok = True
    for stem in targets:
        print(stem)
        ok = _reject_one(stem, dry_run) and ok
    if dry_run:
        print("\n[dry run] %d note(s) would be rejected." % len(targets))
        return 0
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Review the history-review queue; promote approved "
                    "drafts into vault/_History.")
    ap.add_argument("--list", action="store_true", help="list the queue")
    ap.add_argument("--show", metavar="STEM", help="show one draft in full")
    ap.add_argument("--promote", metavar="STEM", nargs="*", default=None,
                    help="move queued note(s) into vault/_History and index")
    ap.add_argument("--reject", metavar="STEM", nargs="*", default=None,
                    help="discard queued note(s)")
    ap.add_argument("--all", action="store_true",
                    help="act on the whole queue (promote/reject)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen and touch nothing")
    args = ap.parse_args(argv)

    if args.show:
        return cmd_show(args.show)
    if args.promote is not None:
        if not args.promote and not args.all:
            print("Usage: history_review_approve.py --promote STEM [STEM ...]  (or --all)")
            return 2
        return cmd_promote(args.promote, args.all, args.dry_run)
    if args.reject is not None:
        if not args.reject and not args.all:
            print("Usage: history_review_approve.py --reject STEM [STEM ...]  (or --all)")
            return 2
        return cmd_reject(args.reject, args.all, args.dry_run)
    return cmd_list()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(0)
