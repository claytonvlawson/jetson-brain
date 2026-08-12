#!/usr/bin/env python3
"""research_review.py - the human gate for the research agent's queue.

research_agent.py DRAFTS notes into /mnt/nvme/brain/research/queue; nothing
reaches retrieval until this tool promotes it. That is the curated-growth
rule: automation proposes, a person disposes.

Flow per candidate (highest relevance score first, grey literature last):

    approve -> the draft note (including any edits you made to the queue .md)
               is written into vault/_Research/ with status: active, the queue
               pair is removed, and the incremental indexer folds it in so the
               running brain can cite it on the next question
    reject  -> queue pair removed, its ids join the permanent rejected set
               (the agent will never draft it again)
    edit    -> opens the draft in $EDITOR, then asks again
    skip    -> leaves it for a later session
    quit    -> stop; approvals so far are kept and indexed

Other commands:

    python3 research_review.py --list            # queue at a glance
    python3 research_review.py --show-discards   # what the LLM auto-discarded
    python3 research_review.py retire <citekey>  # remove a promoted note
                                                 # (undo, or after moving it
                                                 # into Sources/ by hand)

Every decision lands in decisions.jsonl (append-only audit log).
Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import research_note as rn
from research_agent import (QUEUE_DIR, RESEARCH_DIR, VAULT_DIR,
                            load_state, save_state, all_keys)
from web_cache import _atomic_write_text

RESEARCH_NOTES_DIR = os.path.join(VAULT_DIR, "_Research")
DECISIONS_LOG = os.path.join(RESEARCH_DIR, "decisions.jsonl")


def _append_decision(record):
    try:
        os.makedirs(RESEARCH_DIR, exist_ok=True)
        record["ts"] = time.time()
        with open(DECISIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_queue():
    """All queued candidates, sorted for review: scored high to low, unscored
    next, grey literature last."""
    items = []
    try:
        for name in sorted(os.listdir(QUEUE_DIR)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(QUEUE_DIR, name), "r",
                          encoding="utf-8") as fh:
                    cand = json.load(fh)
                cand["queue_stem"] = cand.get("queue_stem") or name[:-5]
                items.append(cand)
            except Exception:
                continue
    except Exception:
        pass

    def sort_key(c):
        grey = 1 if c.get("access") == "grey-lit" else 0
        score = c.get("score")
        return (grey, 0 if score is not None else 1, -(score or 0),
                -(c.get("year") or 0))
    items.sort(key=sort_key)
    return items


def _remove_pair(stem):
    for ext in (".json", ".md"):
        try:
            os.remove(os.path.join(QUEUE_DIR, stem + ext))
        except OSError:
            pass


def show(cand, idx, total):
    print("\n" + "=" * 72)
    score = cand.get("score")
    print("[%d/%d]  score: %s   access: %s   via %s" % (
        idx, total, score if score is not None else "unscored",
        cand.get("access"), cand.get("discovery_source")))
    print("-" * 72)
    print("Title  : " + str(cand.get("title") or ""))
    print("Authors: " + ", ".join(cand.get("authors") or []) or "unknown")
    print("Venue  : %s (%s)   %s" % (
        cand.get("venue") or "unknown", cand.get("year") or "?",
        "PEER-REVIEWED" if cand.get("peer_reviewed") else
        ("grey literature" if cand.get("access") == "grey-lit"
         else "peer review unverified")))
    if cand.get("doi"):
        print("DOI    : https://doi.org/" + cand["doi"])
    elif cand.get("landing_url"):
        print("Link   : " + cand["landing_url"])
    if cand.get("score_reason"):
        print("LLM    : " + cand["score_reason"])
    if cand.get("fetch_note"):
        print("Fetch  : " + cand["fetch_note"])
    elif cand.get("fulltext_via"):
        print("Fetch  : full text via " + cand["fulltext_via"])
    abstract = (cand.get("abstract") or "").strip()
    if abstract:
        print("-" * 72)
        print(abstract[:500] + ("..." if len(abstract) > 500 else ""))
    print("=" * 72)


def _promote(cand):
    """Write the (possibly hand-edited) draft into vault/_Research with
    status flipped to active. Returns the vault path or ''."""
    stem = cand["queue_stem"]
    md_path = os.path.join(QUEUE_DIR, stem + ".md")
    try:
        with open(md_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        text = re.sub(r"(?m)^status:\s*candidate\s*$", "status: active",
                      text, count=1)
    except Exception:
        text = rn.render_note(cand, fulltext=cand.get("fulltext") or "",
                              fulltext_via=cand.get("fulltext_via") or "",
                              status="active")
    os.makedirs(RESEARCH_NOTES_DIR, exist_ok=True)
    dest = os.path.join(RESEARCH_NOTES_DIR, rn.note_filename(cand))
    return dest if _atomic_write_text(dest, text) else ""


def _reindex():
    """Fold approvals/removals into bm25.pkl; the running brain hot-reloads."""
    try:
        import update_index
        return update_index.run(vault=VAULT_DIR, quiet=False)
    except Exception as exc:
        print("[review] reindex failed (%r); run update_index.py manually"
              % exc)
        return 1


def cmd_review(limit):
    queue = load_queue()
    if limit > 0:
        queue = queue[:limit]
    print("Research review: %d draft(s) awaiting judgment." % len(queue))
    if not queue:
        print("Queue is empty. The agent runs weekly (research-agent.timer), "
              "or run research_agent.py yourself.")
        return 0

    state = load_state()
    approvals = 0
    removals = 0
    for i, cand in enumerate(queue, 1):
        show(cand, i, len(queue))
        while True:
            choice = input(
                "  [a]pprove  [r]eject  [e]dit  [s]kip  [q]uit > ").strip().lower()
            if choice == "e":
                editor = os.environ.get("EDITOR", "nano")
                md = os.path.join(QUEUE_DIR, cand["queue_stem"] + ".md")
                try:
                    subprocess.call([editor, md])
                except Exception as exc:
                    print("  could not open editor: %r" % exc)
                continue
            if choice in ("a", "r", "s", "q"):
                break
            print("  Please enter a, r, e, s, or q.")
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "r":
            state["rejected"].extend(all_keys(cand))
            _remove_pair(cand["queue_stem"])
            removals += 1
            _append_decision({"action": "reject", "citekey":
                              rn.make_citekey(cand),
                              "title": cand.get("title")})
            print("  rejected (will not be drafted again).")
            continue
        dest = _promote(cand)
        if not dest:
            print("  could not write the vault note; leaving it queued.")
            continue
        _remove_pair(cand["queue_stem"])
        approvals += 1
        _append_decision({"action": "approve", "citekey":
                          rn.make_citekey(cand), "title": cand.get("title"),
                          "note_path": dest})
        print("  approved -> " + dest)

    save_state(state)
    print("\nSession done: %d approved, %d rejected." % (approvals, removals))
    if approvals:
        print("Folding approved notes into the index...")
        _reindex()
        print("Pull them into the Mac vault with:  ./research-pull.sh")
    return 0


def cmd_list():
    queue = load_queue()
    print("%d draft(s) in the queue:" % len(queue))
    for cand in queue:
        score = cand.get("score")
        print("  [%s] %s (%s) %s" % (
            score if score is not None else "-",
            (cand.get("title") or "")[:64], cand.get("year") or "?",
            cand.get("access")))
    return 0


def cmd_show_discards():
    state = load_state()
    discards = state.get("discards") or []
    print("%d LLM auto-discard(s) recorded (threshold lives in "
          "research_agent.py):" % len(discards))
    for d in discards[-50:]:
        print("  [%s] %s  (%s)" % (d.get("score"), (d.get("title") or "")[:64],
                                   d.get("reason") or "no reason"))
    return 0


def cmd_retire(citekey):
    """Remove a promoted _Research note by citekey and reindex."""
    target = None
    try:
        for name in os.listdir(RESEARCH_NOTES_DIR):
            if not name.endswith(".md"):
                continue
            path = os.path.join(RESEARCH_NOTES_DIR, name)
            try:
                with open(path, "r", encoding="utf-8",
                          errors="ignore") as fh:
                    head = fh.read(2000)
            except Exception:
                continue
            if re.search(r"(?m)^citekey:\s*%s\s*$" % re.escape(citekey), head):
                target = path
                break
    except Exception:
        pass
    if not target:
        print("No _Research note with citekey %r." % citekey)
        return 1
    try:
        os.remove(target)
    except OSError as exc:
        print("Could not remove %s: %r" % (target, exc))
        return 1
    _append_decision({"action": "retire", "citekey": citekey,
                      "note_path": target})
    print("Retired " + target)
    _reindex()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Review the research agent's queue; approve into the vault.")
    ap.add_argument("command", nargs="?", default="review",
                    choices=("review", "retire"))
    ap.add_argument("citekey", nargs="?", help="citekey for retire")
    ap.add_argument("--limit", type=int, default=0,
                    help="review at most N drafts this session")
    ap.add_argument("--list", action="store_true", help="list the queue")
    ap.add_argument("--show-discards", action="store_true",
                    help="show what the LLM auto-discarded")
    args = ap.parse_args(argv)
    if args.list:
        return cmd_list()
    if args.show_discards:
        return cmd_show_discards()
    if args.command == "retire":
        if not args.citekey:
            print("Usage: research_review.py retire <citekey>")
            return 2
        return cmd_retire(args.citekey)
    return cmd_review(args.limit)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(0)
