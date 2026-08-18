#!/usr/bin/env python3
"""refresh_web.py - keep the web-learned knowledge honest.

The other half of the self-updating loop. update_index.py folds fetched pages
into the index the moment they land; this script runs LATER (by hand or from
the brain-learn.timer systemd unit) and makes sure what was learned stays
true:

  refresh (default)  re-fetch every stored page older than --max-age-days.
                     Changed content updates the JSON + note pair in place
                     (web_cache.py is URL-addressed, so no duplicates). A page
                     that now returns 404/410 has its pair PURGED: the brain
                     un-learns dead knowledge. Transient failures are counted
                     in the entry and only purge after MAX_FAILURES in a row.
  --list             show every learned page (stem, where, age, url) and exit.
  --purge STEM       retire one learned page by stem (see --list) and exit.
                     This is the human override when the brain learned
                     something wrong: purge it, and the following index update
                     removes it from retrieval entirely. --deny / --deny-host
                     also denylist the url so the next fetch cannot re-learn it.

IN PLACE is the load-bearing word. A note lives either in the indexed vault
(approved) or in the unindexed review queue (waiting), and a refresh rewrites it
where it already is. This script runs unattended from brain-learn.timer, so a
refresh that wrote a queued page into the vault would auto-promote content
nobody approved, roughly 30 days after it was fetched. Only web_review.py
promotes.

A purge is a THREE-part retire: the .md note, the .json cache entry, and every
AnswerCache entry that cites the url. See _purge_answers for why the third part
is not optional, and why the long-running services have to be restarted before
a retire is actually true.

After any change the script runs update_index.run() once, so the running brain
hot-reloads a corrected index on its next question. Standing style rule: no em
dashes in anything this writes. Python 3.9 compatible.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import datetime

from web_cache import WebCache, _atomic_write_text
from web_paths import (BRAIN_ROOT, WEB_CACHE_DIR, WEB_NOTES_DIR, WEB_QUEUE_DIR,
                       STORE_CHAR_CAP, deny, is_denied)
import update_index

MAX_FAILURES = 3          # consecutive refresh failures before a purge
FETCH_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (jetson-brain refresh_web)"

# Where a note may live, in lookup order: the approved vault copy first, then
# the review queue. A refresh rewrites the note IN PLACE in whichever of these
# it is found in, and a stem found in NEITHER is treated as queued. Promotion
# out of the queue is web_review.py's job and no automated path may do it: this
# script runs unattended from brain-learn.timer, so a refresh that wrote into
# WEB_NOTES_DIR would silently index pages nobody approved.
NOTE_DIRS = (WEB_NOTES_DIR, WEB_QUEUE_DIR)

# The AnswerCache directory (brain.CACHE_DIR). Imported by path rather than by
# importing brain, which would pull in the whole model stack for a purge.
# WEB_CACHE_DIR is the cache/web subdir of this one; AnswerCache's os.listdir is
# non-recursive, so the two never see each other's files.
ANSWER_CACHE_DIR = os.path.join(BRAIN_ROOT, "cache")


def _entries(cache_dir):
    """Yield (stem, entry_dict, json_path) for every stored page, oldest
    first. Unreadable entries are skipped, never fatal."""
    try:
        names = [n for n in os.listdir(cache_dir) if n.endswith(".json")]
    except OSError:
        return
    rows = []
    for name in names:
        path = os.path.join(cache_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            rows.append((entry.get("fetched_at") or "", name[:-5], entry, path))
        except Exception:
            continue
    rows.sort()
    for _at, stem, entry, path in rows:
        yield stem, entry, path


def _age_days(fetched_at):
    try:
        then = datetime.datetime.fromisoformat(fetched_at)
        return (datetime.datetime.now() - then).days
    except Exception:
        return None


def _fetch(url):
    """GET a url. Returns (status, cleaned_text). status is the HTTP code,
    0 for a transport error, and text is '' unless the fetch succeeded."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return 0, ""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT},
                            timeout=FETCH_TIMEOUT)
        status = resp.status_code
        if status != 200:
            return status, ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        import re
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        # STORE_CHAR_CAP, not the old local PAGE_CHAR_CAP of 2000: storage and
        # prompting are separate budgets now, and truncating here to the prompt
        # budget would re-shrink every enlarged note on the weekly timer.
        return status, text[:STORE_CHAR_CAP]
    except Exception:
        return 0, ""


def _as_dirs(notes_dir):
    """Normalize a notes-dir argument to a tuple of directories. Accepts a
    single path (the pre-queue call shape) or any sequence of them."""
    if not notes_dir:
        return NOTE_DIRS
    if isinstance(notes_dir, str):
        return (notes_dir,)
    return tuple(notes_dir)


def _note_dir(stem, notes_dirs=NOTE_DIRS):
    """The directory that actually holds stem.md, searching notes_dirs in
    order, or '' when the note exists nowhere. This is what makes a refresh
    in-place: the note is rewritten where it already lives, so a queued page
    stays queued and an approved page stays approved. Never raises."""
    for directory in _as_dirs(notes_dirs):
        try:
            if os.path.exists(os.path.join(directory, stem + ".md")):
                return directory
        except Exception:
            continue
    return ""


def _entry_haystack(entry):
    """One searchable string per answer-cache entry: its question, its sources
    list, and its DECOMPRESSED answer body. The body matters because a web
    answer often names the url in prose while 'sources' carries only filenames.
    answer_cache is imported lazily and its absence degrades to the plaintext
    'answer' field rather than failing the purge. Never raises."""
    parts = []
    try:
        parts.append(str(entry.get("question") or ""))
        parts.append(json.dumps(entry.get("sources") or [], ensure_ascii=False))
    except Exception:
        pass
    try:
        import answer_cache
        parts.append(str(answer_cache._entry_answer(entry) or ""))
    except Exception:
        parts.append(str(entry.get("answer") or ""))
    return "\n".join(parts)


def _purge_answers(url, stem, answer_dir, quiet):
    """Delete every AnswerCache entry that cites this page, matching on the raw
    url, the stem, or the stem's note filename. Without this a retire is
    cosmetic: PRINCIPAL_SOURCE_BONUS = 0.4 (answer_cache.py:123) actively
    favours a stored answer that cites the url, so the brain keeps serving the
    rejected page's content from cache long after both its files are gone.

    IMPORTANT, and not advisory. AnswerCache.lookup serves out of an in-memory
    self.entries, so a long-running process that already loaded the cache keeps
    answering from a deleted entry. _register_hit does NOT rewrite the entry
    file on the hit itself: it only bumps the in-memory entry and does
    self._dirty.add(path), and the JSON rewrite is deferred to the next
    store()/clear()/atexit flush. Measured: after store() then os.remove(), the
    next lookup still served the answer with the file gone, and one further
    store() recreated the file. A read-only process resurrects it just as well,
    at clean exit, via the atexit-registered flush. _flush_dirty now stats each
    dirty path and skips ones that no longer exist, so the on-disk delete
    sticks, but that is a best-effort guard with a TOCTOU window and it does
    nothing about the in-memory copy. A purge is only correct once brain and
    the three long-running units (advisor, brain-ollama-shim, brain-web) have
    been restarted. Never raises."""
    needles = [n for n in (url, stem, (stem + ".md") if stem else "") if n]
    if not needles:
        return 0
    removed = 0
    try:
        names = [n for n in os.listdir(answer_dir) if n.endswith(".json")]
    except OSError:
        return 0
    for name in names:
        path = os.path.join(answer_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            if not isinstance(entry, dict):
                continue
            hay = _entry_haystack(entry)
            if not any(needle in hay for needle in needles):
                continue
            os.remove(path)
            removed += 1
            if not quiet:
                print("  purged cached answer %s" % name)
        except Exception:
            continue
    return removed


def _purge(stem, cache_dir, notes_dir, quiet, url="", answers=True,
           answer_dir=ANSWER_CACHE_DIR):
    """Retire one learned page COMPLETELY: the .json cache entry, the .md note
    in whichever directory holds it (vault or queue, both if somehow both), and
    every answer-cache entry citing it. Deleting only the first two leaves the
    answer cache serving the retired page, which is the whole reason this is one
    primitive instead of three call sites. url may be passed when the caller
    already read the entry; when it is not, it is recovered from the JSON before
    that file is deleted. Never raises."""
    if not url:
        try:
            with open(os.path.join(cache_dir, stem + ".json"),
                      "r", encoding="utf-8") as fh:
                url = json.load(fh).get("url") or ""
        except Exception:
            url = ""

    paths = [os.path.join(cache_dir, stem + ".json")]
    for directory in _as_dirs(notes_dir):
        paths.append(os.path.join(directory, stem + ".md"))

    removed = 0
    for path in paths:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    answers_removed = _purge_answers(url, stem, answer_dir, quiet) if answers else 0
    if not quiet:
        print("purged %s (%d file(s), %d cached answer(s))"
              % (stem, removed, answers_removed))
    return removed > 0


def _bump_failures(entry, path):
    """Record one more refresh failure on the entry, in place. Returns the
    new count (existing count on write failure, so purging stays conservative)."""
    count = int(entry.get("refresh_failures") or 0) + 1
    entry["refresh_failures"] = count
    try:
        _atomic_write_text(path, json.dumps(entry, ensure_ascii=False, indent=2))
    except Exception:
        pass
    return count


def run(max_age_days=30, quiet=False, dry_run=False,
        cache_dir=WEB_CACHE_DIR, notes_dir=NOTE_DIRS):
    """Re-fetch every stored page older than max_age_days and rewrite it IN
    PLACE. notes_dir is a SEARCH LIST, not a destination: each stem is rewritten
    into the directory its note already occupies, and one that lives nowhere is
    written to the queue. Nothing here ever moves a note into the vault, which
    is the containment guarantee the weekly timer depends on."""
    dirs = _as_dirs(notes_dir)
    caches = {}
    refreshed = updated = purged = failed = skipped = denied = 0
    for stem, entry, path in _entries(cache_dir):
        url = entry.get("url") or ""
        age = _age_days(entry.get("fetched_at") or "")
        if not url:
            continue
        # A url rejected in review is un-learned rather than refreshed. The
        # denylist alone would only stop the WRITE (web_cache.store_result
        # refuses it), leaving the existing pair on disk forever.
        if is_denied(url):
            if dry_run:
                if not quiet:
                    print("would purge denied %s %s" % (stem, url))
            else:
                _purge(stem, cache_dir, dirs, quiet, url=url)
                denied += 1
            continue
        if age is not None and age < max_age_days:
            skipped += 1
            continue
        # The note's own directory decides where the refresh lands. A queued
        # note stays in the queue; an approved note is rewritten in the vault.
        target = _note_dir(stem, dirs) or WEB_QUEUE_DIR
        if dry_run:
            if not quiet:
                print("would refresh %s (age %s d) into %s  %s"
                      % (stem, age, target, url))
            continue
        cache = caches.get(target)
        if cache is None:
            cache = caches[target] = WebCache(cache_dir, target)
        status, text = _fetch(url)
        refreshed += 1
        if status in (404, 410):
            # The page is gone: un-learn it.
            _purge(stem, cache_dir, dirs, quiet, url=url)
            purged += 1
            continue
        if status == 200 and text:
            result = cache.store_result(entry.get("query") or "", url, text,
                                        entry.get("status") or "")
            if result and not result.get("skipped"):
                updated += 1
                if not quiet:
                    print("refreshed %s in %s" % (stem, target))
            continue
        # Transport error, bot-block, or empty text: count it, purge only
        # after MAX_FAILURES consecutive misses (a flaky server or a 403
        # wall should not erase learned knowledge on one bad day).
        failed += 1
        if _bump_failures(entry, path) >= MAX_FAILURES:
            _purge(stem, cache_dir, dirs, quiet, url=url)
            purged += 1

    changed = updated + purged + denied
    if not quiet:
        print("refresh done: %d checked, %d updated, %d purged, %d denied, "
              "%d failed, %d fresh-enough"
              % (refreshed, updated, purged, denied, failed, skipped))
    if changed and not dry_run:
        update_index.run(quiet=quiet)
    return 0


def _offer_deny(url, args):
    """Denylist the purged url, so the next fetch of the same page does not
    quietly re-learn what was just retired. --deny / --deny-host decide it
    outright; otherwise an INTERACTIVE run is asked, defaulting to no. The
    prompt is guarded on isatty so brain-learn.timer, which has no terminal,
    can never block on it. Never raises."""
    try:
        if not url:
            return False
        want_host = bool(args.deny_host)
        want = bool(args.deny) or want_host
        if not want:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                return False
            answer = input("also denylist %s so it is not re-fetched? "
                           "[y/N/host] " % url).strip().lower()
            if answer in ("h", "host"):
                want = want_host = True
            elif answer in ("y", "yes"):
                want = True
        if not want:
            return False
        ok = deny(url, reason=args.reason or "purged via refresh_web",
                  host=want_host)
        if not args.quiet:
            if ok:
                print("denylisted %s%s"
                      % (url, " (whole host)" if want_host else ""))
            else:
                print("could not write the denylist")
        return ok
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Refresh, list, or purge web-learned knowledge.")
    ap.add_argument("--max-age-days", type=int, default=30)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--purge", metavar="STEM")
    ap.add_argument("--deny", action="store_true",
                    help="with --purge: also denylist the url, so a later "
                         "fetch does not simply re-learn the page")
    ap.add_argument("--deny-host", action="store_true",
                    help="with --purge: denylist the whole host, not just "
                         "this one page")
    ap.add_argument("--reason", default="",
                    help="with --deny: why, recorded in the denylist entry")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        count = 0
        for stem, entry, _path in _entries(WEB_CACHE_DIR):
            age = _age_days(entry.get("fetched_at") or "")
            fails = int(entry.get("refresh_failures") or 0)
            flag = (" [%d fail(s)]" % fails) if fails else ""
            # Where the note lives is the reviewer's most useful column: vault
            # means indexed and answering questions, queue means waiting.
            found = _note_dir(stem)
            where = ("vault" if found == WEB_NOTES_DIR
                     else "queue" if found == WEB_QUEUE_DIR
                     else "none")
            state = "snippet" if (entry.get("status") or "") == "snippet" else ""
            print("%-44s %-5s age %3s d%s %-7s %s" % (
                stem, where, "?" if age is None else age, flag, state,
                entry.get("url") or ""))
            count += 1
        print("%d learned page(s)." % count)
        return 0

    if args.purge:
        # Read the url BEFORE the purge deletes the file that carries it, so a
        # deny can still be offered afterwards.
        url = ""
        try:
            with open(os.path.join(WEB_CACHE_DIR, args.purge + ".json"),
                      "r", encoding="utf-8") as fh:
                url = json.load(fh).get("url") or ""
        except Exception:
            url = ""
        ok = _purge(args.purge, WEB_CACHE_DIR, NOTE_DIRS, args.quiet, url=url)
        if ok:
            _offer_deny(url, args)
            update_index.run(quiet=args.quiet)
            if not args.quiet:
                print("restart brain, advisor, brain-ollama-shim and brain-web: "
                      "a running process still holds the purged answers in "
                      "memory and rewrites them on the next hit.")
            return 0
        print("no stored pair named %r (try --list)" % args.purge)
        return 1

    return run(max_age_days=args.max_age_days, quiet=args.quiet,
               dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
