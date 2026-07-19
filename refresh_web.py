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
  --list             show every learned page (stem, age, url) and exit.
  --purge STEM       remove one learned page by stem (see --list) and exit.
                     This is the human override when the brain learned
                     something wrong: purge it, and the following index update
                     removes it from retrieval entirely.

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

from web_cache import (WebCache, WEB_CACHE_DIR, WEB_NOTES_DIR,
                       _atomic_write_text)
import update_index

MAX_FAILURES = 3          # consecutive refresh failures before a purge
FETCH_TIMEOUT = 10
PAGE_CHAR_CAP = 2000      # match brain.fetch_page so refreshed text is comparable
USER_AGENT = "Mozilla/5.0 (jetson-brain refresh_web)"


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
        return status, text[:PAGE_CHAR_CAP]
    except Exception:
        return 0, ""


def _purge(stem, cache_dir, notes_dir, quiet):
    removed = 0
    for path in (os.path.join(cache_dir, stem + ".json"),
                 os.path.join(notes_dir, stem + ".md")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    if not quiet:
        print("purged %s (%d file(s))" % (stem, removed))
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
        cache_dir=WEB_CACHE_DIR, notes_dir=WEB_NOTES_DIR):
    cache = WebCache(cache_dir, notes_dir)
    refreshed = updated = purged = failed = skipped = 0
    for stem, entry, path in _entries(cache_dir):
        url = entry.get("url") or ""
        age = _age_days(entry.get("fetched_at") or "")
        if not url:
            continue
        if age is not None and age < max_age_days:
            skipped += 1
            continue
        if dry_run:
            if not quiet:
                print("would refresh %s (age %s d) %s" % (stem, age, url))
            continue
        status, text = _fetch(url)
        refreshed += 1
        if status in (404, 410):
            # The page is gone: un-learn it.
            _purge(stem, cache_dir, notes_dir, quiet)
            purged += 1
            continue
        if status == 200 and text:
            result = cache.store_result(entry.get("query") or "", url, text)
            if result and not result.get("skipped"):
                updated += 1
                if not quiet:
                    print("refreshed %s" % stem)
            continue
        # Transport error, bot-block, or empty text: count it, purge only
        # after MAX_FAILURES consecutive misses (a flaky server or a 403
        # wall should not erase learned knowledge on one bad day).
        failed += 1
        if _bump_failures(entry, path) >= MAX_FAILURES:
            _purge(stem, cache_dir, notes_dir, quiet)
            purged += 1

    changed = updated + purged
    if not quiet:
        print("refresh done: %d checked, %d updated, %d purged, %d failed, "
              "%d fresh-enough" % (refreshed, updated, purged, failed, skipped))
    if changed and not dry_run:
        update_index.run(quiet=quiet)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Refresh, list, or purge web-learned knowledge.")
    ap.add_argument("--max-age-days", type=int, default=30)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--purge", metavar="STEM")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        count = 0
        for stem, entry, _path in _entries(WEB_CACHE_DIR):
            age = _age_days(entry.get("fetched_at") or "")
            fails = int(entry.get("refresh_failures") or 0)
            flag = (" [%d fail(s)]" % fails) if fails else ""
            print("%-44s age %3s d%s  %s" % (
                stem, "?" if age is None else age, flag, entry.get("url") or ""))
            count += 1
        print("%d learned page(s)." % count)
        return 0

    if args.purge:
        ok = _purge(args.purge, WEB_CACHE_DIR, WEB_NOTES_DIR, args.quiet)
        if ok:
            update_index.run(quiet=args.quiet)
            return 0
        print("no stored pair named %r (try --list)" % args.purge)
        return 1

    return run(max_age_days=args.max_age_days, quiet=args.quiet,
               dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
