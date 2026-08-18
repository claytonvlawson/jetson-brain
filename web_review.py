#!/usr/bin/env python3
"""web_review.py - the human gate for the web-fetch queue.

The web loop DRAFTS: every page the brain fetches is persisted as a pair with a
shared stem, SPLIT across two roots on purpose. The .md note goes to the queue,
web_paths.WEB_QUEUE_DIR, outside the vault so build_index.py's os.walk cannot
see it; the .json entry goes to web_paths.WEB_CACHE_DIR, a dedup cache kept out
of AnswerCache's non-recursive listdir (see web_cache.py's module docstring).
So the QUEUE is whichever .md files sit in WEB_QUEUE_DIR, and the ENTRY for a
stem is ALWAYS loaded from WEB_CACHE_DIR, promoted or not. Reading the entry
from the queue dir finds nothing, which is what used to make reject silently
non-durable. Nothing reaches retrieval until this tool promotes it. Same
curated-growth rule research_review.py enforces for the research agent:
automation proposes, a person disposes.

Commands:

    python3 web_review.py --list              # queue at a glance
    python3 web_review.py --list --all        # queue plus promoted notes
    python3 web_review.py --show STEM         # full record for one page
    python3 web_review.py --promote STEM      # queue pair -> vault/_WebCache,
                                              # reindex, verify by content
    python3 web_review.py --reject STEM       # denylist the url, purge the pair
    python3 web_review.py --retire STEM       # purge an ALREADY-PROMOTED note
                                              # from note + cache + answer cache

--promote and --reject take --all instead of a STEM to act on the whole queue.
--dry-run prints what each command would do and touches nothing.

Four things do not carry over from research_review.py:

  1. Identity is web_cache._deterministic_stem, not a citekey.
  2. Reject must be DURABLE. web_paths.deny() writes a denylist that
     web_cache consults before it persists anything, otherwise the next fetch
     of the same url simply re-learns it.
  3. Retire is a many-file operation (the vault note, a queue note if the page
     was refetched after promotion, the JSON entry, and every answer-cache
     entry citing it), so it delegates to refresh_web._purge with BOTH note
     dirs rather than unlinking by hand, and it prints a REQUIRED restart
     instruction: a running process answers from an in-memory cache that
     deleting files on disk does not touch. See RESTART_NOTICE.
  4. update_index.run() returns 0 both when it indexed and when it lock-skipped,
     and an mtime check on the pickle gives false DONEs because AUTOLEARN can
     spawn a concurrent update that rewrites it. Promotion is therefore verified
     by CONTENT: every chunk the note currently chunks into has to be in
     bm25.pkl under its source. Matching the source string alone is satisfied by
     a stale chunk from an older version of the same page, and a lock-skipped
     run is never reported as verified at all.

Every decision lands in decisions.jsonl (append-only audit log).
Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import web_paths
from web_paths import (WEB_QUEUE_DIR, WEB_NOTES_DIR, WEB_CACHE_DIR,
                       QUEUE_ROOT, VAULT_ROOT)

DECISIONS_LOG = os.path.join(QUEUE_ROOT, "decisions.jsonl")

# The vault-relative source key update_index.py stores on every chunk. Kept as a
# constant because the promote verification depends on matching it exactly.
MANAGED_DIR_NAME = "_WebCache"

# How long to wait for a concurrent update_index (brain-learn.service, or a
# fire-and-forget _spawn_index_update from the REPL) to drop .update-lock before
# reindexing anyway. Waiting is only an optimization: the content check below is
# what actually decides whether a promote landed.
LOCK_WAIT_SECONDS = 90
LOCK_POLL_SECONDS = 1.0

RESTART_NOTICE = """
REQUIRED, not advisory. The retired content is still live in memory:
AnswerCache.lookup answers out of an in-memory dict that a delete on disk does
not touch, so a still-running process keeps serving the retired page. Measured:
after the entry file was removed, lookup still returned a hit.

The file itself is a slower hazard, and not the one older text here described.
_register_hit does NOT rewrite the entry on a hit; it only marks the path dirty
(answer_cache.py:589) and defers the JSON write to the next store(), clear() or
atexit flush. That flush skips paths that no longer exist, so an ordinary hit,
and a clean exit after one, leave the deleted file deleted. Two things still
put it back: a store() under the SAME cache path, which is what happens when a
long-running process is asked that question again and answers it fresh, and the
exists check being a plain stat, so a delete landing inside its TOCTOU window is
overwritten anyway.

Restart the brain REPL and all three long-running units before trusting this
retire:

    sudo systemctl restart advisor brain-ollama-shim brain-web

Until that is done the retired page can still be served."""


# -- audit log ----------------------------------------------------------------
def _append_decision(record):
    """Append one decision to the audit log. Never raises."""
    try:
        os.makedirs(QUEUE_ROOT, exist_ok=True)
        record["ts"] = time.time()
        with open(DECISIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -- reading the queue --------------------------------------------------------
def _age_days(fetched_at):
    """Whole days since an ISO timestamp, or None when it is unusable."""
    try:
        then = datetime.datetime.fromisoformat(str(fetched_at))
        return (datetime.datetime.now() - then).days
    except Exception:
        return None


def _load_entry(directory, stem):
    """The .json record for one stem, or {} when it is missing or corrupt."""
    try:
        with open(os.path.join(directory, stem + ".json"), "r",
                  encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _stems_in(directory, ext):
    """Sorted stems of every ext file in directory; [] when unreadable."""
    try:
        return sorted(n[:-len(ext)] for n in os.listdir(directory)
                      if n.endswith(ext))
    except Exception:
        return []


def load_queue():
    """Every queued candidate as a dict with a 'stem' key, oldest fetch first.

    A queue member is defined by its .md note in WEB_QUEUE_DIR: the note is what
    promotion moves, and a stem with a JSON entry but no note has nothing to
    review. The entry is read from WEB_CACHE_DIR, which is the only place
    web_cache writes it, so a missing or corrupt JSON degrades the listing
    instead of hiding the note."""
    items = []
    for stem in _stems_in(WEB_QUEUE_DIR, ".md"):
        entry = _load_entry(WEB_CACHE_DIR, stem)
        entry["stem"] = stem
        items.append(entry)
    items.sort(key=lambda e: (str(e.get("fetched_at") or ""), e["stem"]))
    return items


def _first_body_line(md_path, entry):
    """The first real sentence of a note: the line after '## Fetched content',
    falling back to the JSON entry's stored text. Chrome lines (headings,
    provenance restatements, the generator comment) are skipped so the listing
    shows page content, which is the thing being judged."""
    skip = ("#", "<!--", "Source:", "Queries:", "Fetched:", "Tags:",
            "Related:", "Web content fetched", "---", "- - -")
    try:
        with open(md_path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
        seen_marker = False
        for line in lines:
            text = line.strip()
            if text.startswith("## Fetched content"):
                seen_marker = True
                continue
            if not seen_marker or not text:
                continue
            if text.startswith(skip):
                continue
            return text
    except Exception:
        pass
    return str((entry or {}).get("text") or "").strip()


def _status(entry):
    """Review-facing status. web_cache stamps 'snippet' when a note fell back to
    the DuckDuckGo blurb rather than the fetched page, which is exactly the case
    a reviewer should reject on sight."""
    return str((entry or {}).get("status") or "full")


def _host_of(url):
    """Host slug for display, matching the denylist's own key shape."""
    try:
        import web_cache
        return web_cache._host(url)
    except Exception:
        return ""


def _resolve(stem, directory, ext=".md"):
    """Accept an exact stem or an unambiguous prefix of one. Returns the stem,
    or '' when nothing matches or the prefix is ambiguous (the caller reports
    which). Prefix matching exists because the stems carry a 12-char sha1 tail
    that nobody wants to retype."""
    stems = _stems_in(directory, ext)
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


# -- index verification -------------------------------------------------------
def _index_path():
    """bm25.pkl's path, taken from update_index so there is one definition."""
    try:
        import update_index
        return update_index.DEFAULT_INDEX
    except Exception:
        return os.path.join(web_paths.BRAIN_ROOT, "index", "bm25.pkl")


def _indexed_chunk_texts(rel_source, index_path):
    """The BODIES of every chunk in the pickle that carries this vault-relative
    source, in index order. Returns None when the pickle cannot be read, which
    is a different answer from [] and the caller must not conflate them.

    Texts, not a count: a count answers 'is something here under this name',
    and a stale chunk left by an earlier version of the same page answers that
    question yes while the page's current content is nowhere in the index.
    Never raises."""
    try:
        with open(index_path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            return None
        chunks = data.get("chunks") or []
        target = rel_source.replace("\\", "/")
        return [str(c.get("text") or "") for c in chunks
                if str(c.get("source") or "").replace("\\", "/") == target]
    except Exception:
        return None


def _indexed_chunk_count(rel_source, index_path):
    """How many chunks carry this source, -1 when the pickle is unreadable.
    Kept because other callers may count, but promotion must NOT be verified
    with it: see _indexed_chunk_texts. Never raises."""
    texts = _indexed_chunk_texts(rel_source, index_path)
    return -1 if texts is None else len(texts)


def _expected_chunk_texts(rel_source, index_path):
    """The chunk texts update_index WOULD produce for the note on disk right
    now, or [] when they cannot be computed (no update_index, unreadable note,
    or a note that tokenizes to nothing). Reusing update_index._chunk_file is
    what makes the comparison exact: it is the same chunker, fed the same
    meta_mode the pickle records, so a match means the indexed chunks are this
    note's current content and not a leftover. Never raises."""
    try:
        import update_index
        meta_mode = "none"
        try:
            with open(index_path, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                meta_mode = str(data.get("meta_mode") or "none")
        except Exception:
            pass
        parts = rel_source.replace("\\", "/").split("/")
        path = os.path.join(VAULT_ROOT, *parts)
        parsed = update_index._chunk_file(path, rel_source, meta_mode)
        if not parsed:
            return []
        return [str(t) for t in (parsed[0] or [])]
    except Exception:
        return []


def _wait_for_lock(lock_path, seconds):
    """Poll until .update-lock is gone or the budget runs out. Returns True when
    the lock is free. update_index takes an O_EXCL lock and returns 0 when it
    skips on a held one, so a reviewer who does not wait sees a success message
    for a promote that never indexed."""
    deadline = time.time() + max(0, seconds)
    waited = False
    while True:
        try:
            if not os.path.exists(lock_path):
                return True
        except Exception:
            return True
        if time.time() >= deadline:
            return False
        if not waited:
            print("[web_review] another index update holds %s; waiting up to "
                  "%ds..." % (os.path.basename(lock_path), seconds))
            waited = True
        time.sleep(LOCK_POLL_SECONDS)


def _reindex_verify(rel_sources, want_present, attempts=2):
    """Run one incremental index update, then VERIFY BY CONTENT.

    rel_sources is the set of vault-relative note sources whose chunks must be
    present (want_present=True) or absent (False) afterwards. Timestamps are not
    consulted: AUTOLEARN can spawn a concurrent update_index that rewrites the
    same pickle, so an mtime+size check reports DONE for work this process never
    did. A status of 'no change' is equally untrustworthy, because it cannot
    distinguish 'already indexed' from 'your file is not under a managed dir at
    all'. Only the chunks in the pickle settle it.

    Present means present BY CONTENT: every chunk the current note chunks into
    has to be in the pickle under that source. Chunk-count-by-source alone is
    satisfied by a stale chunk from an earlier version of the same page, which
    is exactly the state a promote during a held lock leaves behind.

    A run that update_index SKIPPED on a held lock indexed nothing, so it can
    never be reported as verified even when the pickle happens to agree; the
    attempt is retried instead.

    Returns True when every source ended in the wanted state."""
    try:
        import update_index
    except Exception as exc:
        print("[web_review] cannot import update_index (%r); run it by hand."
              % exc)
        return False

    index_path = update_index.DEFAULT_INDEX
    lock_path = index_path + ".update-lock"
    all_skipped = True
    for attempt in range(1, max(1, attempts) + 1):
        if not _wait_for_lock(lock_path, LOCK_WAIT_SECONDS):
            print("[web_review] lock still held after %ds; calling "
                  "update_index anyway. If the lock is real it will skip, "
                  "and a skip is never reported as verified."
                  % LOCK_WAIT_SECONDS)
        try:
            code = update_index.run(vault=VAULT_ROOT, index_path=index_path,
                                    quiet=False)
        except Exception as exc:
            print("[web_review] reindex raised (%r)" % exc)
            code = 1
        if code != 0:
            print("[web_review] update_index exited %d." % code)
        # A lock skip returns 0 as well, so ask rather than infer. getattr keeps
        # this working against an older update_index that has no such helper.
        skipped = False
        try:
            skipped = bool(getattr(update_index, "was_skipped", lambda _rc: False)(code))
        except Exception:
            skipped = False
        if skipped:
            # Nothing was indexed by that call, so nothing it could have done is
            # verifiable. Retry rather than read the pickle: whatever is in
            # there was put there by somebody else, and reporting it as this
            # promote's work is the false success this branch exists to stop.
            print("[web_review] update_index SKIPPED on a held lock; nothing "
                  "was indexed by that call, so this attempt is NOT verified.")
            continue
        all_skipped = False

        disagree = []
        for src in sorted(rel_sources):
            found = _indexed_chunk_texts(src, index_path)
            if found is None:
                print("[web_review] could not read %s to verify; treat this "
                      "promote as UNVERIFIED." % index_path)
                return False
            if want_present:
                expected = _expected_chunk_texts(src, index_path)
                if not expected:
                    print("[web_review] cannot re-chunk %s to check its "
                          "content, so this promote is UNVERIFIED. Run "
                          "update_index.py by hand and re-check." % src)
                    return False
                missing = [t for t in expected if t not in found]
                if not missing:
                    print("[web_review] verified: %s is in the index by "
                          "content (%d of %d chunk(s) matched)."
                          % (src, len(expected), len(expected)))
                else:
                    disagree.append((src, len(found), len(expected) -
                                     len(missing), len(expected)))
            elif not found:
                print("[web_review] verified: %s is gone from the index." % src)
            else:
                disagree.append((src, len(found), 0, 0))
        if not disagree:
            return True
        for src, found_n, matched, wanted in disagree:
            if want_present:
                print("[web_review] attempt %d: %s has %d chunk(s) in the "
                      "index but only %d of its %d current chunk(s) match; the "
                      "rest is stale or missing."
                      % (attempt, src, found_n, matched, wanted))
            else:
                print("[web_review] attempt %d: %s still has %d chunk(s) in "
                      "the index, expected 0." % (attempt, src, found_n))
        rel_sources = [row[0] for row in disagree]

    if all_skipped:
        print("[web_review] every attempt was SKIPPED on a held .update-lock, "
              "so this process indexed nothing. Run update_index.py by hand "
              "once the other update finishes, then re-check.")
    else:
        print("[web_review] the index does NOT agree with the files on disk. "
              "Run update_index.py by hand and re-check.")
    return False


# -- shared purge primitive ---------------------------------------------------
def _purge_pair(stem, cache_dir, notes_dir, quiet=False):
    """Delete one stem's stored pair through refresh_web._purge.

    refresh_web owns this primitive and has been extended to reach the answer
    cache too, so calling it (rather than unlinking two paths here) is what
    keeps retire from being cosmetic: answer_cache entries carry
    PRINCIPAL_SOURCE_BONUS = 0.4 for the url they cite, so a stored answer
    outranks everything else on the very question the retire was meant to fix.
    notes_dir may be a single directory or a sequence of them, which
    refresh_web._purge normalizes either way; passing both note dirs is how one
    call removes a stem from the vault and the queue at once.
    Returns True when something was removed. Never raises."""
    try:
        import refresh_web
    except Exception as exc:
        print("[web_review] cannot import refresh_web (%r); nothing purged."
              % exc)
        return False
    try:
        return bool(refresh_web._purge(stem, cache_dir, notes_dir, quiet))
    except Exception as exc:
        print("[web_review] purge of %s failed (%r)" % (stem, exc))
        return False


# -- commands -----------------------------------------------------------------
def cmd_list(show_all):
    queue = load_queue()
    print("%d page(s) awaiting review in %s:" % (len(queue), WEB_QUEUE_DIR))
    if not queue:
        print("  (queue empty)")
    for entry in queue:
        stem = entry["stem"]
        url = str(entry.get("url") or "")
        age = _age_days(entry.get("fetched_at") or "")
        print("  %-40s %-22s age %3s d  %-8s %s" % (
            stem[:40], (_host_of(url) or "?")[:22],
            "?" if age is None else age, _status(entry), url))
        first = _first_body_line(os.path.join(WEB_QUEUE_DIR, stem + ".md"),
                                 entry)
        if first:
            print("      %s" % first[:100])
    if not show_all:
        return 0

    promoted = _stems_in(WEB_NOTES_DIR, ".md")
    print("\n%d promoted note(s) in %s:" % (len(promoted), WEB_NOTES_DIR))
    for stem in promoted:
        entry = _load_entry(WEB_CACHE_DIR, stem)
        url = str(entry.get("url") or "")
        age = _age_days(entry.get("fetched_at") or "")
        print("  %-40s %-22s age %3s d  %-8s %s" % (
            stem[:40], (_host_of(url) or "?")[:22],
            "?" if age is None else age, _status(entry), url))
    deny = web_paths.load_denylist()
    print("\ndenylist: %d stem(s), %d host(s)  (%s)" % (
        len(deny["stems"]), len(deny["hosts"]), web_paths.DENYLIST_PATH))
    return 0


def cmd_show(stem):
    resolved = _resolve(stem, WEB_QUEUE_DIR)
    directory = WEB_QUEUE_DIR
    if not resolved:
        # Fall back to a promoted note, so --show works on anything reviewable.
        resolved = _resolve(stem, WEB_NOTES_DIR)
        directory = WEB_NOTES_DIR
    if not resolved:
        print("No queued or promoted note matching %r (try --list --all)."
              % stem)
        return 1
    # The .md lives in the queue or the vault; the .json only ever lives in
    # WEB_CACHE_DIR, so the entry is read from there in both cases.
    entry = _load_entry(WEB_CACHE_DIR, resolved)
    url = str(entry.get("url") or "")
    print("=" * 72)
    print("stem    : " + resolved)
    print("where   : " + directory)
    print("url     : " + (url or "(no JSON entry found)"))
    print("host    : " + (_host_of(url) or "?"))
    print("query   : " + str(entry.get("query") or ""))
    print("fetched : %s (age %s d)" % (entry.get("fetched_at") or "?",
                                       _age_days(entry.get("fetched_at") or "")))
    print("chars   : %s   status: %s" % (entry.get("char_len") or "?",
                                         _status(entry)))
    print("denied  : %s" % ("YES" if url and web_paths.is_denied(url)
                            else "no"))
    print("=" * 72)
    try:
        with open(os.path.join(directory, resolved + ".md"), "r",
                  encoding="utf-8", errors="ignore") as fh:
            print(fh.read())
    except Exception as exc:
        print("(could not read the note: %r)" % exc)
    return 0


def _promote_one(stem, dry_run):
    """Move one queue pair into the managed vault dir and the web cache dir.

    Promotion is only a file move: update_index.py picks up anything that
    appears in vault/_WebCache and MANAGED_DIRS already scopes the automation,
    so neither indexer needs to know this tool exists. The JSON entry does not
    move, because web_cache already wrote it to WEB_CACHE_DIR, where refresh_web
    walks; it is rewritten in place so its md_path cross-link points at the
    vault copy instead of the queue one.
    Returns the vault-relative source on success, '' otherwise."""
    entry = _load_entry(WEB_CACHE_DIR, stem)
    url = str(entry.get("url") or "")
    src_md = os.path.join(WEB_QUEUE_DIR, stem + ".md")
    dest_md = os.path.join(WEB_NOTES_DIR, stem + ".md")
    rel_source = MANAGED_DIR_NAME + "/" + stem + ".md"

    if dry_run:
        print("  would promote %s -> %s" % (src_md, dest_md))
        if url and web_paths.is_denied(url):
            print("  would clear the denylist entry for " + url)
        return rel_source

    try:
        with open(src_md, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        print("  cannot read %s (%r); leaving it queued." % (src_md, exc))
        return ""

    web_paths.ensure_dirs()
    from web_cache import _atomic_write_text
    if not _atomic_write_text(dest_md, text):
        print("  could not write %s; leaving it queued." % dest_md)
        return ""

    # The entry's md_path is a cross-link consumers follow; it now points into
    # the vault, not the queue.
    if entry:
        entry = dict(entry)
        entry["md_path"] = dest_md
        entry.pop("stem", None)
        try:
            payload = json.dumps(entry, ensure_ascii=False, indent=2)
        except Exception:
            payload = ""
        if payload:
            _atomic_write_text(os.path.join(WEB_CACHE_DIR, stem + ".json"),
                               payload)

    # Approving a page overrides an earlier rejection of it. undeny clears the
    # host ban as well as the stem, so a whole domain is un-banned here; that is
    # web_paths.undeny's documented behaviour, not an accident of this call.
    if url and web_paths.is_denied(url):
        web_paths.undeny(url)
        print("  cleared the denylist entry (stem AND host) for " + url)

    # The queue copy of the note goes; the canonical .json in WEB_CACHE_DIR
    # stays (it was just rewritten above). A queue-side .json is only ever a
    # leftover from an older layout, so unlink it too and ignore its absence.
    for ext in (".md", ".json"):
        try:
            os.remove(os.path.join(WEB_QUEUE_DIR, stem + ext))
        except OSError:
            pass

    _append_decision({"action": "promote", "stem": stem, "url": url,
                      "note_path": dest_md})
    print("  promoted -> " + dest_md)
    return rel_source


def cmd_promote(stems, do_all, dry_run):
    targets = [e["stem"] for e in load_queue()] if do_all else []
    if not do_all:
        for raw in stems:
            resolved = _resolve(raw, WEB_QUEUE_DIR)
            if not resolved:
                print("No queued note matching %r (try --list)." % raw)
                return 1
            targets.append(resolved)
    if not targets:
        print("Nothing to promote; the queue is empty.")
        return 0

    promoted = []
    for stem in targets:
        print(stem)
        rel = _promote_one(stem, dry_run)
        if rel:
            promoted.append(rel)
    if dry_run:
        print("\n[dry run] %d note(s) would be promoted, then reindexed and "
              "verified by content." % len(promoted))
        return 0
    if not promoted:
        return 1
    print("\nFolding %d promoted note(s) into the index..." % len(promoted))
    if _reindex_verify(promoted, True):
        return 0
    # The move itself already happened; only the indexing is outstanding, so
    # say that plainly rather than leaving the reviewer to guess whether to
    # promote again (which would find nothing left in the queue).
    print("[web_review] the note(s) ARE in %s; only the index is behind. "
          "Re-run:  python3 update_index.py" % WEB_NOTES_DIR)
    return 1


def _reject_one(stem, reason, deny_host, dry_run):
    """Denylist a queued page's url, then remove its pair.

    The denylist is the whole point: without it the next fetch of the same url
    re-learns the page and the rejection has to be made again every week.

    Returns {'stem', 'url', 'denied', 'removed'} so the caller can report what
    actually happened instead of asserting that every reject was durable. A
    reject with denied=False purged files and nothing more."""
    entry = _load_entry(WEB_CACHE_DIR, stem)
    url = str(entry.get("url") or "")
    if not url:
        print("  no url recorded for %s; removing the pair but NOTHING is "
              "denied, so a refetch will re-queue it." % stem)
    if dry_run:
        print("  would deny %s (host=%s) and purge the pair"
              % (url or "(unknown url)", bool(deny_host)))
        return {"stem": stem, "url": url, "denied": bool(url),
                "removed": True}

    denied = web_paths.deny(url, reason, host=deny_host) if url else False
    if url and not denied:
        print("  WARNING: could not write %s; the rejection is NOT durable."
              % web_paths.DENYLIST_PATH)
    # One call, both note dirs: the .json only ever lives in WEB_CACHE_DIR, so
    # a second pass rooted at the queue would find no entry to read the url
    # from and the answer-cache half would run needle-less.
    removed = _purge_pair(stem, WEB_CACHE_DIR, (WEB_NOTES_DIR, WEB_QUEUE_DIR),
                          quiet=True)
    _append_decision({"action": "reject", "stem": stem, "url": url,
                      "reason": reason, "host": bool(deny_host),
                      "denied": bool(denied)})
    if denied:
        print("  rejected %s%s" % (url, " (whole host)" if deny_host else ""))
    else:
        print("  removed %s, but NOTHING was denied." % stem)
    return {"stem": stem, "url": url, "denied": bool(denied),
            "removed": bool(removed)}


def cmd_reject(stems, do_all, reason, deny_host, dry_run):
    targets = [e["stem"] for e in load_queue()] if do_all else []
    if not do_all:
        for raw in stems:
            resolved = _resolve(raw, WEB_QUEUE_DIR)
            if not resolved:
                print("No queued note matching %r (try --list)." % raw)
                return 1
            targets.append(resolved)
    if not targets:
        print("Nothing to reject; the queue is empty.")
        return 0
    denied = []
    purged_only = []
    failed = []
    for stem in targets:
        print(stem)
        result = _reject_one(stem, reason, deny_host, dry_run)
        if dry_run:
            continue
        if result.get("denied"):
            denied.append(stem)
        elif result.get("removed"):
            purged_only.append(stem)
        else:
            failed.append(stem)
    if dry_run:
        print("\n[dry run] %d page(s) would be denied and purged."
              % len(targets))
        return 0

    # Report what happened, not what the command is for. A page whose url could
    # not be recovered, or whose denylist write failed, was only unfiled: the
    # next fetch re-queues it.
    if denied:
        print("\n%d page(s) denied durably: web_cache consults %s before it "
              "persists anything." % (len(denied), web_paths.DENYLIST_PATH))
    if purged_only:
        print("\n%d page(s) were PURGED BUT NOT DENIED, so a refetch will "
              "re-queue them: %s" % (len(purged_only), ", ".join(purged_only)))
    if failed:
        print("\n%d page(s) were neither denied nor purged: %s"
              % (len(failed), ", ".join(failed)))
    return 0 if denied and not failed else 1


def cmd_retire(stems, reason, do_deny, deny_host, dry_run):
    """Remove an ALREADY-PROMOTED note from all three places it lives."""
    targets = []
    for raw in stems:
        resolved = _resolve(raw, WEB_NOTES_DIR)
        if not resolved:
            print("No promoted note matching %r (try --list --all)." % raw)
            # A stem can exist in the queue but not the vault: a re-fetch after a
            # promote recreates the queue copy, and retiring the vault copy first
            # leaves that one behind. Retire is scoped to promoted notes on
            # purpose, so name the command that does clean this state rather than
            # leaving the operator with a bare refusal.
            if _resolve(raw, WEB_QUEUE_DIR):
                print("  It IS in the review queue. To remove that copy and deny "
                      "the url: python3 web_review.py --reject %s" % raw)
            return 1
        targets.append(resolved)
    if not targets:
        print("Usage: web_review.py --retire STEM")
        return 2

    rel_sources = []
    for stem in targets:
        entry = _load_entry(WEB_CACHE_DIR, stem)
        url = str(entry.get("url") or "")
        print(stem + ("  " + url if url else ""))
        if dry_run:
            print("  would purge the note (vault AND queue copies), the JSON "
                  "entry, and every answer cache entry citing it")
            if do_deny:
                print("  would deny %s (host=%s)" % (url, bool(deny_host)))
            rel_sources.append(MANAGED_DIR_NAME + "/" + stem + ".md")
            continue
        if do_deny and url:
            if not web_paths.deny(url, reason, host=deny_host):
                print("  WARNING: could not write the denylist; a refetch can "
                      "re-learn this page.")
        # BOTH note dirs, which is what _purge's docstring promises and what
        # refresh_web.NOTE_DIRS means. A page can be promoted and then
        # refetched, which re-persists the queue .md next to the vault one;
        # purging only the vault copy leaves a queue note whose .json is gone,
        # so it comes back as a reviewable candidate reject can no longer deny.
        _purge_pair(stem, WEB_CACHE_DIR, (WEB_NOTES_DIR, WEB_QUEUE_DIR),
                    quiet=False)
        _append_decision({"action": "retire", "stem": stem, "url": url,
                          "reason": reason, "denied": bool(do_deny)})
        rel_sources.append(MANAGED_DIR_NAME + "/" + stem + ".md")

    if dry_run:
        print("\n[dry run] %d note(s) would be retired, then the index would "
              "be updated and verified by content." % len(targets))
        return 0

    print("\nRemoving %d retired note(s) from the index..." % len(rel_sources))
    ok = _reindex_verify(rel_sources, False)
    print(RESTART_NOTICE)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Review the web-fetch queue; promote approved pages into "
                    "the vault.")
    ap.add_argument("--list", action="store_true", help="list the queue")
    ap.add_argument("--show", metavar="STEM", help="show one record in full")
    ap.add_argument("--promote", metavar="STEM", nargs="*", default=None,
                    help="move queued note(s) into vault/_WebCache and index")
    ap.add_argument("--reject", metavar="STEM", nargs="*", default=None,
                    help="denylist the url(s) and purge the queue pair")
    ap.add_argument("--retire", metavar="STEM", nargs="*", default=None,
                    help="purge an already-promoted note from all three places")
    ap.add_argument("--all", action="store_true",
                    help="act on the whole queue (promote/reject), or show "
                         "promoted notes too (list)")
    ap.add_argument("--reason", default="",
                    help="reason recorded with a reject or retire")
    ap.add_argument("--host", action="store_true",
                    help="deny the whole host, not just this page")
    ap.add_argument("--deny", action="store_true",
                    help="with --retire, also denylist the url so a refetch "
                         "cannot re-learn it")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen and touch nothing")
    args = ap.parse_args(argv)

    if args.show:
        return cmd_show(args.show)
    if args.promote is not None:
        if not args.promote and not args.all:
            print("Usage: web_review.py --promote STEM [STEM ...]  (or --all)")
            return 2
        return cmd_promote(args.promote, args.all, args.dry_run)
    if args.reject is not None:
        if not args.reject and not args.all:
            print("Usage: web_review.py --reject STEM [STEM ...]  (or --all)")
            return 2
        return cmd_reject(args.reject, args.all, args.reason, args.host,
                          args.dry_run)
    if args.retire is not None:
        if not args.retire:
            print("Usage: web_review.py --retire STEM [STEM ...]")
            return 2
        return cmd_retire(args.retire, args.reason, args.deny, args.host,
                          args.dry_run)
    return cmd_list(args.all)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(0)
