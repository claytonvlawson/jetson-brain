#!/usr/bin/env python3
"""Offline correctness test for the web-fetch review queue: containment,
promote, reject (durable denylist), retire (all three places), and the
lock-skip case.

Nothing in this repo tested web_cache.py, update_index.py, or the promote path
before this file. It exercises web_paths, web_cache, web_review, refresh_web
and update_index against a throwaway BRAIN_ROOT under /tmp, so no network, no
Ollama, no ssh, and nothing under /mnt/nvme or the real vault is touched. The
seeded pickle deliberately carries NO dense block, which is what keeps
update_index.run() from trying to embed anything.

TWO PROCESS-LEVEL CONSTRAINTS, both load bearing:

  1. This file must NOT share a process with test_web_semantic.py. That test
     permanently rebinds brain.persist_web_fetch = None to prove the web path
     degrades without persistence; anything importing it afterwards would see a
     dead persistence hook. deploy.sh runs every test as its own process, which
     is what keeps the two apart. Do not merge them.
  2. web_paths.BRAIN_ROOT is read once, at import time, and every path constant
     in web_paths, web_cache, refresh_web, web_review and update_index is
     derived eagerly from it. So $BRAIN_ROOT is set BELOW, before the first
     import of any of them, and asserted afterwards. An import hoisted above
     that line would silently point this whole file at the live Jetson tree.

Run: python3 test_web_review.py
"""
from __future__ import annotations

import io
import os
import sys
import json
import pickle
import shutil
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- the sandbox, established BEFORE any brain module is imported ------------
ROOT = tempfile.mkdtemp(prefix="test_web_review-")
os.environ["BRAIN_ROOT"] = ROOT

import web_paths                                    # noqa: E402
import web_cache                                    # noqa: E402
import update_index                                 # noqa: E402
import refresh_web                                  # noqa: E402
import web_review                                   # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def run_quiet(fn, *args, **kwargs):
    """Call fn with stdout captured. Returns (result, printed_text). The CLI
    commands are chatty and their messages are part of what is under test."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# --- sandbox containment guard ----------------------------------------------
# If any of these is outside ROOT the process was imported wrong and the rest of
# the file would write into the live tree, so bail rather than "just" fail.
SANDBOXED = [web_paths.BRAIN_ROOT, web_paths.WEB_QUEUE_DIR,
             web_paths.WEB_NOTES_DIR, web_paths.WEB_CACHE_DIR,
             web_paths.DENYLIST_PATH, web_cache.WEB_QUEUE_DIR,
             refresh_web.ANSWER_CACHE_DIR, update_index.DEFAULT_INDEX,
             web_review.WEB_QUEUE_DIR]
if not all(str(p).startswith(ROOT + os.sep) or p == ROOT for p in SANDBOXED):
    print("FAIL sandbox: a module resolved a path outside %s" % ROOT)
    print("  " + "\n  ".join(str(p) for p in SANDBOXED))
    sys.exit(1)
check("every module resolved its paths under the temp BRAIN_ROOT", True)

INDEX_PATH = update_index.DEFAULT_INDEX
LOCK_PATH = INDEX_PATH + ".update-lock"
ANSWER_DIR = refresh_web.ANSWER_CACHE_DIR

URL = "https://example.com/hearing-aids-on-sale"
PAGE = ("Get one hundred dollars off a pair of Omni hearing aids. "
        "Additional research is cited further down the page. ")
MARKER = "Omni hearing aids"


def reset_world():
    """A clean sandbox plus a seeded index that holds exactly one hand authored
    chunk, so update_index.run() has a valid pickle to update in place."""
    for path in (web_paths.QUEUE_ROOT, web_paths.VAULT_ROOT,
                 os.path.join(ROOT, "cache"), os.path.join(ROOT, "index")):
        shutil.rmtree(path, ignore_errors=True)
    web_paths.ensure_dirs()
    for path in (web_paths.WEB_NOTES_DIR, ANSWER_DIR,
                 os.path.dirname(INDEX_PATH)):
        os.makedirs(path, exist_ok=True)
    data = {"chunks": [{"id": 0, "text": "hand authored vault prose",
                        "source": "Sources/Humphries2012.md",
                        "title": "Humphries 2012"}],
            "by_source": {"Sources/Humphries2012.md": [0]},
            "link_map": {}, "links_out": {}, "urls_out": {},
            "meta_mode": "none"}
    with open(INDEX_PATH, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def index_chunks():
    with open(INDEX_PATH, "rb") as fh:
        return pickle.load(fh)["chunks"]


def chunk_texts_for(stem):
    """Every indexed chunk BODY for one web note. Content, not mtime or size:
    a concurrent update_index rewrites the same pickle, so timestamps report
    DONE for work this process never did."""
    src = "_WebCache/" + stem + ".md"
    return [str(c.get("text") or "") for c in index_chunks()
            if str(c.get("source") or "") == src]


def stage_queue_pair(url=URL, text=PAGE, query="hearing aid prices"):
    """Persist one fetch exactly the way the live code does: defaults only.
    Returns the stem."""
    web_cache.persist_web_context(query, [(url, text)])
    return web_cache._deterministic_stem(url)


# =============================================================================
# 1. Containment: a fetch persisted with defaults lands in the queue, and the
#    vault is untouched. This is the whole point of the change; if the note
#    lands in vault/_WebCache the review queue is decoration.
# =============================================================================
reset_world()
stem = stage_queue_pair()
queued_md = os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")
vault_md = os.path.join(web_paths.WEB_NOTES_DIR, stem + ".md")

check("default persist writes the note into the queue dir",
      os.path.exists(queued_md))
check("default persist does NOT write into vault/_WebCache",
      not os.path.exists(vault_md))
vault_files = []
for base, _dirs, names in os.walk(web_paths.VAULT_ROOT):
    vault_files.extend(os.path.join(base, n) for n in names)
check("default persist leaves the whole vault root empty", vault_files == [])
check("WebCache's own default notes_dir is the queue, not the vault",
      web_cache.WebCache().notes_dir == web_paths.WEB_QUEUE_DIR)
check("the note body carries the fetched page text",
      MARKER in open(queued_md, encoding="utf-8").read())

# The queue note is invisible to the indexer because it is outside the vault
# walk, not because anything filters it. Prove that rather than assume it.
rc, _out = run_quiet(update_index.run, vault=web_paths.VAULT_ROOT,
                     index_path=INDEX_PATH, quiet=True)
check("an index update sees no web chunks while the note is queued",
      chunk_texts_for(stem) == [])


# =============================================================================
# 2. Promote: the note moves into the vault and its chunks are verifiably in
#    the pickle afterwards, asserted by CONTENT.
# =============================================================================
reset_world()
stem = stage_queue_pair()
rc, out = run_quiet(web_review.cmd_promote, [stem], False, False)

check("promote exits 0", rc == 0)
check("promote moved the note into vault/_WebCache",
      os.path.exists(os.path.join(web_paths.WEB_NOTES_DIR, stem + ".md")))
check("promote removed the note from the queue",
      not os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")))
promoted_texts = chunk_texts_for(stem)
check("promoted note has at least one chunk in the pickle",
      len(promoted_texts) >= 1)
check("the promoted chunks carry the page's actual text",
      any(MARKER in t for t in promoted_texts))
check("promote did not disturb the hand authored chunk",
      any(str(c.get("source")) == "Sources/Humphries2012.md"
          for c in index_chunks()))
check("promote reported a content verification, not a timestamp one",
      "verified:" in out)

# A dry run must touch nothing at all.
reset_world()
stem = stage_queue_pair()
rc, out = run_quiet(web_review.cmd_promote, [stem], False, True)
check("promote --dry-run leaves the note queued",
      os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md"))
      and not os.path.exists(os.path.join(web_paths.WEB_NOTES_DIR,
                                          stem + ".md")))
check("promote --dry-run indexes nothing", chunk_texts_for(stem) == [])


# =============================================================================
# 3. The lock-skip case. update_index.run() returns 0 both when it indexed and
#    when it skipped on a held .update-lock, so a promote during
#    brain-learn.service (or a fire and forget _spawn_index_update) used to
#    report success for a note that was never indexed.
# =============================================================================
reset_world()
stem = stage_queue_pair()
with open(LOCK_PATH, "w", encoding="utf-8") as fh:
    fh.write("99999")            # a fresh lock, well inside LOCK_STALE_SECONDS
saved_wait = web_review.LOCK_WAIT_SECONDS
web_review.LOCK_WAIT_SECONDS = 0     # do not sit through the real 90s budget
try:
    rc, out = run_quiet(web_review.cmd_promote, [stem], False, False)
finally:
    web_review.LOCK_WAIT_SECONDS = saved_wait
    os.remove(LOCK_PATH)

check("update_index.run() flags a lock skip",
      update_index.was_skipped(update_index.SKIPPED_LOCKED))
check("was_skipped is False for an ordinary 0", not update_index.was_skipped(0))
check("SKIPPED_LOCKED still compares equal to 0 for old callers",
      update_index.SKIPPED_LOCKED == 0)
check("a locked promote does NOT report success", rc != 0)
check("a locked promote says SKIPPED out loud", "SKIPPED" in out)
check("a locked promote indexed nothing", chunk_texts_for(stem) == [])
check("a locked promote still moved the file, and says so",
      os.path.exists(os.path.join(web_paths.WEB_NOTES_DIR, stem + ".md"))
      and "only the index is behind" in out)

# With the lock released the same note indexes, so the skip was the cause.
rc, out = run_quiet(web_review._reindex_verify,
                    ["_WebCache/" + stem + ".md"], True)
check("the same note indexes once the lock is gone",
      rc is True and any(MARKER in t for t in chunk_texts_for(stem)))


# --- 3b. The stale-chunk variant, which chunk COUNTING cannot survive. -------
# A promote of a page that was indexed once before leaves a chunk already
# sitting under the very source string the promote is about to verify. If the
# verifier asks "are there chunks under _WebCache/<stem>.md" it gets yes from
# the leftover and reports verified for an update_index call that indexed
# nothing at all. Verification has to be by CONTENT.
reset_world()
stem = stage_queue_pair()
STALE = "OLD STALE CONTENT from an earlier version of this same page."
with open(INDEX_PATH, "rb") as fh:
    seeded = pickle.load(fh)
seeded["chunks"].append({"id": 1, "text": STALE,
                         "source": "_WebCache/" + stem + ".md",
                         "title": "example.com"})
seeded["by_source"]["_WebCache/" + stem + ".md"] = [1]
with open(INDEX_PATH, "wb") as fh:
    pickle.dump(seeded, fh, protocol=pickle.HIGHEST_PROTOCOL)
check("stale-chunk setup: a chunk already carries the promoted source string",
      chunk_texts_for(stem) == [STALE])

with open(LOCK_PATH, "w", encoding="utf-8") as fh:
    fh.write("99999")
saved_wait = web_review.LOCK_WAIT_SECONDS
web_review.LOCK_WAIT_SECONDS = 0
try:
    rc, out = run_quiet(web_review.cmd_promote, [stem], False, False)
finally:
    web_review.LOCK_WAIT_SECONDS = saved_wait
    os.remove(LOCK_PATH)

check("a locked promote over a stale chunk exits non-zero", rc != 0)
check("a locked promote over a stale chunk never says verified",
      "verified:" not in out)
check("a locked promote over a stale chunk says SKIPPED out loud",
      "SKIPPED" in out)
check("the stale chunk is still all the index has, so nothing was indexed",
      chunk_texts_for(stem) == [STALE])

# And the same note, once the lock is gone, replaces the stale chunk and only
# then verifies. This is what proves the failure above was the lock, not an
# unpromotable note.
rc, out = run_quiet(web_review._reindex_verify,
                    ["_WebCache/" + stem + ".md"], True)
texts = chunk_texts_for(stem)
check("with the lock gone the stale chunk is replaced by the real content",
      rc is True and STALE not in texts and any(MARKER in t for t in texts))


# =============================================================================
# 4. Reject: the denylist is durable, and a later persist of the same url is
#    refused rather than re-learned.
# =============================================================================
reset_world()
stem = stage_queue_pair()

# Diagnostic first, because it explains every reject failure below. Reject can
# only deny a url it can find, and it reads that url from the stem's JSON
# entry. The pair is SPLIT ACROSS TWO ROOTS by design: web_cache writes the .md
# to notes_dir (the queue) and the .json to WEB_CACHE_DIR, which is a dedup
# cache deliberately kept out of AnswerCache's non-recursive listdir. So the
# queue is defined by which .md files exist in WEB_QUEUE_DIR, and the entry for
# a stem is always loaded from WEB_CACHE_DIR. Reading the entry from the queue
# dir is the reader-side bug, not a missing write.
check("web_cache writes the pair split: .md queued, .json in the dedup cache",
      os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md"))
      and os.path.exists(os.path.join(web_paths.WEB_CACHE_DIR, stem + ".json"))
      and not os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR,
                                          stem + ".json")))
check("web_review loads the entry from the dir web_cache writes it to",
      web_review._load_entry(web_paths.WEB_CACHE_DIR, stem).get("url") == URL)
queue_now = web_review.load_queue()
check("load_queue returns exactly the one queued stem",
      [e.get("stem") for e in queue_now] == [stem])
check("load_queue returns a POPULATED entry, not a bare stem",
      len(queue_now) == 1
      and queue_now[0].get("url") == URL
      and str(queue_now[0].get("fetched_at") or "") != ""
      and web_review._status(queue_now[0]) == "full")

rc, out = run_quiet(web_review.cmd_reject, [stem], False, "advertising", False,
                    False)

check("reject exits 0", rc == 0)
check("reject removed the queued note",
      not os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")))
check("reject wrote a denylist file on disk",
      os.path.exists(web_paths.DENYLIST_PATH))
check("the denylist is durable: a fresh read still denies the url",
      web_paths.is_denied(URL))

# The load bearing half. Without the gate the next fetch simply re-learns the
# page and the reviewer has to reject it again every week.
stage_queue_pair()
check("a later persist of a rejected url writes NO note",
      not os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")))
check("a later persist of a rejected url writes NO json entry",
      not os.path.exists(os.path.join(web_paths.WEB_CACHE_DIR,
                                      stem + ".json")))
denied = web_cache.WebCache().store_result("q", URL, PAGE)
check("store_result reports the refusal instead of swallowing it",
      denied.get("denied") is True and denied.get("md_path") == "")
check("an unrelated url is still persisted after the reject",
      web_cache.WebCache().store_result("q", "https://example.org/other",
                                        PAGE).get("md_path"))

# Host level reject bans every page on the host, which is what audicus needed.
reset_world()
stem = stage_queue_pair()
rc, out = run_quiet(web_review.cmd_reject, [stem], False, "retailer", True,
                    False)
sibling = "https://example.com/a-completely-different-page"
check("a host reject denies a sibling page on the same host",
      web_paths.is_denied(sibling))
check("a host reject does not deny another host",
      not web_paths.is_denied("https://gallaudet.edu/clerc"))


# =============================================================================
# 5. Retire: all three places are purged, the .md, the .json, and the answer
#    cache entries. Deleting the first two leaves answer_cache serving the
#    retired page with PRINCIPAL_SOURCE_BONUS favouring it.
# =============================================================================
reset_world()
stem = stage_queue_pair()
rc, out = run_quiet(web_review.cmd_promote, [stem], False, False)
check("retire setup: the note is promoted and indexed",
      rc == 0 and any(MARKER in t for t in chunk_texts_for(stem)))

# Three cached answers: one cites the url, one cites the note filename, one is
# unrelated and must survive.
answers = {
    "cites_url.json": {"question": "are these hearing aids any good",
                       "answer": "According to " + URL + " they are on sale.",
                       "sources": ["_WebCache/" + stem + ".md"]},
    "cites_note.json": {"question": "hearing aid prices",
                        "answer": "See the cached page.",
                        "sources": [stem + ".md"]},
    "unrelated.json": {"question": "what is language deprivation",
                       "answer": "Hall 2017 defines it as ...",
                       "sources": ["Sources/Hall2017.md"]},
}
for name, entry in answers.items():
    with open(os.path.join(ANSWER_DIR, name), "w", encoding="utf-8") as fh:
        json.dump(entry, fh)

rc, out = run_quiet(web_review.cmd_retire, [stem], "advertising", False, False,
                    False)

check("retire exits 0", rc == 0)
check("retire removed the vault note",
      not os.path.exists(os.path.join(web_paths.WEB_NOTES_DIR, stem + ".md")))
check("retire removed the json cache entry",
      not os.path.exists(os.path.join(web_paths.WEB_CACHE_DIR,
                                      stem + ".json")))
check("retire removed the answer citing the url",
      not os.path.exists(os.path.join(ANSWER_DIR, "cites_url.json")))
check("retire removed the answer citing the note filename",
      not os.path.exists(os.path.join(ANSWER_DIR, "cites_note.json")))
check("retire kept the unrelated cached answer",
      os.path.exists(os.path.join(ANSWER_DIR, "unrelated.json")))
check("retire removed the note's chunks from the index",
      chunk_texts_for(stem) == [])
check("retire left the hand authored chunk alone",
      any(str(c.get("source")) == "Sources/Humphries2012.md"
          for c in index_chunks()))
check("retire prints the restart instruction as REQUIRED",
      "REQUIRED" in out and "systemctl restart" in out)

# Retire has to sweep the QUEUE as well as the vault. A promoted page whose url
# is fetched again gets its pair re-persisted, which recreates the queue .md
# (store_result's identical-content short circuit needs the note to still be
# there, and promotion moved it away). Retiring only the vault copy leaves that
# queue copy behind, and the retired page walks straight back in at the next
# review as an unreviewed candidate.
reset_world()
stem = stage_queue_pair()
rc, out = run_quiet(web_review.cmd_promote, [stem], False, False)
q_md = os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")
v_md = os.path.join(web_paths.WEB_NOTES_DIR, stem + ".md")
stage_queue_pair()          # the refetch, no deny in play
check("queue-copy setup: a refetch after promote recreates the queue note",
      rc == 0 and os.path.exists(q_md) and os.path.exists(v_md))

rc, out = run_quiet(web_review.cmd_retire, [stem], "advertising", False, False,
                    False)
check("retire with a queue copy present exits 0", rc == 0)
check("retire removed the QUEUE copy, not just the vault copy",
      not os.path.exists(q_md))
check("retire removed the vault copy in the same call",
      not os.path.exists(v_md))
check("retire removed the json entry from the dedup cache",
      not os.path.exists(os.path.join(web_paths.WEB_CACHE_DIR,
                                      stem + ".json")))
check("the retired stem is gone from the review queue",
      [e.get("stem") for e in web_review.load_queue()] == [])


# --retire --deny is the audicus case: purge AND make sure a refetch cannot
# re-learn the page.
reset_world()
stem = stage_queue_pair()
run_quiet(web_review.cmd_promote, [stem], False, False)
rc, out = run_quiet(web_review.cmd_retire, [stem], "retailer", True, False,
                    False)
check("retire --deny denies the url", web_paths.is_denied(URL))
stage_queue_pair()
check("retire --deny stops a refetch from re-queueing the page",
      not os.path.exists(os.path.join(web_paths.WEB_QUEUE_DIR, stem + ".md")))


# =============================================================================
# 6. The weekly refresher must not resurrect queued notes into the vault.
#    brain-learn.timer fires refresh_web on notes older than 30 days; before
#    the queue split its default target was vault/_WebCache, which would have
#    auto promoted content nobody approved.
# =============================================================================
reset_world()
stem = stage_queue_pair()
check("refresh_web searches both the vault and the queue",
      set(refresh_web.NOTE_DIRS) == {web_paths.WEB_NOTES_DIR,
                                     web_paths.WEB_QUEUE_DIR})
check("a queued note resolves to the queue dir, never the vault",
      refresh_web._note_dir(stem) == web_paths.WEB_QUEUE_DIR)
promoted_stem = "gallaudet-edu-0000deadbeef"
open(os.path.join(web_paths.WEB_NOTES_DIR, promoted_stem + ".md"),
     "w", encoding="utf-8").write("# promoted\n")
check("a promoted note resolves to the vault dir",
      refresh_web._note_dir(promoted_stem) == web_paths.WEB_NOTES_DIR)


# --- teardown ---------------------------------------------------------------
shutil.rmtree(ROOT, ignore_errors=True)

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
