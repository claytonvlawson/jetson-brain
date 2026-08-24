#!/usr/bin/env python3
"""Offline correctness test for web_review_digest.py: the empty-queue note,
a populated-queue note (age/status/excerpt/commands all present), and the
build_index.py prune that keeps _PendingReview out of the index.

Same sandbox convention as test_web_review.py: a throwaway BRAIN_ROOT under
/tmp, set BEFORE the first import of any brain module, so nothing under
/mnt/nvme or the real vault is ever touched.

Run: python3 test_web_review_digest.py
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = tempfile.mkdtemp(prefix="test_web_review_digest-")
os.environ["BRAIN_ROOT"] = ROOT

import web_paths                                    # noqa: E402
import web_cache                                     # noqa: E402
import web_review_digest                             # noqa: E402
import build_index                                    # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def reset_world():
    for path in (web_paths.QUEUE_ROOT, web_paths.VAULT_ROOT):
        shutil.rmtree(path, ignore_errors=True)
    web_paths.ensure_dirs()


def stage_queue_pair(url="https://example.org/asl-report",
                      text="First sentence of the fetched page body here.",
                      query="asl education outcomes"):
    web_cache.persist_web_context(query, [(url, text)])
    return web_cache._deterministic_stem(url)


# =============================================================================
# 1. Empty queue still writes a note, explicitly reporting zero pending.
# =============================================================================
reset_world()
text, count = web_review_digest.build_digest()
check("empty queue reports zero pending", count == 0)
check("empty queue note says nothing is pending",
      "Nothing pending review" in text)
check("empty queue note still has the Pending header",
      "## Pending (0)" in text)

rc = web_review_digest.main()
check("main() exits 0 on an empty queue", rc == 0)
check("main() writes the digest file",
      os.path.exists(web_review_digest.DIGEST_PATH))


# =============================================================================
# 2. A populated queue: the note carries title/url/query/age/status/excerpt
#    and both promote/reject commands for the stem.
# =============================================================================
reset_world()
stem = stage_queue_pair()
text, count = web_review_digest.build_digest()

check("populated queue reports one pending item", count == 1)
check("note includes the source url", "https://example.org/asl-report" in text)
check("note includes the query", "asl education outcomes" in text)
check("note includes an age label", "fetched today" in text)
check("note includes the fetch status", "Status:" in text)
check("note includes an excerpt of the body",
      "First sentence of the fetched page body here." in text)
check("note includes the promote command for this stem",
      ("web_review.py --promote %s" % stem) in text)
check("note includes the reject command for this stem",
      ("web_review.py --reject %s" % stem) in text)

rc = web_review_digest.main()
check("main() exits 0 on a populated queue", rc == 0)
with open(web_review_digest.DIGEST_PATH, "r", encoding="utf-8") as fh:
    on_disk = fh.read()
check("the file on disk matches build_digest()'s text", on_disk == text)


# =============================================================================
# 3. The digest note itself is pruned from the index the same way
#    Templates/ and _Fulltext are: build_index.build() must skip it.
# =============================================================================
reset_world()
stage_queue_pair()
web_review_digest.main()
check("the digest note exists inside _PendingReview before indexing",
      os.path.isfile(web_review_digest.DIGEST_PATH))

# One real vault note so the index isn't empty regardless of the prune.
real_note = os.path.join(web_paths.VAULT_ROOT, "Sources", "Real.md")
os.makedirs(os.path.dirname(real_note), exist_ok=True)
with open(real_note, "w", encoding="utf-8") as fh:
    fh.write("# Real note\n\nHand authored vault prose for the index.\n")

out_path = os.path.join(ROOT, "index", "bm25.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
rc = build_index.build(web_paths.VAULT_ROOT, out_path, embed=False,
                        no_embed=True)
check("build_index.build() succeeds", rc == 0)

import pickle
with open(out_path, "rb") as fh:
    data = pickle.load(fh)
sources = {str(c.get("source") or "") for c in data.get("chunks", [])}
check("no chunk source comes from _PendingReview",
      not any("_PendingReview" in s for s in sources))
check("the real vault note WAS indexed (the prune isn't over-broad)",
      any("Real.md" in s for s in sources))


# --- teardown ---------------------------------------------------------------
shutil.rmtree(ROOT, ignore_errors=True)

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
