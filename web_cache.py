#!/usr/bin/env python3
"""web_cache.py - dual-write persistence for DuckDuckGo-fetched web content.

Every excerpt the brain fetches from the web is written to disk TWICE:

  (1) a machine-readable JSON cache file under WEB_CACHE_DIR, keyed by
      URL + content hash, for fast dedup and reuse; and
  (2) a human-readable Obsidian note, written by default into the REVIEW QUEUE
      (WEB_QUEUE_DIR, outside the vault and therefore not indexed) and only
      into WEB_NOTES_DIR (under the build_index.py vault root, where the next
      build_index.py run folds it into the searchable index) when the caller
      asks for that directory explicitly. A url on the web_paths denylist is
      skipped entirely: neither file is written, which is what makes a reviewer's
      'reject' durable against the next fetch of the same page.

This mirrors answer_cache.py's idioms: standard-library only, a strict
never-raise-into-caller contract, atomic temp-file + os.replace writes,
os.makedirs(..., exist_ok=True) before every write, and kebab-slug + sha1
filenames. It does NOT touch answer_cache.py or its Q/A entries; this is raw
web-content caching (dedup + retrievability), a separate concern that lives in
its own cache/web subdirectory so AnswerCache's non-recursive os.listdir never
sees it.

One important indexing fact drives the note layout: build_index.py strips YAML
frontmatter BEFORE tokenizing, so nothing in the frontmatter (the source url,
the query, the title) is searchable. Anything that must be retrievable by BM25
is therefore ALSO restated in the note body. The frontmatter is provenance only.

No em dashes appear anywhere in the notes this module writes (a standing style
rule); the fetched text is sanitized to replace them before it lands on disk.

Written for the Jetson Orin Nano 8GB. The interpreter may be as old as 3.9, so
'from __future__ import annotations' keeps every annotation lazy and PEP 604
unions (X | None) are avoided in favor of Optional[...].
"""
from __future__ import annotations

import os
import re
import json
import hashlib
import tempfile
import datetime
from urllib.parse import urlparse


# -- Paths (owned by web_paths.py; this module never imports brain) -----------
# These names are IMPORTED, not redefined. A second definition of _WebCache is
# exactly how a retire deletes out of one directory while a refresh rewrites
# into another, so web_paths.py is the single source of truth and every name
# below is re-exported here only so existing importers (refresh_web.py,
# brain.py, research_note.py, the tests) keep working unchanged.
#
# WEB_NOTES_DIR lives under VAULT_ROOT and is therefore INDEXED: it holds notes
# a human approved. WEB_QUEUE_DIR lives outside the vault and is not indexed at
# all: it is where a fresh fetch lands, awaiting review. Writing to the queue is
# the default; promotion into the vault is a deliberate move by web_review.py.
from web_paths import (BRAIN_ROOT, VAULT_ROOT, WEB_CACHE_DIR, WEB_NOTES_DIR,
                       WEB_QUEUE_DIR, NOTE_TEXT_CAP, is_denied)

SLUG_MAX_LEN    = 48     # match answer_cache slug cap
HASH_PREFIX_LEN = 12     # sha1(url) prefix used in filenames + entry key
MAX_WEB_NOTES   = 2000   # cap on stored pairs; oldest are pruned past this

# The wikilink targets the notes attach to. They need not resolve to real files
# for indexing to succeed; unresolved targets are simply dropped by build_index.
_WIKILINK_MOC   = "_WebCache"
_WIKILINK_VAULT = "ThesisVault"


# -- private helpers (all never-raise) ----------------------------------------
def _sha1(text: str) -> str:
    """Hex sha1 of a string; '' on any failure."""
    try:
        return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()
    except Exception:
        return ""


def _now_iso() -> str:
    """Current local time as an ISO string (same shape as answer_cache)."""
    try:
        return datetime.datetime.now().isoformat()
    except Exception:
        return ""


def _now_human() -> str:
    """Human-readable fetched-at stamp for the note body. No em dashes."""
    try:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _host(url: str) -> str:
    """Netloc of a url, lowercased, leading 'www.' stripped, every non
    [a-z0-9] run collapsed to a single '-', edge '-' trimmed. Garbage or empty
    input yields 'unknown-host'. Never raises."""
    try:
        netloc = urlparse(str(url)).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        slug = re.sub(r"[^a-z0-9]+", "-", netloc).strip("-")
        return slug or "unknown-host"
    except Exception:
        return "unknown-host"


def _slugify(text: str, max_len: int = SLUG_MAX_LEN) -> str:
    """Lowercase kebab slug from the alnum tokens of text, truncated to max_len
    and right-stripped of '-'. Same character class as answer_cache. '' when the
    text has no alnum tokens. Never raises."""
    try:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        slug = "-".join(tokens)
        if len(slug) > max_len:
            slug = slug[:max_len].rstrip("-")
        return slug
    except Exception:
        return ""


def _deterministic_stem(url: str) -> str:
    """Shared filename stem for BOTH the .json and the .md of one excerpt:
    '<host>-<sha1(url)[:HASH_PREFIX_LEN]>'. The host keeps it human-scannable;
    the URL hash makes it URL-addressed, so ONE url maps to exactly one pair for
    its whole lifetime: a re-fetch of the same url with CHANGED content resolves
    to the same stem and os.replace updates it in place (no stale duplicate, no
    unbounded growth), while two distinct urls never collide even if their text
    is byte-identical. The changing content hash is carried as a field inside the
    JSON/frontmatter, not in the filename. Never raises; falls back to a bare hash
    prefix if the host is unusable."""
    try:
        prefix = (_sha1(url) or "")[:HASH_PREFIX_LEN] or "nohash"
        return "%s-%s" % (_host(url), prefix)
    except Exception:
        return "unknown-host-nohash"


def _atomic_write_text(path: str, data: str) -> bool:
    """Write data to path atomically: a temp file in the SAME directory, then
    os.replace (an atomic intra-filesystem rename). A crash mid-write leaves
    either the old complete file or the new complete one, never a truncated file
    that would corrupt the next build_index.py run. Stray temp files are cleaned
    up on failure. Returns True on success, False otherwise. Never raises."""
    tmp = None
    try:
        # "or ." because dirname("bare.md") is "", and makedirs("") raises
        # FileNotFoundError -- so a bare relative filename failed the write
        # here while succeeding in web_paths' copy of this same function.
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                # fsync is best-effort; a filesystem that rejects it must
                # not fail the write.
                pass
        os.replace(tmp, path)
        return True
    except Exception:
        if tmp is not None:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return False


def _sanitize_frontmatter_value(s: str) -> str:
    """Collapse newlines and runs of whitespace to single spaces and strip, so a
    url / query / title carrying a stray newline or a lone '---' line cannot
    break the YAML block and desync the index. Never raises."""
    try:
        return re.sub(r"\s+", " ", str(s)).strip()
    except Exception:
        return ""


def _no_dashes(s: str) -> str:
    """Replace em dash (U+2014) and en dash (U+2013) with a plain hyphen, per the
    standing no-em-dash rule. Applied to every string written into a note."""
    try:
        return str(s).replace("—", "-").replace("–", "-")
    except Exception:
        return ""


def _clean_body_text(text: str) -> str:
    """Prepare fetched text for the note body: strip em/en dashes and neutralize
    any line that is exactly '---' (which build_index would never read as
    frontmatter mid-body, but which we defuse anyway for cleanliness), then cap
    at NOTE_TEXT_CAP. Never raises."""
    try:
        cleaned = _no_dashes(text)
        # Defuse a bare '---' delimiter line anywhere in the fetched text.
        cleaned = re.sub(r"(?m)^\s*---\s*$", "- - -", cleaned)
        if len(cleaned) > NOTE_TEXT_CAP:
            cleaned = cleaned[:NOTE_TEXT_CAP]
        return cleaned
    except Exception:
        return ""


def _yaml_quote(s: str) -> str:
    """Double-quote a sanitized YAML scalar, escaping embedded quotes/backslashes
    so a value can never break the block. Assumes s is already newline-collapsed
    by _sanitize_frontmatter_value. Never raises."""
    try:
        safe = s.replace("\\", "\\\\").replace('"', '\\"')
        return '"%s"' % safe
    except Exception:
        return '""'


# -- note + entry templating --------------------------------------------------
def _render_note(query: str, url: str, final_url: str, title: str,
                 fetched_iso: str, body_text: str, status: str = "") -> str:
    """Build the .md note. Leading YAML frontmatter (provenance only, stripped
    before indexing) then a body that RESTATES the retrieval-load-bearing fields
    (title H1, source markdown link, the query, a #web-cache hashtag, wikilinks)
    so BM25 can actually find the page.

    status is the fetch quality marker, 'snippet' when the text is only the
    DuckDuckGo blurb rather than a real page fetch. It is written to the
    frontmatter for the reviewer AND restated in the body, because build_index
    strips frontmatter before tokenizing: a body line is the only version a
    search for thin notes can ever find. Never raises."""
    fm_title = _sanitize_frontmatter_value(_no_dashes(title))
    fm_query = _sanitize_frontmatter_value(_no_dashes(query))
    fm_url = _sanitize_frontmatter_value(url)
    fm_final = _sanitize_frontmatter_value(final_url)
    host = _host(url)
    heading = fm_title or host
    fm_status = _sanitize_frontmatter_value(_no_dashes(status)) or "full"
    status_tag = "web-snippet" if fm_status == "snippet" else "web-full"

    frontmatter = "\n".join([
        "---",
        "title: %s" % _yaml_quote(heading),
        "source: %s" % _yaml_quote(fm_url),
        "final_url: %s" % _yaml_quote(fm_final),
        "query: %s" % _yaml_quote(fm_query),
        "fetched_at: %s" % _yaml_quote(fetched_iso),
        "status: %s" % _yaml_quote(fm_status),
        "tags: [web, web-cache, %s]" % status_tag,
        "generator: %s" % _yaml_quote("web_cache.py"),
        "aliases: [%s]" % _yaml_quote(fm_url),
        "---",
    ])

    # Body. Everything here is tokenized by BM25; the frontmatter above is not.
    body = "\n".join([
        "",
        "<!-- AUTO-GENERATED BY web_cache.py. Do not hand-edit; overwritten on refetch. -->",
        "",
        "# %s" % heading,
        "",
        "Web content fetched for query: **%s**" % fm_query,
        "",
        "Source: [%s](%s)" % (host, fm_url),
        "Queries: %s" % fm_query,
        "Fetched: %s" % _now_human(),
        "Status: %s" % fm_status,
        "Tags: #web-cache #%s" % status_tag,
        "Related: [[%s]] [[%s]]" % (_WIKILINK_MOC, _WIKILINK_VAULT),
        "",
        "## Fetched content",
        "",
        body_text,
        "",
    ])
    return frontmatter + "\n" + body


class WebCache:
    """Dual-write persistence for one web-fetch batch. Constructed cheaply per
    batch (no in-memory index load, unlike AnswerCache, to keep memory flat on
    the Jetson). Every method is guarded and returns a safe default rather than
    raising, so the brain keeps answering even if the disk is full or read-only.

    Layout:
        cache_dir/<stem>.json   raw JSON record (dedup/reuse)
        notes_dir/<stem>.md     indexable vault note (retrievability)
    where <stem> = '<host>-<sha1(url)[:12]>' pairs the two files, and notes_dir
    defaults to the unindexed REVIEW QUEUE, never the vault. A caller that
    really wants to write straight into the indexed vault (refresh_web.py
    refreshing an already-approved note in place) has to name that directory.
    The stem is
    URL-addressed, so one url owns one pair for its lifetime: re-fetching the
    same url with changed content updates that pair in place rather than piling
    up stale copies. The store is bounded at MAX_WEB_NOTES pairs; oldest pairs
    are pruned past the cap so the vault and cache dirs cannot grow without end.
    """

    def __init__(self, cache_dir: str = WEB_CACHE_DIR,
                 notes_dir: str = WEB_QUEUE_DIR) -> None:
        self.cache_dir = cache_dir
        self.notes_dir = notes_dir
        # A dir that cannot be created leaves that half disabled but never
        # raises; the object is always usable (possibly a partial no-op).
        for directory in (cache_dir, notes_dir):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass

    def has(self, url: str, text: str) -> bool:
        """True iff this URL is already stored with byte-identical content, i.e.
        the JSON entry exists AND its recorded sha1 matches sha1(text). A url whose
        content has since changed returns False, so store_result knows to refresh
        it. Returns False on any error (fail-open: re-storing is idempotent and
        safe). Never raises."""
        try:
            if not (url and text):
                return False
            stem = _deterministic_stem(url)
            json_path = os.path.join(self.cache_dir, stem + ".json")
            if not os.path.exists(json_path):
                return False
            with open(json_path, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            return stored.get("sha1") == _sha1(text)
        except Exception:
            return False

    def store_result(self, query: str, url: str, text: str,
                     status: str = "") -> dict:
        """Persist ONE excerpt as the paired JSON entry + note. URL-addressed
        and idempotent: one url owns one pair, so a re-fetch with the SAME
        content is a no-op (short-circuit below) and a re-fetch with CHANGED
        content overwrites the pair in place (os.replace). status is the fetch
        quality marker ('snippet' for a DuckDuckGo blurb, '' for a real fetch).
        Returns a result dict describing what was written; never raises."""
        try:
            # 1. Guard inputs. Empty/whitespace text or url writes nothing.
            if not (url and text and str(text).strip()):
                return {}

            # 1a. The denylist gate, consulted BEFORE anything touches disk.
            #     A rejected url writes NEITHER file, so a reviewer's reject
            #     survives the next fetch of the same page instead of being
            #     silently re-learned. Reported, not swallowed, so a caller can
            #     tell "denied" apart from "failed".
            if is_denied(url):
                return {"stem": _deterministic_stem(url), "denied": True,
                        "json_path": "", "md_path": "", "skipped": True}

            # No caller supplies a title today; the note derives its heading from
            # the page host. Kept as an explicit field for the JSON schema.
            title_text = ""

            # 2. URL-address the pair (filename stem depends only on the url).
            content_sha1 = _sha1(text)
            stem = _deterministic_stem(url)

            # 3. Target paths (shared stem).
            json_path = os.path.join(self.cache_dir, stem + ".json")
            md_path = os.path.join(self.notes_dir, stem + ".md")

            # 3a. Short-circuit: identical content already on disk for this url.
            #     Skip the rewrite (and the disk churn) but report success.
            if self.has(url, text) and os.path.exists(md_path):
                return {"stem": stem, "sha1": content_sha1,
                        "json_path": json_path, "md_path": md_path,
                        "skipped": True}

            fetched_iso = _now_iso()
            body_text = _clean_body_text(text)

            # 4a. Write the vault note (atomic). Best-effort; a note failure must
            #     not stop the JSON record from being written.
            note = _render_note(query, url, url, title_text,
                                fetched_iso, body_text, status)
            md_ok = _atomic_write_text(md_path, note)

            # 4b. Write the JSON entry (atomic). Carries md_path as a cross-link.
            entry = {
                "url": url,
                "final_url": url,   # brain's fetch_page does not expose the
                                    # post-redirect url today; reserved field.
                "query": query if query is not None else "",
                "title": title_text,
                "sha1": content_sha1,
                "fetched_at": fetched_iso,
                "status": str(status or ""),
                "char_len": len(text),
                "text": text,
                "md_path": md_path,
                "schema": 1,
            }
            try:
                data = json.dumps(entry, ensure_ascii=False, indent=2)
            except Exception:
                data = ""
            json_ok = _atomic_write_text(json_path, data) if data else False

            result = {
                "stem": stem,
                "sha1": content_sha1,
                "json_path": json_path if json_ok else "",
                "md_path": md_path if md_ok else "",
                "skipped": False,
            }
            if not (json_ok and md_ok):
                result["error"] = True
            # Enforce the store cap after each successful new write. Best-effort;
            # a prune failure never affects what was just stored.
            self._prune()
            return result
        except Exception:
            return {}

    def _prune(self) -> None:
        """Keep at most MAX_WEB_NOTES pairs: when the cache dir holds more JSON
        entries than the cap, delete the oldest-mtime ones together with their
        matching .md note. Fully guarded so pruning can never raise into a
        caller, and so a single unlink failure does not abort the rest."""
        try:
            entries = []
            for name in os.listdir(self.cache_dir):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(self.cache_dir, name)
                try:
                    entries.append((os.path.getmtime(path), name[:-5]))
                except Exception:
                    continue
            overflow = len(entries) - MAX_WEB_NOTES
            if overflow <= 0:
                return
            entries.sort()   # oldest mtime first
            for _mtime, stem in entries[:overflow]:
                for path in (os.path.join(self.cache_dir, stem + ".json"),
                             os.path.join(self.notes_dir, stem + ".md")):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception:
            return

    def store_many(self, query: str, excerpts, statuses=None) -> list:
        """Persist a batch of (url, text) tuples. Malformed items (not a 2-tuple,
        None/empty url or text) are skipped individually so one bad excerpt never
        aborts the rest. Returns a per-item list of result dicts ({} for skipped
        items).

        statuses is an OPTIONAL side dict {url: 'snippet'}. It is a side dict and
        not a third tuple element on purpose: excerpts are unpacked as
        'for u, t in excerpts' in brain_skills.py and in _answer_from_excerpts,
        so widening the tuple would break every existing consumer. Never
        raises."""
        results = []
        try:
            if not excerpts:
                return results
            for item in excerpts:
                try:
                    if not item:
                        results.append({})
                        continue
                    # Accept 2-tuples/lists; tolerate longer sequences by taking
                    # the first two elements (url, text).
                    seq = list(item)
                    if len(seq) < 2:
                        results.append({})
                        continue
                    url, text = seq[0], seq[1]
                    status = ""
                    try:
                        if statuses:
                            status = statuses.get(url) or ""
                    except Exception:
                        status = ""
                    results.append(
                        self.store_result(query, url, text, status))
                except Exception:
                    results.append({})
            return results
        except Exception:
            return results


def persist_web_context(query, excerpts,
                        cache_dir: str = WEB_CACHE_DIR,
                        notes_dir: str = WEB_QUEUE_DIR,
                        statuses=None) -> None:
    """One-call hook for brain.gather_web_context. Constructs a WebCache and
    stores every (url, text) excerpt as a JSON + note pair. Swallows EVERYTHING
    (returns None) so the brain keeps working if the disk is full, read-only, or
    the target dir is missing. `excerpts` is exactly the list of (url, text)
    tuples gather_web_context already builds; `statuses` is the optional
    {url: 'snippet'} side dict described on store_many.

    notes_dir is EXPLICIT and defaults to the review queue: the caller decides
    queue versus vault, and a caller that decides nothing gets containment
    rather than an unreviewed write into the indexed vault. cache_dir /
    notes_dir are also how a test points this at a temp directory instead of the
    Jetson's /mnt tree."""
    try:
        WebCache(cache_dir, notes_dir).store_many(query, excerpts, statuses)
    except Exception:
        return


# The wiring design imports this name; keep it as an alias of the canonical
# entry point so either name works from brain.py.
persist_web_fetch = persist_web_context
