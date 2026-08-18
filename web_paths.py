#!/usr/bin/env python3
"""web_paths.py - the single definition of every web-loop path and cap.

Before this module the same values were written down four separate times:
brain.py (WEB_VAULT_DIR), web_cache.py (WEB_NOTES_DIR, NOTE_TEXT_CAP) and
refresh_web.py (PAGE_CHAR_CAP). Two copies of a path is how a retire deletes
from one directory while a refresh rewrites into another, so all of them now
import from here instead of redefining.

The module also owns the review queue and its denylist, because both are pure
path-and-policy concerns that several unrelated files (web_cache, refresh_web,
web_review, brain) have to agree on exactly.

Idioms match web_cache.py: standard library only, a strict never-raise-into-
caller contract on every persistence helper, atomic temp-file + os.replace
writes, and os.makedirs(..., exist_ok=True) before every write. Nothing is
imported at module import time beyond os/json/tempfile, so any file in the repo
can import this one safely and cheaply.

Python may be as old as 3.9 on the Jetson, so 'from __future__ import
annotations' keeps annotations lazy and PEP 604 unions (X | None) are avoided.
"""
from __future__ import annotations

import os
import json
import tempfile


# -- Roots ---------------------------------------------------------------------
# BRAIN_ROOT honours the env var exactly as update_index.py does, so a test or a
# Mac-side checkout can point the whole web loop at a scratch tree with one
# variable instead of monkeypatching four modules.
BRAIN_ROOT = os.environ.get("BRAIN_ROOT", "/mnt/nvme/brain")

VAULT_ROOT = os.path.join(BRAIN_ROOT, "vault")            # == build_index DEFAULT_VAULT
WEB_CACHE_DIR = os.path.join(BRAIN_ROOT, "cache", "web")  # raw JSON entries, never indexed
WEB_NOTES_DIR = os.path.join(VAULT_ROOT, "_WebCache")     # APPROVED, indexed .md notes

# The review queue holds fetched-but-unapproved pairs. It lives OUTSIDE
# VAULT_ROOT on purpose: build_index.py walks the vault with os.walk and indexes
# every .md it finds, so a queue directory placed anywhere under vault/ would be
# indexed the moment it was written and the containment would be a fiction.
# Under BRAIN_ROOT/queue it is unreachable by that walk, and promotion is then
# just a file move from QUEUE_ROOT into WEB_NOTES_DIR.
QUEUE_ROOT = os.path.join(BRAIN_ROOT, "queue")
WEB_QUEUE_DIR = os.path.join(QUEUE_ROOT, "_WebCache")

# Rejected stems and hosts, consulted BEFORE anything is persisted. Also outside
# the vault, for the same reason.
DENYLIST_PATH = os.path.join(QUEUE_ROOT, "denylist.json")

# Every directory this module manages, in creation order.
MANAGED_PATH_DIRS = (WEB_CACHE_DIR, WEB_NOTES_DIR, QUEUE_ROOT, WEB_QUEUE_DIR)


# -- Caps ----------------------------------------------------------------------
# PAGE_CHAR_CAP is PROMPT-facing and deliberately unchanged: every existing
# excerpt consumer (gather_web_context's returned tuples, the FETCH branch,
# brain_skills' fetch-url path) already assumes 2000 characters of context.
PAGE_CHAR_CAP = 2000

# STORE_CHAR_CAP is PERSISTENCE-facing and new: how much of a fetched page is
# kept on disk for later retrieval. Storage and prompting are separate budgets,
# so a note can carry far more than one answer's window shows.
STORE_CHAR_CAP = 10000

# The note body cap must never be the binding constraint ahead of
# STORE_CHAR_CAP, or notes would clip silently with nothing in the logs.
NOTE_TEXT_CAP = 10000

assert NOTE_TEXT_CAP >= STORE_CHAR_CAP, "NOTE_TEXT_CAP would silently clip stored notes"


def ensure_dirs():
    """makedirs(exist_ok=True) every managed directory. Returns True when all
    of them exist afterwards, False otherwise. Never raises."""
    ok = True
    for path in MANAGED_PATH_DIRS:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            ok = False
    return ok


# -- Atomic write --------------------------------------------------------------
def _atomic_write_text(path, data):
    """Write data to path atomically: a temp file in the SAME directory, then
    os.replace. A crash mid-write leaves either the old complete file or the new
    complete one, never a truncated denylist that would fail to parse and quietly
    un-reject everything. Returns True on success, False otherwise. Never raises.

    This duplicates web_cache._atomic_write_text on purpose: web_cache.py imports
    THIS module, so importing back the other way would be circular."""
    tmp = None
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                # fsync is best-effort; a filesystem that rejects it must not
                # fail the write.
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


# -- Denylist ------------------------------------------------------------------
# On-disk shape:
#   {"stems": {"<stem>": {"url": ..., "reason": ..., "at": ...}, ...},
#    "hosts": {"<host>": {"url": ..., "reason": ..., "at": ...}, ...}}
# The nested dicts carry provenance for the reviewer; the public load_denylist()
# hands back plain sets because that is all any hot-path caller needs.

def _identity(url):
    """(stem, host) for a url. web_cache is imported HERE, inside the function,
    not at module top level: web_cache.py imports web_paths, so a top-level
    import in this direction would be circular and would break every consumer of
    either module. Returns ('', '') if the import or the helpers fail."""
    try:
        import web_cache
        return (web_cache._deterministic_stem(url), web_cache._host(url))
    except Exception:
        return ("", "")


def _load_raw():
    """The denylist file as its nested-dict on-disk shape, with both keys always
    present. Missing, empty or corrupt files read as an empty denylist rather
    than an error, so a damaged file can never wedge the fetch path. Never
    raises."""
    empty = {"stems": {}, "hosts": {}}
    try:
        with open(DENYLIST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    out = {}
    for key in ("stems", "hosts"):
        section = data.get(key)
        out[key] = section if isinstance(section, dict) else {}
    return out


def load_denylist():
    """The denylist as {'stems': set, 'hosts': set}. Both keys are always
    present, both values are always sets, even when nothing is on disk. Never
    raises."""
    raw = _load_raw()
    try:
        return {"stems": set(raw.get("stems", {}).keys()),
                "hosts": set(raw.get("hosts", {}).keys())}
    except Exception:
        return {"stems": set(), "hosts": set()}


def is_denied(url):
    """True when url is rejected by stem OR by host. A host entry blocks every
    page on that site, which is how a whole retailer domain is dropped in one
    decision. An unusable url is never treated as denied. Never raises."""
    try:
        stem, host = _identity(url)
        if not stem and not host:
            return False
        deny = load_denylist()
        if stem and stem in deny["stems"]:
            return True
        if host and host in deny["hosts"]:
            return True
        return False
    except Exception:
        return False


def deny(url, reason="", host=False):
    """Add url to the denylist and atomically rewrite the file. By default only
    this one page's stem is rejected; pass host=True to reject its whole host as
    well. Existing entries are updated in place, so denying twice is a no-op on
    the set. Returns True when the file was written, False otherwise. Never
    raises."""
    try:
        stem, host_slug = _identity(url)
        if not stem and not host_slug:
            return False
        raw = _load_raw()
        record = {"url": str(url), "reason": str(reason or ""), "at": _now_iso()}
        if stem:
            raw["stems"][stem] = record
        if host and host_slug:
            raw["hosts"][host_slug] = dict(record)
        return _write_denylist(raw)
    except Exception:
        return False


def undeny(url):
    """Remove url from the denylist, by stem AND by host, and atomically rewrite
    the file. Removing something that is not listed is a successful no-op that
    still returns True. Returns False only when the rewrite itself failed. Never
    raises."""
    try:
        stem, host_slug = _identity(url)
        if not stem and not host_slug:
            return False
        raw = _load_raw()
        if stem:
            raw["stems"].pop(stem, None)
        if host_slug:
            raw["hosts"].pop(host_slug, None)
        return _write_denylist(raw)
    except Exception:
        return False


def _write_denylist(raw):
    """Serialize the nested-dict denylist and write it atomically. Never
    raises."""
    try:
        payload = {"stems": raw.get("stems", {}), "hosts": raw.get("hosts", {})}
        return _atomic_write_text(
            DENYLIST_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception:
        return False


def _now_iso():
    """Current local time as an ISO string; '' on any failure. datetime is
    imported inside the function to keep module import cost to os/json/tempfile
    as promised in the docstring."""
    try:
        import datetime
        return datetime.datetime.now().isoformat()
    except Exception:
        return ""
