#!/usr/bin/env python3
"""update_index.py - incremental index update for web-learned knowledge.

The self-updating half of the brain's learning loop. web_cache.py already
persists every web fetch as an indexable note under vault/_WebCache, but until
now those notes only became retrievable at the NEXT full build_index.py run.
This script folds them in incrementally, in seconds, so the brain that just
searched the web can answer from that knowledge on the very next question
(brain.py hot-reloads the index when the pickle's mtime changes).

Scope is deliberately narrow: ONLY sources under the _WebCache/ folder are
added, replaced, or removed. Hand-authored vault notes are never touched here;
they still flow through the full build_index.py rebuild at deploy time. That
containment is the safety story: automatic learning cannot corrupt or displace
the curated vault, and purging a bad web note (refresh_web.py --purge) followed
by one run of this script removes every trace of it from retrieval.

Mechanics per run:
  1. Take a lock file next to the pickle (skip silently if another update is
     mid-flight; a stale lock older than 10 minutes is broken).
  2. Load bm25.pkl and diff the _WebCache chunks in it against the .md files
     on disk: new file -> add, changed file -> replace, deleted file -> drop.
  3. Re-chunk with build_index.py's own chunk_note/tokenize (one chunking
     implementation, no drift) and register link_map/urls_out entries the same
     way build() does.
  4. If the pickle carries a dense block, embed ONLY the new chunks with the
     block's own model and append the vectors; rows of dropped chunks are
     removed. Embedding failure just leaves new chunks dense-invisible (BM25
     still finds them), exactly like build_index's own contract.
  5. Write the pickle atomically (temp file + os.replace in the same dir) so a
     reader never sees a truncated index.

Python 3.9 compatible (the Jetson may run 3.9): lazy annotations, no PEP 604.
"""
from __future__ import annotations

import os
import sys
import time
import array
import pickle
import tempfile
import argparse
import contextlib
import io

import build_index as bi

BRAIN_ROOT = os.environ.get("BRAIN_ROOT", "/mnt/nvme/brain")
DEFAULT_VAULT = os.path.join(BRAIN_ROOT, "vault")
DEFAULT_INDEX = os.path.join(BRAIN_ROOT, "index", "bm25.pkl")
WEB_DIR_NAME = "_WebCache"
LOCK_STALE_SECONDS = 600


def _web_prefix():
    return WEB_DIR_NAME + "/"


def _is_web_source(source):
    s = str(source or "").replace("\\", "/")
    return s.startswith(_web_prefix())


def _acquire_lock(lock_path):
    """O_EXCL lock file so two updates (REPL trigger + timer) never interleave
    a pickle write. Returns True when the lock is ours. A lock older than
    LOCK_STALE_SECONDS is treated as a crash leftover and broken."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii", "replace"))
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock_path) > LOCK_STALE_SECONDS:
                os.remove(lock_path)
                return _acquire_lock(lock_path)
        except OSError:
            pass
        return False
    except OSError:
        return False


def _release_lock(lock_path):
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _chunk_file(path, rel_source):
    """Chunk one web note exactly the way build_index.build() does. Returns
    (chunk_texts, frontmatter, body) or None when the file is unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
    except OSError:
        return None
    fallback_title = os.path.splitext(os.path.basename(rel_source))[0]
    frontmatter, body_no_fm = bi.split_frontmatter(raw)
    title = bi.extract_title(body_no_fm, fallback_title)
    texts = []
    for chunk_text in bi.chunk_note(body_no_fm, title, fallback_title):
        if bi.tokenize(chunk_text):
            texts.append(chunk_text)
    return texts, frontmatter, body_no_fm, title


def run(vault=DEFAULT_VAULT, index_path=DEFAULT_INDEX, quiet=False,
        ollama_host=bi.DEFAULT_OLLAMA_HOST):
    """Perform one incremental update. Returns an exit code (0 also for a
    clean no-op) and never leaves a partial pickle behind."""
    def say(msg):
        if not quiet:
            print(msg)

    web_dir = os.path.join(vault, WEB_DIR_NAME)
    if not os.path.exists(index_path):
        say("No index at %s; run build_index.py first." % index_path)
        return 1

    lock_path = index_path + ".update-lock"
    if not _acquire_lock(lock_path):
        say("Another update is running; skipping.")
        return 0

    try:
        try:
            with open(index_path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:
            say("Could not read index: %s" % exc)
            return 1
        if not isinstance(data, dict) or not data.get("chunks"):
            say("Index has an unexpected format; refusing to touch it.")
            return 1

        chunks = data["chunks"]
        by_source = data.get("by_source") or {}
        link_map = data.get("link_map") or {}
        links_out = data.get("links_out") or {}
        urls_out = data.get("urls_out") or {}

        # ---- Diff: index's web chunks vs the notes on disk ---------------
        indexed = {}   # rel_source -> [chunk texts in order]
        for c in chunks:
            src = str(c.get("source") or "")
            if _is_web_source(src):
                indexed.setdefault(src, []).append(c.get("text") or "")

        on_disk = {}   # rel_source -> full path
        if os.path.isdir(web_dir):
            for name in sorted(os.listdir(web_dir)):
                if name.lower().endswith(".md"):
                    on_disk[_web_prefix() + name] = os.path.join(web_dir, name)

        removed = [s for s in indexed if s not in on_disk]
        fresh = {}     # rel_source -> (texts, frontmatter, body, title)
        for src, path in on_disk.items():
            parsed = _chunk_file(path, src)
            if parsed is None:
                continue
            if src in indexed and indexed[src] == parsed[0]:
                continue   # unchanged: keep its chunks (and dense rows) as-is
            fresh[src] = parsed

        if not removed and not fresh:
            say("Web knowledge already up to date (%d web notes indexed)."
                % len(indexed))
            return 0

        # ---- Drop chunks of removed/replaced sources ---------------------
        stale_sources = set(removed) | set(fresh)
        dropped_ids = set()
        kept_chunks = []
        for c in chunks:
            src = str(c.get("source") or "")
            if src in stale_sources:
                dropped_ids.add(c.get("id"))
            else:
                kept_chunks.append(c)
        for src in stale_sources:
            by_source.pop(src, None)
            links_out.pop(src, None)
            urls_out.pop(src, None)
        if dropped_ids or stale_sources:
            link_map = {k: v for k, v in link_map.items()
                        if v not in stale_sources}

        # ---- Append chunks for new/changed notes -------------------------
        next_id = max((c.get("id", -1) for c in chunks), default=-1) + 1
        new_chunks = []
        for src in sorted(fresh):
            texts, frontmatter, body, title = fresh[src]
            ids_for_src = []
            for text in texts:
                chunk = {"id": next_id, "text": text,
                         "source": src, "title": title}
                new_chunks.append(chunk)
                ids_for_src.append(next_id)
                next_id += 1
            if ids_for_src:
                by_source[src] = ids_for_src
            # Same link_map registration as build(): stem, path-no-ext, aliases;
            # first writer wins so hand-authored notes always outrank web notes.
            stem = os.path.splitext(os.path.basename(src))[0]
            rel_no_ext = os.path.splitext(src)[0]
            for key in [stem, rel_no_ext] + bi.parse_aliases(frontmatter):
                nk = bi.normalize_key(key)
                if nk and nk not in link_map:
                    link_map[nk] = src
            urls = bi.extract_urls(body)
            if urls:
                urls_out[src] = urls
            resolved, seen = [], set()
            for target in bi.extract_wikilink_targets(body):
                dest = link_map.get(bi.normalize_key(target))
                if dest is None:
                    dest = link_map.get(
                        bi.normalize_key(os.path.basename(target)))
                if dest is None or dest == src or dest in seen:
                    continue
                seen.add(dest)
                resolved.append(dest)
            if resolved:
                links_out[src] = resolved

        kept_chunks.extend(new_chunks)

        # ---- Dense block maintenance -------------------------------------
        dense = data.get("dense")
        if dense is not None:
            ids = list(dense.get("ids") or [])
            buf = dense.get("buf")
            dim = dense.get("dim")
            if dropped_ids and ids and buf is not None and dim:
                new_ids = []
                new_buf = array.array("f")
                for r, cid in enumerate(ids):
                    if cid in dropped_ids:
                        continue
                    new_ids.append(cid)
                    base = r * dim
                    new_buf.extend(buf[base:base + dim])
                ids, buf = new_ids, new_buf
            if new_chunks and dim:
                # Embed only the additions with the block's own model. Chatty
                # build_dense prints are suppressed in quiet mode.
                sink = io.StringIO() if quiet else sys.stdout
                with contextlib.redirect_stdout(sink):
                    block = bi.build_dense(new_chunks, dense.get("model"),
                                           ollama_host)
                if block and block.get("dim") == dim:
                    ids.extend(block["ids"])
                    if buf is None:
                        buf = array.array("f")
                    buf.extend(block["buf"])
                else:
                    say("NOTE: new web chunks are dense-invisible "
                        "(embedder unavailable); BM25 still retrieves them.")
            dense["ids"] = ids
            if buf is not None:
                dense["buf"] = buf
            data["dense"] = dense

        data["chunks"] = kept_chunks
        data["by_source"] = by_source
        data["link_map"] = link_map
        data["links_out"] = links_out
        data["urls_out"] = urls_out

        # ---- Atomic write ------------------------------------------------
        out_dir = os.path.dirname(index_path) or "."
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, index_path)
        except Exception as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            say("Could not write index: %s" % exc)
            return 1

        say("Web knowledge folded in: +%d note(s) (%d chunk(s)), "
            "-%d removed, %d total chunks."
            % (len(fresh), len(new_chunks), len(removed), len(kept_chunks)))
        return 0
    finally:
        _release_lock(lock_path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fold _WebCache notes into bm25.pkl incrementally.")
    ap.add_argument("--vault", default=DEFAULT_VAULT)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--ollama-host", default=bi.DEFAULT_OLLAMA_HOST)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return run(vault=args.vault, index_path=args.index, quiet=args.quiet,
               ollama_host=args.ollama_host)


if __name__ == "__main__":
    sys.exit(main())
