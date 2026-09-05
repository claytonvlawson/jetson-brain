#!/usr/bin/env python3
"""
brain_web_note.py - read one vault note for brain_web's GET /note endpoint.

Kept out of brain_web.py so tests can import it without triggering the index
load that brain_web performs at import time. Pure functions, no server, no
Ollama.

    note_path(root, rel)      -> absolute path inside root, or None if rejected
    split_frontmatter(text)   -> (frontmatter, body)   (build_index's, re-exported)
    load_note(root, rel)      -> dict for the JSON response, or None

Path rules (note_path): the request path is vault-relative, must end in .md,
may not be absolute, may not contain a backslash or a ".." segment, and after
realpath must still sit under realpath(root). That last check is what stops a
symlink inside the vault from serving a file outside it.
"""
import os
import urllib.parse

from build_index import split_frontmatter, parse_frontmatter_fields, chunk_tags

MAX_BODY = 200 * 1024   # _Fulltext/ holds whole papers; the phone does not need all of one


def path_shape_ok(rel):
    """Syntactic check only: is this the shape of a vault-relative .md path?

    True does not mean the file exists; it means the request is well-formed
    enough to look up (brain_web answers 400 when this is False, 404 when the
    lookup then finds nothing).
    """
    if not rel:
        return False
    rel = urllib.parse.unquote(str(rel)).strip()
    if not rel or not rel.lower().endswith(".md"):
        return False
    if rel.startswith(("/", "~")) or "\\" in rel or "\x00" in rel:
        return False
    if any(seg in ("..", "", ".") for seg in rel.split("/")):
        return False
    return True


def note_path(root, rel):
    """Resolve a vault-relative note path safely. Returns abs path or None."""
    if not root or not path_shape_ok(rel):
        return None
    rel = urllib.parse.unquote(str(rel)).strip()
    real_root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(real_root, rel))
    if not candidate.startswith(real_root + os.sep):
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def load_note(root, rel):
    """Read a note and shape it for the /note JSON response. None if rejected."""
    path = note_path(root, rel)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    frontmatter, body = split_frontmatter(text)
    fields = parse_frontmatter_fields(frontmatter) if frontmatter else {}
    rel_clean = urllib.parse.unquote(str(rel)).strip()
    fallback = os.path.splitext(os.path.basename(rel_clean))[0]
    # Paper notes carry `title:`, People/ notes carry `name:`; the filename is
    # the last resort (it is the Obsidian display name anyway).
    title = (fields.get("extra", {}).get("title") or fields.get("name") or "").strip() or fallback
    truncated = len(body) > MAX_BODY
    if truncated:
        body = body[:MAX_BODY]
    return {
        "path": rel_clean,
        "title": title,
        "tags": chunk_tags(fields),
        "body": body.strip("\n"),
        "truncated": truncated,
    }
