"""Citation-discipline tools for the Jetson agent, bound to the SAME vault
brain.py already indexes (SCHOLAR_VAULT_DIR / BRAIN_VAULT).

This is a Jetson-native sibling of ThesisVault's LanguageDeprivationAgent/
agent/vault_tools.py -- same four verbs (search_vault, get_note, list_by_tag,
cite), reimplemented against brain.py's pickle+BM25 index and plain vault
walk instead of that agent's SQLite/FTS5 table, so it needs no second index
and no new parsing dependency. It reuses build_index.py's own frontmatter
parser (split_frontmatter / parse_frontmatter_fields) rather than adding a
PyYAML dependency for a second, competing parse of the same files.

vault_agent.py, mcp_server.py, and the Mac Claude-Opus path are untouched by
this file -- it is a parallel on-device implementation, not a migration.
"""

import os

import brain
import build_index

_JUNK_DIR_NAMES = build_index._JUNK_DIR_NAMES


def _vault_root():
    return brain.SCHOLAR_VAULT_DIR


def _walk_markdown_files():
    """Yield (full_path, rel_path) for every non-junk .md file in the vault."""
    root = _vault_root()
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _JUNK_DIR_NAMES
                   and not d.startswith(".")]
        for fn in files:
            if fn.lower().endswith(".md"):
                full = os.path.join(r, fn)
                yield full, os.path.relpath(full, root)


def _read_note(full_path):
    try:
        with open(full_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        return None


def _frontmatter_fields(markdown_text):
    fm_text, _body = build_index.split_frontmatter(markdown_text)
    return build_index.parse_frontmatter_fields(fm_text)


def get_note(name):
    """Return {"name", "path", "markdown", "frontmatter"} for a note found by
    filename (with or without .md) or by a YAML `aliases:` entry. Never
    raises; returns {"error": ...} when nothing matches."""
    target = name[:-3] if name.lower().endswith(".md") else name
    target_lower = target.lower()

    alias_hit = None
    for full, rel in _walk_markdown_files():
        stem = os.path.splitext(os.path.basename(full))[0]
        if stem.lower() == target_lower:
            md = _read_note(full)
            if md is None:
                continue
            return {"name": stem, "path": rel, "markdown": md,
                    "frontmatter": _frontmatter_fields(md)}
        if alias_hit is None:
            md = _read_note(full)
            if md is None:
                continue
            fm = _frontmatter_fields(md)
            aliases = fm.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if name in aliases or target in aliases:
                alias_hit = {"name": stem, "path": rel, "markdown": md,
                              "frontmatter": fm, "resolved_via_alias": name}

    if alias_hit is not None:
        return alias_hit
    return {"error": "Note not found: %s" % name}


def list_by_tag(tag):
    """Return {"tag", "count", "notes": [{"note", "path"}]} for every note
    whose frontmatter `tags:` list contains `tag` (exact match)."""
    matches = []
    for full, rel in _walk_markdown_files():
        md = _read_note(full)
        if md is None:
            continue
        fm = _frontmatter_fields(md)
        tags = fm.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if tag in tags:
            matches.append({"note": os.path.splitext(os.path.basename(full))[0],
                             "path": rel})
    return {"tag": tag, "count": len(matches), "notes": matches}


_STYLES = ("apa", "acm", "ieee")


def cite(source_id, style="apa"):
    """Format a citation from a note's frontmatter (authors/year/venue/doi),
    falling back to title-only when those fields are absent. Mirrors
    LanguageDeprivationAgent/agent/vault_tools.py's cite() output shape so a
    caller can treat either agent's citations the same way."""
    hit = get_note(source_id)
    if "error" in hit:
        return {"error": hit["error"]}
    if style not in _STYLES:
        style = "apa"

    fm = hit["frontmatter"]
    title = hit["name"]
    authors = fm.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    year = fm.get("year") or ""
    venue = fm.get("venue") or ""
    doi = fm.get("doi") or ""

    tier = "standard" if authors else "title-only"

    if style == "apa":
        author_str = ", ".join(authors) if authors else title
        formatted = "%s (%s). %s." % (author_str, year, title)
        if venue:
            formatted += " %s." % venue
        if doi:
            formatted += " https://doi.org/%s" % doi
    elif style == "acm":
        formatted = "%s. %s. %s. %s." % (", ".join(authors) or title, year,
                                          title, venue)
    else:  # ieee
        formatted = "[1] %s, \"%s,\" %s, %s." % (
            ", ".join(authors) or title, title, venue, year)

    return {"source_id": source_id, "style": style, "citation": formatted,
            "authority_tier": tier, "path": hit["path"]}


def search_vault(question, bm25, chunks, meta, k=8, persona="default"):
    """Thin adapter over brain.retrieve_context() -- no retrieval logic is
    reimplemented here. Returns a list of {"source", "title", "text"} for the
    top-k retrieved chunks, in brain.py's own ranked order."""
    result = brain.retrieve_context(question, bm25, chunks, meta,
                                     persona=persona)
    out = []
    for chunk in result["retrieved"][:k]:
        out.append({
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
        })
    return out
