#!/usr/bin/env python3
"""research_note.py - renders research-agent candidates as ThesisVault notes.

The output MUST match the hand-authored Sources/ schema so the vault stays
homogeneous (name: not title:, type: source, tags list, authors as a YAML
list, year, venue, doi, citekey), plus provenance keys the review CLI reads
(status: candidate until approved, access, score, discovery_source).

Two indexing facts drive the layout, both inherited from web_cache.py:
  1. build_index.py strips YAML frontmatter BEFORE tokenizing, so every
     retrieval-load-bearing field is RESTATED in the body.
  2. A stray '---' line inside fetched text would confuse humans (never the
     indexer), so body text is cleaned the same way web notes are.

Reuses web_cache.py's hardened primitives (_atomic_write_text, sanitizers,
the no-em-dash rule). Never raises into the caller. Python 3.9 compatible.
"""
from __future__ import annotations

import os
import re

from web_cache import (_atomic_write_text, _sanitize_frontmatter_value,
                       _no_dashes, _yaml_quote, _clean_body_text)

# Same forbidden-character class papers.py uses for Obsidian filenames.
_SAFE_NAME_RE = re.compile(r'[\\/:#^|\[\]?*"<>]')

# Wikilink targets for the Related section. Unresolved targets are simply
# dropped by build_index, so these are safe even before the hub notes exist.
FOCUS_AUTHOR_LINKS = {
    "hall": "Wyatte Hall",
    "de meulder": "Maartje De Meulder",
    "caselli": "Naomi Caselli",
    "ballard": "Carey Ballard",
}
MOC_LINK = "Language Deprivation"
ABSTRACT_CAP = 3000


def safe_name(text, max_len=90):
    """Obsidian-safe filename fragment (no forbidden chars, no em dashes)."""
    try:
        name = _SAFE_NAME_RE.sub("", _no_dashes(str(text))).strip().rstrip(".")
        name = re.sub(r"\s+", " ", name)
        return name[:max_len].strip()
    except Exception:
        return "untitled"


def _last_name(full):
    """Best-effort surname: last whitespace token, particles kept simple."""
    parts = str(full or "").strip().split()
    return parts[-1] if parts else ""


def normalize_author(name):
    """Fold ERIC's inverted 'Last, First M.' form into 'First M. Last' and
    strip any remaining commas/brackets, so an author entry can never split
    the YAML flow sequence into bogus authors or hand _last_name an initial
    ('C.') as the surname for the citekey and filename."""
    s = re.sub(r"\s+", " ", str(name or "").strip())
    if s.count(",") == 1:
        last, first = [p.strip() for p in s.split(",")]
        if last and first:
            s = first + " " + last
    s = s.replace(",", " ").replace("[", " ").replace("]", " ")
    return re.sub(r"\s+", " ", s).strip()


def make_citekey(cand):
    """lowercase snake citekey: firstauthor_year_firstwords, e.g.
    hall_2017_what_you. Mirrors the vault's citekey style."""
    try:
        author = _last_name(
            normalize_author((cand.get("authors") or [""])[0])).lower()
        author = re.sub(r"[^a-z0-9]+", "", author) or "anon"
        year = str(cand.get("year") or "nd")
        words = re.findall(r"[a-z0-9]+", str(cand.get("title") or "").lower())
        tail = "_".join(words[:2]) if words else "untitled"
        return "%s_%s_%s" % (author, year, tail)
    except Exception:
        return "unknown_source"


def note_filename(cand):
    """'<FirstAuthorLast> <Year> - <Title>.md', the Sources/ naming style."""
    first = _last_name(normalize_author((cand.get("authors") or [""])[0])) or "Unknown"
    year = str(cand.get("year") or "n.d.")
    title = safe_name(cand.get("title") or "Untitled", max_len=70)
    return safe_name("%s %s - %s" % (first, year, title)) + ".md"


def _author_links(cand):
    names = " ".join(cand.get("authors") or []).lower()
    return [full for key, full in FOCUS_AUTHOR_LINKS.items() if key in names]


def _tags_for(cand):
    tags = ["source", "paper"]
    if cand.get("access") == "grey-lit":
        tags = ["source", "grey-lit"]
    tags.append("research-agent")
    src = str(cand.get("discovery_source") or "")
    if src.startswith("openalex-gallaudet") or "clerc" in src:
        tags.append("gallaudet")
    return tags


def render_note(cand, fulltext="", fulltext_via="", status="candidate"):
    """The full .md text for one candidate. Frontmatter is provenance (and
    matches the hand-authored Sources/ schema); the body restates every
    load-bearing field so BM25 can retrieve the note."""
    title = _no_dashes(str(cand.get("title") or "Untitled").strip())
    authors = [normalize_author(a) for a in (cand.get("authors") or [])
               if normalize_author(a)]
    year = cand.get("year") or "unknown"
    venue = _no_dashes(str(cand.get("venue") or "").strip())
    doi = str(cand.get("doi") or "").strip()
    url = str(cand.get("landing_url") or cand.get("fulltext_url") or "").strip()
    citekey = make_citekey(cand)
    display_name = "%s %s, %s" % (
        " and ".join(_last_name(a) for a in authors[:3]) or "Unknown",
        year, title)

    fm_authors = "[" + ", ".join(
        _sanitize_frontmatter_value(_no_dashes(a)) for a in authors) + "]"
    score = cand.get("score")
    fm = [
        "---",
        "name: %s" % _yaml_quote(_sanitize_frontmatter_value(display_name)),
        "type: source",
        "tags: [%s]" % ", ".join(_tags_for(cand)),
        "status: %s" % status,
        "authors: %s" % fm_authors,
        "year: %s" % year,
        "venue: %s" % _yaml_quote(_sanitize_frontmatter_value(venue)),
        "doi: %s" % _yaml_quote(doi),
        "url: %s" % _yaml_quote(_sanitize_frontmatter_value(url)),
        "citekey: %s" % citekey,
        "access: %s" % str(cand.get("access") or "abstract-only"),
        "discovery_source: %s" % _yaml_quote(
            _sanitize_frontmatter_value(str(cand.get("discovery_source") or ""))),
        "discovered: %s" % _yaml_quote(
            _sanitize_frontmatter_value(str(cand.get("discovered_at") or ""))),
    ]
    if cand.get("openalex_id"):
        fm.append("openalex_id: %s" % cand["openalex_id"])
    if cand.get("eric_id"):
        fm.append("eric_id: %s" % cand["eric_id"])
    if score is not None:
        fm.append("relevance_score: %s" % score)
    fm.append("---")

    peer = "peer-reviewed" if cand.get("peer_reviewed") else (
        "grey literature (not peer-reviewed)"
        if cand.get("access") == "grey-lit" else "peer-review status unverified")

    body = [
        "",
        "# " + title,
        "",
        "**Authors:** " + (", ".join(_no_dashes(a) for a in authors) or "unknown") + "  ",
        "**Year:** %s  " % year,
        "**Venue:** %s  " % (venue or "unknown"),
    ]
    if doi:
        body.append("**DOI:** https://doi.org/%s  " % doi)
    if url:
        body.append("**Link:** %s  " % url)
    body.append("**Status:** %s, found by the research agent (%s)  "
                % (peer, _no_dashes(str(cand.get("discovery_source") or ""))))
    note = str(cand.get("relevance_note") or "").strip()
    if note:
        body.extend(["", "> [!abstract] Why this matters",
                     "> " + _no_dashes(re.sub(r"\s+", " ", note))])
    abstract = str(cand.get("abstract") or "").strip()
    if abstract:
        body.extend(["", "## Abstract", "",
                     _clean_body_text(abstract)[:ABSTRACT_CAP]])
    if fulltext:
        via = (" (via %s)" % fulltext_via) if fulltext_via else ""
        body.extend(["", "## Full text%s" % via, "",
                     _clean_body_text(fulltext)])
    elif not abstract:
        body.extend(["", "Only the citation metadata is openly available "
                     "for this work; follow the link above for access."])
    links = ["[[%s]]" % MOC_LINK] + ["[[%s]]" % a for a in _author_links(cand)]
    body.extend(["", "## Related", "", " ".join(links), ""])

    return "\n".join(fm) + "\n" + "\n".join(body)


def write_note(cand, dest_dir, fulltext="", fulltext_via="",
               status="candidate"):
    """Render and atomically write the note. Returns the path or '' on
    failure. Never raises."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, note_filename(cand))
        text = render_note(cand, fulltext=fulltext, fulltext_via=fulltext_via,
                           status=status)
        return path if _atomic_write_text(path, text) else ""
    except Exception:
        return ""
