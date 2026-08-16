#!/usr/bin/env python3
"""Pure helpers for the two-pass DEPTH answer: facet selection, merge, citations.

Deliberately dependency-free. This module must never import brain.py, never
touch the network, and never call a model: brain.py cannot be imported on the
Mac (no index, no Ollama), so these functions are the only part of the depth
path that is testable off-device -- and the same functions double as the
offline scorer's definition of "deeper" (see facet_coverage and score_depth.py).
Everything the orchestrator needs from the index arrives as a plain argument;
there is no DI object here on purpose (brain_skills.SkillContext already exists,
and one orchestrator does not justify a second DI framework).

Facets come from the vault's OWN [[wikilink]] graph -- meta["links_out"], the
map expand_with_links already consumes -- not from a hardcoded topic table. The
hub note Core/What Is Language Deprivation.md alone links out to 18 distinct
notes, so the facet table is already in the index and stays correct as the vault
grows.

CITATIONS ARE BUILT FROM FRONTMATTER ONLY. Never from chunk body text: the
Gulati 2018 note has no frontmatter doi: yet its reference list carries 18 other
papers' DOIs, so a body-text regex confidently attributes Hall/Levin/Anderson
2017's 10.1007/s00127-017-1351-7 to Gulati. The frontmatter url: field is also
deliberately dropped -- brain.py's faithfulness_flags treats any URL in the
answer that is absent from the retrieved context as a fabrication flag, so
emitting our own verified links would make correct citations look unfaithful.
"""

import os
import re

# Footer marker. Kept here so the scorer, the judge and any ablation can strip
# the citation block from an answer with one shared constant, which is what
# makes the citation flag independently measurable from the depth flag.
CITE_SENTINEL = "\n\n[retrieved sources]\n"

# Frontmatter type that marks the seven LanguageDeprivationAgent/Core notes:
# the best LD prose in the vault, and completely uncitable (no authors, no
# year, no doi). They dominate facet retrieval, so they degrade to a labelled
# synthesis note rather than to an empty or half-built citation line.
SYNTHESIS_LABEL = "synthesis note"

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""
a an and are as at be by for from how in into is it its of on or over the their
this to was what when which who why with within without vs
""".split())


def _words(text):
    """Lowercased content words, stopwords and 1-char tokens removed."""
    return [w for w in _WORD_RE.findall((text or "").lower())
            if len(w) > 1 and w not in _STOP]


def _stem_of(source):
    """Filename stem of a vault-relative source path, minus the .md."""
    return os.path.splitext(os.path.basename(str(source or "")))[0]


def facet_label(source):
    """Human-readable facet name for a source path. For a Sources note titled
    'Gulati 2018 - Language Deprivation Syndrome' the topical half after the
    dash is what an answer would actually discuss, so that half is the label."""
    stem = _stem_of(source)
    if " - " in stem:
        return stem.split(" - ", 1)[1].strip() or stem
    return stem


def strip_citations(text):
    """Answer body with the citation footer removed. Coverage and every scored
    body metric run on this: names listed in a footer are not discussion, and
    scoring them would let the footer inflate the depth number for free."""
    if not text:
        return ""
    i = text.find(CITE_SENTINEL.strip())
    return text[:i].rstrip() if i >= 0 else text


def graph_facets(links_out, top_source, context_sources):
    """Candidate facets for the second pass, taken from the wikilink graph.

    links_out is meta["links_out"] ({source: [linked source, ...]}). top_source
    is the best-ranked note of pass 1 (its neighbourhood is the hub), and
    context_sources is every note the draft already saw. Returns a list of
    facets -- {"source", "label", "terms"} -- for notes the draft did NOT
    already cover, hub neighbours first, then neighbours of the remaining
    context notes in order. Never raises; an empty/odd graph yields []."""
    facets, seen = [], set()
    try:
        links_out = links_out or {}
        covered = {s for s in (context_sources or []) if s}
        if top_source:
            covered.add(top_source)
        seeds = [top_source] if top_source else []
        seeds += [s for s in (context_sources or []) if s and s != top_source]
        for seed in seeds:
            for target in links_out.get(seed) or []:
                if not target or target in covered or target in seen:
                    continue
                seen.add(target)
                facets.append({"source": target,
                               "label": facet_label(target),
                               "terms": _words(facet_label(target))})
    except Exception:
        return facets
    return facets


def rank_facets(candidates, draft, question, limit=3):
    """The `limit` facets the draft says LEAST about.

    Scored on how many of a facet's label terms already appear in the draft
    (fewer is better -- that is the gap the second pass is for), tie-broken by
    overlap with the question (a facet the question gestures at wins) and then
    by graph order, which keeps the choice deterministic across runs."""
    if not candidates or limit <= 0:
        return []
    dwords = set(_words(strip_citations(draft)))
    qwords = set(_words(question))
    scored = []
    for i, f in enumerate(candidates):
        terms = f.get("terms") if isinstance(f, dict) else _words(str(f))
        terms = terms or []
        n = float(len(terms)) or 1.0
        present = sum(1 for t in terms if t in dwords) / n
        asked = sum(1 for t in terms if t in qwords) / n
        scored.append((present, -asked, i, f))
    scored.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in scored[:limit]]


def new_chunks(candidates, seen_ids, cap):
    """Chunks from `candidates` whose id the first pass has not already seen,
    capped at `cap`. Same dedup contract as expand_with_links: id-based, order
    preserving, and a chunk with no id is dropped rather than double-counted."""
    out = []
    if cap is None or cap <= 0:
        return out
    seen = set(seen_ids or [])
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
        if len(out) >= cap:
            break
    return out


def merge(draft, expansion):
    """Join the pass-1 draft and the pass-2 expansion into one answer.

    Either side may be empty (an expansion that found nothing must not blank
    the draft), and a small model asked to expand sometimes restates a
    paragraph it already wrote, so exactly repeated paragraphs are dropped."""
    draft = (draft or "").strip()
    expansion = (expansion or "").strip()
    if not expansion:
        return draft
    if not draft:
        return expansion
    seen = {p.strip() for p in draft.split("\n\n") if p.strip()}
    kept = [p.strip() for p in expansion.split("\n\n")
            if p.strip() and p.strip() not in seen]
    if not kept:
        return draft
    return draft + "\n\n" + "\n\n".join(kept)


# Same shape as brain.py's _CITE_URL_RE; duplicated rather than imported so this
# module keeps its no-brain-import guarantee.
_CITE_URL_RE = re.compile(r"https?://\S+")


def _citation_line(source, fields):
    """One citation from frontmatter fields only. Returns None for a source
    with nothing at all to say."""
    fields = fields or {}
    name = fields.get("name") or _stem_of(source)
    authors = fields.get("authors") or []
    year = fields.get("year")
    venue = fields.get("venue")
    doi = fields.get("doi")
    if fields.get("type") == "core-topic" or not (authors or year or doi):
        # Best prose in the vault, nothing citable in it. Say so plainly.
        return "- %s (%s, %s)" % (name, SYNTHESIS_LABEL, source)
    parts = []
    if authors:
        parts.append(", ".join(str(a) for a in authors))
    if year:
        parts.append("(%s)" % year)
    parts.append(name)
    if venue:
        parts.append(str(venue))
    if doi:
        # doi: prefix only, never a resolver URL -- see the module docstring.
        parts.append("doi:%s" % doi)
    line = "- " + ". ".join(p for p in parts if p)
    return line


def citation_footer(sources, read_frontmatter):
    """Citation block for `sources`, or "" when there is nothing to cite.

    read_frontmatter is injected (source path -> frontmatter fields dict, in
    production build_index.split_frontmatter + parse_frontmatter_fields) so this
    module stays pure and the trap is structural: body text never reaches this
    function, so a body DOI can never be attributed to the wrong paper. Any URL
    that somehow arrives in a frontmatter field is stripped, because
    brain.py:faithfulness_flags would flag it as a fabricated citation."""
    lines, seen = [], set()
    for src in sources or []:
        if not src or src in seen:
            continue
        seen.add(src)
        try:
            fields = read_frontmatter(src)
        except Exception:
            fields = None
        line = _citation_line(src, fields)
        if line:
            lines.append(_CITE_URL_RE.sub("", line).rstrip())
    if not lines:
        return ""
    return CITE_SENTINEL + "\n".join(lines)


# A facet counts as discussed when this fraction of its label terms appear in
# the answer body. Below 1.0 because an answer that engages a facet paraphrases
# its title rather than quoting it; above a half so a single shared word ("deaf")
# cannot claim coverage of a note the answer never touched.
FACET_HIT_RATIO = 0.6


def facet_coverage(text, facets):
    """Fraction of `facets` the answer body actually discusses, in [0.0, 1.0].

    This is the operational definition of "deeper" and it is a pure function of
    the answer string, so it runs offline over any --answers-out file with no
    model, no judge and no index. The citation footer is stripped first: a
    source listed in the footer is a reference, not discussion."""
    facets = [f for f in (facets or [])]
    if not facets:
        return 0.0
    body = set(_words(strip_citations(text)))
    hits = 0
    for f in facets:
        if isinstance(f, dict):
            terms = f.get("terms") or _words(f.get("label") or f.get("source") or "")
        else:
            terms = _words(facet_label(f)) if str(f).endswith(".md") else _words(str(f))
        if not terms:
            continue
        ratio = sum(1 for t in terms if t in body) / float(len(terms))
        if ratio >= FACET_HIT_RATIO:
            hits += 1
    return hits / float(len(facets))
