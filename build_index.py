#!/usr/bin/env python3
"""Build the BM25 index pickle for the brain RAG agent.

Walks the vault of markdown files, cleans and chunks each document, tokenizes
the chunks, and writes a single pickle in the exact INDEX FORMAT shared with
brain.py. Safe to re-run (idempotent): it overwrites the output pickle.

INDEX FORMAT v2 is a superset of v1. All v1 keys are unchanged; v2 adds four
new maps that let brain.py expand retrieval along Obsidian wikilinks and surface
external reference URLs:
  by_source: {source_rel_path: [chunk_id, ...]}    chunk ids per note, doc order
  link_map:  {normalized_key: source_rel_path}     resolve [[wikilinks]] to a note
  links_out: {source: [target_source, ...]}        resolved outgoing wikilinks
  urls_out:  {source: [external_url, ...]}          external http(s) links per note
"""

import argparse
import array
import os
import pickle
import re
import sys
import unicodedata

# Bumped whenever tokenize() changes. brain.py refuses a pickle whose
# tokenizer_version does not match, so index tokens and query tokens can never
# silently diverge (a stale pickle + updated tokenizer breaks retrieval).
TOKENIZER_VERSION = "tok-nfkd-acr-sstem"

DEFAULT_VAULT = "/mnt/nvme/brain/vault"
DEFAULT_OUT = "/mnt/nvme/brain/index/bm25.pkl"

# Chunking parameters (in words). v3 uses structure-aware chunking: split on
# markdown headings, merge tiny sections, split oversized ones on sentence
# boundaries, and prepend a heading breadcrumb to every chunk so its heading
# path enters the BM25 index (term injection -- the biggest reliable win for a
# lexical retriever over structured markdown).
CHUNK_WORDS = 220        # TARGET: flush an accumulating chunk at ~this size
FLOOR_WORDS = 90         # sections smaller than this get merged with neighbors
CEIL_WORDS = 320         # sections larger than this get split (sentence-safe)

# v5 definition card. Every note also emits ONE short synthetic lead chunk
# holding its identity facts: title, frontmatter scalars, aliases, tags, and the
# note's single most answer-dense line (an [!abstract]-style callout body, else
# the first sentence after the H1).
#
# This exists because BM25 length-normalizes (rank_bm25 b=0.75 divides by
# |d|/avgdl), so a short answer-dense doc outranks the same terms diluted across
# a long one. FLOOR_WORDS=90 merges a 25-word one-line summary into a 250-word
# prose blob, which is exactly the chunk a short factual question needs and
# exactly the chunk that merging destroys. The card restores it WITHOUT changing
# the prose chunks: "Where is Gallaudet University?" now has a ~40-word doc
# reading "Gallaudet University. Type: org. Founded: 1864. ... Federally
# chartered Deaf university in Washington DC ..." to score against.
CARD_MAX_WORDS = 70      # hard cap; a card longer than this stops being a card

# Cap on external URLs captured per note.
MAX_URLS_PER_NOTE = 20

# Optional dense-retrieval sidecar (see brain.py hybrid RRF). Built ONLY when
# --embed is passed, so a plain build stays pure-stdlib and Ollama-independent.
# Each chunk is embedded once with a local Ollama model and L2-normalized so the
# query-time cosine is a plain dot product. Vectors live in the SAME pickle as
# one flat float32 buffer (~4 bytes/float, ~12 MB for this vault) that loads with
# or without numpy. The document embedding uses the chunk text verbatim (heading
# breadcrumb included) with NO task prefix, to match brain.py's query embedding
# and the existing rerank/router embed path exactly.
#
# nomic-embed-text is, however, trained ASYMMETRICALLY: documents are meant to
# carry "search_document: " and queries "search_query: ", and dropping both
# sides is a standing accuracy loss on every semantic query rather than a
# neutral simplification. --embed-prefix builds an index that uses them.
#
# The prefix scheme is RECORDED IN THE INDEX as prefix_version, and brain.py
# reads it back to choose the query prefix. That direction matters: the one
# unrecoverable failure here is an index embedded one way and queried the
# other, which does not crash -- it just quietly retrieves worse, exactly the
# kind of defect this repo has been burned by before. Making the artifact
# describe itself means the two sides cannot drift.
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_DOC_CHARS = 2000    # truncate a very long chunk before embedding
EMBED_TIMEOUT = 60        # generous: a cold model load on the Jetson is slow

# prefix_version -> (document prefix, query prefix). "none" reproduces every
# index built before this option existed.
EMBED_PREFIXES = {
    "none": ("", ""),
    "nomic-v1": ("search_document: ", "search_query: "),
}
DEFAULT_EMBED_PREFIX = "none"


def _load_embed_fns():
    # Prefer semantic_cache._embed/_normalize so index-time and query-time
    # embeddings share ONE implementation (no silent preprocessing drift). Fall
    # back to inline stdlib copies if semantic_cache is not importable, so
    # build_index.py can still embed when run standalone.
    try:
        from semantic_cache import _embed, _normalize
        return _embed, _normalize
    except Exception:
        import json as _json
        import math as _math
        import urllib.request as _url

        def _embed(text, host, model, timeout):
            try:
                data = _json.dumps({"model": model, "prompt": text}).encode("utf-8")
                req = _url.Request(
                    host + "/api/embeddings", data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                with _url.urlopen(req, timeout=timeout) as resp:
                    vec = _json.loads(resp.read().decode("utf-8")).get("embedding")
                return vec if isinstance(vec, list) and vec else None
            except Exception:
                return None

        def _normalize(vec):
            n = _math.sqrt(sum(x * x for x in vec)) or 1.0
            return [x / n for x in vec]

        return _embed, _normalize


def build_dense(chunks, model, host, prefix_version=DEFAULT_EMBED_PREFIX):
    # Embed every chunk once and return a dense block, or None if embedding is
    # not usable (Ollama down / model not pulled / mostly failing). Only
    # successfully-embedded chunks get a row; a chunk that fails to embed is
    # simply dense-invisible (BM25 still covers it), which keeps alignment
    # trivial and never blocks the build.
    embed, normalize = _load_embed_fns()
    doc_prefix = EMBED_PREFIXES.get(prefix_version, EMBED_PREFIXES[DEFAULT_EMBED_PREFIX])[0]
    total = len(chunks)
    print("Embedding %d chunks with %s (this runs on the Jetson GPU) ..." % (total, model))
    if doc_prefix:
        print("  document prefix: %r (prefix_version=%s)" % (doc_prefix, prefix_version))
    ids = []
    flat = array.array("f")
    dim = None
    failures = 0
    for n, c in enumerate(chunks):
        # Truncate FIRST, then prefix, so the prefix is never the thing that
        # gets cut and the document budget stays EMBED_DOC_CHARS either way.
        text = doc_prefix + (c.get("text") or "")[:EMBED_DOC_CHARS]
        vec = embed(text, host, model, EMBED_TIMEOUT)
        if vec and dim is None:
            dim = len(vec)
        if not vec or len(vec) != dim:
            failures += 1
        else:
            flat.extend(normalize(vec))
            ids.append(c["id"])
        if (n + 1) % 200 == 0:
            print("  embedded %d/%d (%d failed)" % (n + 1, total, failures))

    if dim is None or not ids:
        print("WARNING: dense embedding produced no vectors. Is Ollama running and "
              "is '%s' pulled? Skipping the dense block; the index stays BM25-only." % model)
        return None
    if failures > total * 0.5:
        print("WARNING: over half of chunk embeddings failed (%d/%d). Skipping the "
              "dense block so a half-built matrix is never shipped." % (failures, total))
        return None
    if failures:
        print("NOTE: %d/%d chunks could not be embedded and are dense-invisible "
              "(BM25 still retrieves them)." % (failures, total))
    return {"model": model, "dim": dim, "ids": ids, "buf": flat,
            "prefix_version": prefix_version}


def _s_stem(w):
    # Conservative S-stemmer: fold only regular English plurals so a query and
    # the corpus agree on "implant"/"implants", "study"/"studies". MUST be
    # byte-identical in build_index.py and brain.py. Guards keep it from
    # mangling acronyms (len<4), numbers (not isalpha), and -us/-ss/-is/-is
    # words. Irregular plurals (child/children) are intentionally NOT folded.
    if len(w) < 4 or not w.isalpha():
        return w
    if w.endswith("ies") and not w.endswith(("eies", "aies")):
        return w[:-3] + "y"
    if w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        return w[:-1]
    if w.endswith("s") and not w.endswith(("us", "ss", "is")):
        return w[:-1]
    return w


def tokenize(s):
    # Shared tokenizer. MUST be byte-identical in build_index.py and brain.py;
    # any change here requires rebuilding the index (bump TOKENIZER_VERSION).
    # Order is load-bearing: NFKD accent-fold (also lowercases via .lower()),
    # collapse dotted acronyms (A.S.L. -> asl, U.S. -> us), split, then S-stem
    # each token to fold regular plurals.
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), s)
    return [_s_stem(t) for t in re.findall(r"[a-z0-9]+", s)]


def split_frontmatter(text):
    # Split a leading YAML frontmatter block from the body in a single scan.
    # Returns (frontmatter, body): the raw lines between the leading --- and its
    # closing --- (or "" if absent/malformed), and the text after the closing ---
    # (or the whole text unchanged when there is no well-formed frontmatter).
    if not text.startswith("---"):
        return "", text
    lines = text.split("\n")
    # First line is the opening ---. Find the closing --- after it.
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    # No closing delimiter found; treat as no frontmatter, leave text as-is.
    return "", text


def clean_obsidian(text):
    # Drop image embeds like ![[image.png]] entirely.
    text = re.sub(r"!\[\[[^\]]*\]\]", "", text)
    # Turn [[Link|alias]] into "alias".
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    # Turn [[Link]] into "Link".
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # Drop the callout TYPE marker but keep the callout TEXT: "> [!abstract]
    # One-line" -> "One-line". Without this the type name tokenizes as a real
    # term ("abstract", "note", "info") and lands in the index as noise on every
    # note that uses callouts, while the prose it labels is what we want.
    text = re.sub(r"\[!\w+\][-+]?", "", text)
    # Strip leading blockquote markers so callout/quote bodies read as prose.
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    return text


def strip_heading_hashes(line):
    # Strip leading markdown heading hashes but keep the heading text.
    m = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
    if m:
        return m.group(1).strip()
    return line


def extract_title(raw_text, fallback):
    # First markdown heading in the cleaned text, else the fallback filename.
    for line in raw_text.split("\n"):
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", line)
        if m:
            title = m.group(1).strip()
            if title:
                return title
    return fallback


def normalize_body(text):
    # Clean Obsidian syntax, then strip heading hashes line by line, keeping
    # heading text. Returns a plain-text body suitable for chunking.
    text = clean_obsidian(text)
    out_lines = []
    for line in text.split("\n"):
        out_lines.append(strip_heading_hashes(line))
    return "\n".join(out_lines)


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sections(body_no_fm):
    # Split a note's post-frontmatter text into heading-delimited sections,
    # tracking the heading breadcrumb (the H1>H2>H3 path) for each. A section's
    # text NEVER crosses a heading. Returns a list of {"crumb": [titles],
    # "text": str}. The Obsidian/heading syntax is cleaned per section by the
    # caller so headings stay visible here for structure detection.
    sections = []
    stack = []            # current heading titles by depth
    cur_lines = []
    cur_crumb = []

    def flush():
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append({"crumb": list(cur_crumb), "text": text})

    for line in body_no_fm.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            # A heading starts a new section. Flush whatever body came before.
            flush()
            cur_lines[:] = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # Trim the stack to the parent depth, then push this heading.
            stack[:] = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            cur_crumb = [t for t in stack if t]
        else:
            cur_lines.append(line)
    flush()
    return sections


def _sentences(text):
    # Split on newlines first, then sentence-split within each line, so every
    # markdown table row / bullet item is its own packing unit for
    # split_oversized(). Without this, a big periodless table is one monster
    # "sentence" that the last-resort word-slice cuts mid-row, orphaning an
    # acronym from its expansion. Soft-wrapped prose (one paragraph per line, as
    # in this Obsidian vault) keeps byte-identical boundaries.
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENTENCE_SPLIT_RE.split(line):
            if s.strip():
                units.append(s.strip())
    return units


def split_oversized(text):
    # Split a too-large section into pieces <= CEIL_WORDS on sentence
    # boundaries, carrying a 1-sentence overlap between adjacent pieces so an
    # argument that spans the cut is not severed. A single monster sentence
    # (e.g. a long list) is hard-split by words as a last resort.
    sents = _sentences(text)
    if not sents:
        return []
    pieces = []
    cur = []
    cur_w = 0
    for s in sents:
        w = len(s.split())
        if w > CEIL_WORDS:
            if cur:
                pieces.append(" ".join(cur))
                cur = []
                cur_w = 0
            words = s.split()
            for i in range(0, len(words), CEIL_WORDS):
                pieces.append(" ".join(words[i:i + CEIL_WORDS]))
            continue
        if cur and cur_w + w > CEIL_WORDS:
            pieces.append(" ".join(cur))
            # 1-sentence overlap: seed the next piece with the last sentence.
            cur = [cur[-1]]
            cur_w = len(cur[-1].split())
        cur.append(s)
        cur_w += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _crumb_prefix(stem, note_title, crumb):
    # Breadcrumb prepended to every chunk before tokenizing. This injects the
    # heading path into the BM25 index so pronoun-heavy bodies ("this delay",
    # "it") still match on their topic terms. The filename stem leads the trail
    # so filename proper nouns (author surname, year, citation key) enter the
    # index even when the note's H1 is generic ("# Introduction").
    head = [p for p in (stem, note_title) if p]
    parts = head + [c for c in crumb if c and c != note_title and c != stem]
    # De-duplicate consecutive repeats while preserving order. When stem ==
    # note_title (the common no-heading case) this collapses the duplicate.
    trail = []
    for p in parts:
        if not trail or trail[-1] != p:
            trail.append(p)
    return "[" + " > ".join(trail) + "]"


def chunk_note(body_no_fm, note_title, stem):
    # Structure-aware chunking for one note. Returns a list of chunk texts, each
    # already carrying its heading breadcrumb. Sections are cleaned of Obsidian
    # syntax here (headings were used for structure and are dropped from bodies).
    sections = split_into_sections(body_no_fm)
    if not sections:
        return []

    # Clean each section body and compute word counts.
    prepared = []
    for sec in sections:
        text = normalize_body(sec["text"]).strip()
        if not text:
            continue
        prepared.append({"crumb": sec["crumb"], "text": text,
                         "words": len(text.split())})
    if not prepared:
        return []

    chunks = []
    buf_text = []
    buf_words = 0
    buf_crumb = None

    def flush_buffer():
        nonlocal buf_text, buf_words, buf_crumb
        if buf_text:
            body = " ".join(buf_text).strip()
            if body:
                prefix = _crumb_prefix(stem, note_title, buf_crumb or [])
                chunks.append(prefix + "\n" + body)
        buf_text = []
        buf_words = 0
        buf_crumb = None

    for sec in prepared:
        words = sec["words"]
        if words >= FLOOR_WORDS:
            # Substantial section. Flush any pending small buffer first, folding
            # it into this section when they fit together under the ceiling.
            if buf_words and buf_words + words <= CEIL_WORDS:
                sec_text = " ".join(buf_text) + " " + sec["text"]
                sec_crumb = buf_crumb or sec["crumb"]
                buf_text = []
                buf_words = 0
                buf_crumb = None
            else:
                flush_buffer()
                sec_text = sec["text"]
                sec_crumb = sec["crumb"]

            if len(sec_text.split()) <= CEIL_WORDS:
                prefix = _crumb_prefix(stem, note_title, sec_crumb)
                chunks.append(prefix + "\n" + sec_text.strip())
            else:
                for piece in split_oversized(sec_text):
                    prefix = _crumb_prefix(stem, note_title, sec_crumb)
                    chunks.append(prefix + "\n" + piece.strip())
        else:
            # Tiny section: accumulate with neighbors so 1-line stubs do not
            # pollute BM25 IDF statistics as standalone chunks. Inline the
            # section's own leaf heading so its terms survive the merge (the
            # buffer's outer breadcrumb only carries the first section's path).
            if buf_crumb is None:
                buf_crumb = sec["crumb"]
            leaf = sec["crumb"][-1] if sec["crumb"] else ""
            if leaf and leaf != note_title and (not buf_crumb or leaf != buf_crumb[-1]):
                buf_text.append(leaf + ": " + sec["text"])
            else:
                buf_text.append(sec["text"])
            buf_words += words
            if buf_words >= CHUNK_WORDS:
                flush_buffer()

    # A sub-FLOOR buffer trailing ALL substantial sections (a short closing
    # stub) would otherwise flush as a standalone micro-chunk that skews BM25
    # length normalization. Fold it back into the last same-note chunk when they
    # fit under the ceiling; otherwise emit it normally.
    if buf_text:
        body = " ".join(buf_text).strip()
        if body and buf_words < FLOOR_WORDS and chunks and \
                len(chunks[-1].split()) + buf_words <= CEIL_WORDS:
            chunks[-1] = chunks[-1].rstrip() + " " + body
        else:
            flush_buffer()
    return chunks


def normalize_key(key):
    # Lowercase, strip, and collapse internal whitespace to single spaces.
    return re.sub(r"\s+", " ", key.strip().lower())


def _strip_yaml_scalar(value):
    # Trim whitespace and surrounding single or double quotes from a scalar.
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()


# Frontmatter keys surfaced into the index. List keys accept an inline
# [a, b] form, a single scalar, or a block "- item" list; scalar keys take the
# rest of the line. "alias" folds into "aliases" so both spellings work.
_FM_LIST_KEYS = {"aliases": "aliases", "alias": "aliases",
                 "tags": "tags", "authors": "authors"}
_FM_SCALAR_KEYS = ("name", "type", "year", "venue", "doi", "citekey", "url")

# Any OTHER top-level scalar is captured too, under fields["extra"], so vault
# facts held only in frontmatter reach the index. Gallaudet's "founded: 1864"
# was invisible to retrieval until now purely because it is not in the tuple
# above. Bookkeeping keys carry no answer and would only add per-note noise, so
# they are dropped. fm_meta_sentence deliberately IGNORES extras and keeps its
# original 7-field output, so existing chunks are byte-identical; only the new
# definition card reads them.
_FM_DENY_KEYS = frozenset({
    "status", "created", "updated", "modified", "date", "cssclass", "cssclasses",
    "publish", "permalink", "position", "id", "uid", "rating", "priority",
})


def _parse_yaml_list(lines, i, rest):
    # Parse one list-valued key starting at line i whose post-colon text is
    # `rest`. Returns the item list. Block items are the "- item" lines that
    # follow; any non-indented, non-list line ends the block.
    items = []
    if rest:
        if rest.startswith("[") and rest.endswith("]"):
            for part in rest[1:-1].split(","):
                val = _strip_yaml_scalar(part)
                if val:
                    items.append(val)
        else:
            val = _strip_yaml_scalar(rest)
            if val:
                items.append(val)
        return items
    for j in range(i + 1, len(lines)):
        item = lines[j]
        if not item.strip():
            continue
        lm = re.match(r"^\s*-\s+(.*)$", item)
        if lm:
            val = _strip_yaml_scalar(lm.group(1))
            if val:
                items.append(val)
        elif not (item.startswith(" ") or item.startswith("\t")):
            break
    return items


def parse_frontmatter_fields(frontmatter_text):
    # Best-effort parse of the flat YAML used across this vault. Returns a dict
    # holding any of: aliases/tags/authors (lists) and name/type/year/venue/
    # doi/citekey/url (strings), plus every other top-level scalar under the
    # "extra" key (see _FM_DENY_KEYS). Missing or empty keys are absent. Only
    # top-level (non-indented) keys are read, so nested blocks never leak in.
    # First writer wins per key. Never raises.
    fields = {}
    extra = {}
    if not frontmatter_text:
        return fields
    lines = frontmatter_text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1).lower()
        rest = m.group(2).strip()
        if key in _FM_LIST_KEYS:
            canon = _FM_LIST_KEYS[key]
            if canon not in fields:
                items = _parse_yaml_list(lines, i, rest)
                if items:
                    fields[canon] = items
        elif key in _FM_SCALAR_KEYS:
            if key not in fields:
                val = _strip_yaml_scalar(rest)
                if val:
                    fields[key] = val
        elif rest and key not in _FM_DENY_KEYS and key not in extra:
            # Unknown scalar. Skip anything that opens a block (an empty rest
            # means the value is on following indented lines, which this flat
            # parser does not read) and anything list-shaped.
            val = _strip_yaml_scalar(rest)
            if val and not val.startswith(("[", "{", "|", ">")):
                extra[key] = val
    if extra:
        fields["extra"] = extra
    return fields


def parse_aliases(frontmatter_text):
    # Back-compat wrapper: the raw "aliases:" list (see parse_frontmatter_fields).
    return parse_frontmatter_fields(frontmatter_text).get("aliases", [])


_CALLOUT_RE = re.compile(r"^\s{0,3}>\s*\[!(\w+)\]")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Callout types that hold a definition rather than an aside. A [!warning] or
# [!todo] body is not what a "what/where is X" question wants.
_DEFINITION_CALLOUTS = frozenset({
    "abstract", "summary", "tldr", "note", "info", "definition", "quote", "cite",
})


def _callout_body(lines, i):
    """Text of the callout whose marker is at line i. The marker line's tail is
    the callout's TITLE ("One-line"), not its content; the content is the
    blockquote lines that follow."""
    parts = []
    for j in range(i + 1, len(lines)):
        nxt = lines[j]
        if not nxt.lstrip().startswith(">") or _CALLOUT_RE.match(nxt):
            break
        txt = nxt.lstrip()[1:].strip()
        if txt:
            parts.append(txt)
    return clean_obsidian(" ".join(parts)).strip()


def extract_definition(body_no_fm, note_title):
    """The note's single most answer-dense line, for the definition card.

    Three tiers, in order: the body of the first DEFINITIONAL callout (this
    vault uses `> [!abstract] One-line` as its house style for exactly this),
    else the first substantive prose line after the H1, else any callout body at
    all. The last tier matters for notes whose only content is a callout: a
    [!warning] is not a definition and must lose to real prose, but it still
    beats emitting a card with no descriptive text. Returns "" when the note has
    none of the three. Never raises."""
    lines = body_no_fm.split("\n")

    for i, line in enumerate(lines):
        m = _CALLOUT_RE.match(line)
        if not m or m.group(1).lower() not in _DEFINITION_CALLOUTS:
            continue
        body = _callout_body(lines, i)
        if body:
            return body

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith((">", "---", "|", "```")):
            continue
        if _HEADING_RE.match(line):
            continue
        text = clean_obsidian(_LIST_MARKER_RE.sub("", stripped)).strip()
        # Skip a bare restatement of the title and checkbox/task lines.
        if not text or text.startswith("[ ]") or text.lower() == note_title.lower():
            continue
        sents = _SENTENCE_SPLIT_RE.split(text)
        return " ".join(sents[:2]).strip() if sents else text

    for i, line in enumerate(lines):
        if _CALLOUT_RE.match(line):
            body = _callout_body(lines, i)
            if body:
                return body
    return ""


def note_card(fm_fields, note_title, stem, body_no_fm):
    """One short synthetic chunk carrying a note's identity facts, or "" when
    the note has nothing card-worthy. See CARD_MAX_WORDS for why this earns its
    place. SHARED by build() and update_index via build_note_chunks()."""
    facts = []
    name = (fm_fields.get("name") or note_title or stem or "").strip()
    if name:
        facts.append(name.rstrip(".") + ".")

    aliases = [a for a in fm_fields.get("aliases", []) if a]
    if aliases:
        facts.append("Also known as: %s." % ", ".join(aliases))

    # Known scalars in their declared order, then any extras alphabetically, so
    # a card's field order is stable across rebuilds (a churn-free diff).
    for key in _FM_SCALAR_KEYS:
        if key in ("name", "url"):
            continue                       # name leads the card; url adds no terms
        if fm_fields.get(key):
            facts.append("%s: %s." % (key.capitalize(), fm_fields[key]))
    for key in sorted(fm_fields.get("extra", {})):
        facts.append("%s: %s." % (key.replace("_", " ").capitalize(),
                                  fm_fields["extra"][key]))

    if fm_fields.get("authors"):
        facts.append("Authors: %s." % ", ".join(fm_fields["authors"]))
    if fm_fields.get("tags"):
        facts.append("Tags: %s." % ", ".join(fm_fields["tags"]))

    definition = extract_definition(body_no_fm, note_title)
    if definition:
        facts.append(definition)

    body = " ".join(f for f in facts if f).strip()
    if not body:
        return ""
    # A card that needs truncating has stopped being a card, but a long tail is
    # the definition sentence, so trim words rather than drop the whole thing.
    words = body.split()
    if len(words) > CARD_MAX_WORDS:
        body = " ".join(words[:CARD_MAX_WORDS])
    # A card with no more terms than its own breadcrumb already carries is pure
    # duplication; require at least one fact beyond the bare title.
    if len(facts) < 2:
        return ""
    return _crumb_prefix(stem, note_title, []) + "\n" + body


def build_note_chunks(body_no_fm, frontmatter, title, fallback_title, meta):
    """Produce one note's full chunk list: definition card first, then the
    structure-aware prose chunks, then frontmatter meta injection.

    THE shared seam between build() and update_index._chunk_file. Both must call
    this and nothing else, or a full build and an incremental web-autolearn
    update produce different chunks for the same note and every unchanged note
    churns on the next update.

    Returns (chunk_texts, meta_injected, has_card). When has_card is True the
    card is chunk_texts[0]; callers flag that chunk so retrieval can prefer it
    for fact-seeking questions (see brain._apply_card_boost)."""
    fm_fields = parse_frontmatter_fields(frontmatter)
    chunk_texts = list(chunk_note(body_no_fm, title, fallback_title))

    # apply_meta with meta="inline" appends to chunk_texts[0], so the card must
    # be inserted AFTER that call or the meta sentence lands on the card (which
    # already carries those fields) instead of on the first prose chunk.
    chunk_texts, injected = apply_meta(
        chunk_texts, frontmatter, meta, fallback_title, title)

    card = note_card(fm_fields, title, fallback_title, body_no_fm)
    if card:
        chunk_texts.insert(0, card)
    return chunk_texts, injected, bool(card)


def apply_meta(chunk_texts, frontmatter, meta, fallback_title, title):
    """Inject the frontmatter metadata sentence into one note's chunk list
    according to the meta mode. SHARED by build() and update_index._chunk_file
    so a full build and an incremental update produce identical chunks for the
    same note (the index records its mode as index[\"meta_mode\"]). Mutates and
    returns (chunk_texts, injected)."""
    if meta == "none":
        return chunk_texts, False
    sentence = fm_meta_sentence(parse_frontmatter_fields(frontmatter))
    if not sentence:
        return chunk_texts, False
    if meta == "inline" and chunk_texts:
        chunk_texts[0] = chunk_texts[0].rstrip() + "\n" + sentence
    else:
        # 'chunk' mode, or a frontmatter-only note with no body chunks.
        prefix = _crumb_prefix(fallback_title, title, [])
        chunk_texts.insert(0, prefix + "\n" + sentence)
    return chunk_texts, True


def fm_meta_sentence(fields):
    # Compact metadata sentence injected into the note's first chunk (--meta
    # inline) or emitted as a synthetic lead chunk (--meta chunk) so the
    # Sources/ frontmatter finally reaches BM25 and the dense vectors: author
    # first names, venue, year, DOI, and citekey are otherwise invisible to
    # retrieval (frontmatter is stripped before chunking).
    parts = []
    if fields.get("name"):
        parts.append("Source: %s." % fields["name"])
    if fields.get("authors"):
        parts.append("Authors: %s." % ", ".join(fields["authors"]))
    if fields.get("year"):
        parts.append("Year: %s." % fields["year"])
    if fields.get("venue"):
        parts.append("Published in %s." % fields["venue"])
    if fields.get("doi"):
        parts.append("DOI: %s." % fields["doi"])
    if fields.get("citekey"):
        parts.append("Citekey: %s." % fields["citekey"])
    if fields.get("tags"):
        parts.append("Tags: %s." % ", ".join(fields["tags"]))
    return " ".join(parts)


def extract_wikilink_targets(text):
    # Find [[target]], [[target|alias]], and embeds ![[target]] in the raw
    # post-frontmatter text. Return a list of cleaned target strings with any
    # |alias, #heading, and ^block anchor removed. Not normalized here; the
    # caller normalizes and resolves via link_map.
    targets = []
    for m in re.finditer(r"!?\[\[([^\]]+)\]\]", text):
        inner = m.group(1)
        # Drop the |alias portion.
        if "|" in inner:
            inner = inner.split("|", 1)[0]
        # Drop any #heading anchor.
        if "#" in inner:
            inner = inner.split("#", 1)[0]
        # Drop any ^block anchor.
        if "^" in inner:
            inner = inner.split("^", 1)[0]
        inner = inner.strip()
        if inner:
            targets.append(inner)
    return targets


def extract_urls(text):
    # Capture markdown [text](http...) hrefs and bare https?://... occurrences
    # from the raw post-frontmatter text. Dedupe (order preserved), cap at
    # MAX_URLS_PER_NOTE.
    urls = []
    seen = set()

    def _add(url):
        # Trim trailing punctuation commonly adjacent to prose URLs.
        url = url.rstrip('.,;:!?)]}\'"')
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    # Markdown-style link hrefs first so they win ordering.
    for m in re.finditer(r"\]\((https?://[^)\s]+)\)", text):
        _add(m.group(1))
    # Bare URLs anywhere in the text.
    for m in re.finditer(r"https?://[^\s<>()\[\]\"']+", text):
        _add(m.group(0))

    return urls[:MAX_URLS_PER_NOTE]


# Directory names that hold build/dependency artifacts, never vault content.
# Any note whose path descends through one of these is skipped so their READMEs
# and license .md files do not pollute BM25 statistics. Pruned in-place during
# os.walk so the walker never descends into them (handles arbitrary nesting such
# as Publishing/docx/node_modules/... and LanguageDeprivationAgent/agent/.venv/...).
#
# "Templates" is Obsidian scaffolding rather than a build artifact, but it fails
# the same test: three of its four notes carry an unrendered "{{title}}" as their
# H1 over a stub body, so every chunk they contribute is placeholder syntax that
# can only displace a real note. deploy.sh also stops shipping the folder, but
# rsync's --delete deliberately spares excluded paths, so any Templates/ already
# on the device would otherwise be indexed forever. Pruning here is what actually
# keeps it out of the index.
_JUNK_DIR_NAMES = {"node_modules", "site-packages", ".venv", "Templates"}


def build(vault, out, embed=False, embed_model=DEFAULT_EMBED_MODEL,
          ollama_host=DEFAULT_OLLAMA_HOST, meta="inline",
          embed_prefix=DEFAULT_EMBED_PREFIX):
    if not os.path.isdir(vault):
        print("ERROR: vault directory not found: %s" % vault)
        return 1

    md_files = []
    pruned = 0
    for root, dirs, files in os.walk(vault):
        # Prune build/dependency subtrees in-place so os.walk does not descend.
        keep = [d for d in dirs if d not in _JUNK_DIR_NAMES]
        pruned += len(dirs) - len(keep)
        dirs[:] = keep
        for name in files:
            if name.lower().endswith(".md"):
                md_files.append(os.path.join(root, name))
    md_files.sort()
    if pruned:
        print("Skipped %d build/dependency subtree(s) (node_modules/site-packages/.venv)." % pruned)

    chunks = []
    next_id = 0
    meta_notes = 0

    by_source = {}
    link_map = {}
    urls_out = {}
    # Per-file records carried into pass 2 for link resolution.
    file_records = []

    # ------------------------------------------------------------------
    # Pass 1: read each file, chunk/tokenize, collect link targets and urls,
    # and populate link_map. First writer wins on any key collision.
    # ------------------------------------------------------------------
    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except OSError as exc:
            print("WARNING: could not read %s: %s" % (path, exc))
            continue

        rel_source = os.path.relpath(path, vault)
        fallback_title = os.path.splitext(os.path.basename(path))[0]

        frontmatter, body_no_fm = split_frontmatter(raw)
        title = extract_title(body_no_fm, fallback_title)
        fm_fields = parse_frontmatter_fields(frontmatter)

        # Parse links and urls from the raw post-frontmatter text, BEFORE
        # clean_obsidian strips the wikilink syntax away.
        link_targets = extract_wikilink_targets(body_no_fm)
        urls = extract_urls(body_no_fm)

        # A frontmatter url: is the note's canonical external link; merge it in
        # ahead of body urls so the phase-2 FETCH candidate list can see it.
        fm_url = fm_fields.get("url", "")
        if fm_url.startswith(("http://", "https://")):
            if fm_url in urls:
                urls.remove(fm_url)
            urls.insert(0, fm_url)
            urls = urls[:MAX_URLS_PER_NOTE]

        # Definition card + structure-aware prose chunks + frontmatter meta
        # injection, all via the one seam update_index.py also calls.
        chunk_texts, injected, has_card = build_note_chunks(
            body_no_fm, frontmatter, title, fallback_title, meta)
        if injected:
            meta_notes += 1

        by_source.setdefault(rel_source, [])
        for pos, chunk_text in enumerate(chunk_texts):
            toks = tokenize(chunk_text)
            if not toks:
                # Nothing tokenizable (punctuation-only); skip so that
                # re-tokenizing this chunk at load can never yield an empty
                # doc. The token list itself is NOT stored (see below).
                continue
            row = {
                "id": next_id,
                "text": chunk_text,
                "source": rel_source,
                "title": title,
            }
            # Flag set only on cards, so the pickle carries one extra key per
            # note rather than per chunk.
            if has_card and pos == 0:
                row["card"] = True
            chunks.append(row)
            by_source[rel_source].append(next_id)
            next_id += 1

        # Register link_map keys for this note. First writer wins. The
        # frontmatter name: joins the aliases so [[name-style]] references and
        # wikilink expansion resolve on it too.
        stem = os.path.splitext(os.path.basename(rel_source))[0]
        rel_no_ext = os.path.splitext(rel_source)[0]
        aliases = list(fm_fields.get("aliases", []))
        if fm_fields.get("name"):
            aliases.append(fm_fields["name"])
        for key in [stem, rel_no_ext] + aliases:
            nk = normalize_key(key)
            if nk and nk not in link_map:
                link_map[nk] = rel_source

        if urls:
            urls_out[rel_source] = urls

        file_records.append({
            "source": rel_source,
            "targets": link_targets,
        })

    # ------------------------------------------------------------------
    # Pass 2: resolve outgoing wikilinks using the completed link_map.
    # links_out excludes self-links and unresolved targets, deduped.
    # ------------------------------------------------------------------
    links_out = {}
    for rec in file_records:
        source = rec["source"]
        resolved = []
        seen = set()
        for target in rec["targets"]:
            dest = link_map.get(normalize_key(target))
            if dest is None:
                # Fall back to the basename of the target path.
                dest = link_map.get(normalize_key(os.path.basename(target)))
            if dest is None or dest == source or dest in seen:
                continue
            seen.add(dest)
            resolved.append(dest)
        if resolved:
            links_out[source] = resolved

    # Optional dense block for brain.py hybrid retrieval. Built last (after all
    # chunks exist) and only when --embed is passed. A None result (Ollama down
    # or mostly failing) leaves the index BM25-only and brain.py falls back to
    # its BM25+rerank path, so the build never hard-fails on the embedder.
    dense = build_dense(chunks, embed_model, ollama_host, embed_prefix) if embed else None

    # 'tokenized' is intentionally omitted: it is fully derivable from each
    # chunk's text via tokenize(), so storing it roughly doubled the pickle.
    # load_index() re-tokenizes at load. version 4 signals an optional dense
    # block may be present; brain.py reads it only when present, so a v3 (no
    # dense) and a v4-without-dense index behave identically.
    index = {
        # 5: definition-card chunks, callout cleaning, and generalized
        # frontmatter capture. Chunk TEXT changed, so a v4 pickle answers
        # differently than a v5 one; the tokenizer did not, so
        # TOKENIZER_VERSION is unchanged and old pickles still load.
        "version": 5,
        "tokenizer_version": TOKENIZER_VERSION,
        "meta_mode": meta,
        "chunks": chunks,
        "by_source": by_source,
        "link_map": link_map,
        "links_out": links_out,
        "urls_out": urls_out,
    }
    if dense is not None:
        index["dense"] = dense

    out_dir = os.path.dirname(out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(out, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        print("ERROR: could not write index to %s: %s" % (out, exc))
        return 1

    try:
        size_bytes = os.path.getsize(out)
    except OSError:
        size_bytes = 0

    notes_with_links = len(links_out)
    total_links = sum(len(v) for v in links_out.values())
    notes_with_urls = len(urls_out)
    total_urls = sum(len(v) for v in urls_out.values())

    print("Files scanned:  %d" % len(md_files))
    print("Chunks built:   %d" % len(chunks))
    print("Metadata notes: %d (--meta %s)" % (meta_notes, meta))
    print("Link map keys:  %d" % len(link_map))
    print("Outgoing links: %d note(s), %d resolved link(s)" % (notes_with_links, total_links))
    print("Reference URLs: %d note(s), %d url(s)" % (notes_with_urls, total_urls))
    if dense is not None:
        dmb = len(dense["buf"]) * 4 / (1024.0 * 1024.0)
        print("Dense vectors:  %d vec(s) x %d dims (%s, ~%.1f MB)" % (
            len(dense["ids"]), dense["dim"], dense["model"], dmb))
    elif embed:
        print("Dense vectors:  NONE (embedding was requested but produced no usable block)")
    print("Output path:    %s" % out)
    print("Output size:    %d bytes (%.2f KB)" % (size_bytes, size_bytes / 1024.0))
    print("DONE: index build complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build the BM25 index pickle from the brain vault markdown."
    )
    parser.add_argument("--vault", default=DEFAULT_VAULT,
                        help="Path to the vault directory of markdown files.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Path to write the bm25.pkl index.")
    parser.add_argument("--embed", action="store_true",
                        help="Also embed every chunk (needs Ollama running) and "
                             "store a dense block for brain.py hybrid retrieval.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Ollama embedding model for --embed (default %(default)s). "
                             "Must match brain.py's index dense model.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST,
                        help="Ollama host for --embed (default %(default)s).")
    parser.add_argument("--embed-prefix", choices=tuple(EMBED_PREFIXES), 
                        default=DEFAULT_EMBED_PREFIX,
                        help="task-prefix scheme for --embed. 'none' (default) "
                             "embeds chunk text verbatim; 'nomic-v1' prepends "
                             "'search_document: ' to documents and makes brain.py "
                             "prepend 'search_query: ' to queries, which is what "
                             "nomic-embed-text was trained on. Recorded in the "
                             "index so the query side cannot drift from it.")
    parser.add_argument("--meta", choices=("none", "inline", "chunk"), default="inline",
                        help="How frontmatter metadata enters retrieval: none "
                             "(pre-A1 behavior), inline (append a metadata "
                             "sentence to each note's first chunk, default), or "
                             "chunk (emit it as a synthetic lead chunk).")
    args = parser.parse_args()
    return build(args.vault, args.out, embed=args.embed,
                 embed_model=args.embed_model, ollama_host=args.ollama_host,
                 meta=args.meta, embed_prefix=args.embed_prefix)


if __name__ == "__main__":
    sys.exit(main())
