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
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_DOC_CHARS = 2000    # truncate a very long chunk before embedding
EMBED_TIMEOUT = 60        # generous: a cold model load on the Jetson is slow


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


def build_dense(chunks, model, host):
    # Embed every chunk once and return a dense block, or None if embedding is
    # not usable (Ollama down / model not pulled / mostly failing). Only
    # successfully-embedded chunks get a row; a chunk that fails to embed is
    # simply dense-invisible (BM25 still covers it), which keeps alignment
    # trivial and never blocks the build.
    embed, normalize = _load_embed_fns()
    total = len(chunks)
    print("Embedding %d chunks with %s (this runs on the Jetson GPU) ..." % (total, model))
    ids = []
    flat = array.array("f")
    dim = None
    failures = 0
    for n, c in enumerate(chunks):
        text = (c.get("text") or "")[:EMBED_DOC_CHARS]
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
    return {"model": model, "dim": dim, "ids": ids, "buf": flat}


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


def parse_aliases(frontmatter_text):
    # Best-effort parse of a simple YAML "aliases:" key from the frontmatter
    # block. Handles:
    #   aliases: foo
    #   aliases: [foo, bar]
    #   aliases:
    #     - foo
    #     - bar
    # Returns a list of raw alias strings (not normalized). Never raises.
    if not frontmatter_text:
        return []
    lines = frontmatter_text.split("\n")
    aliases = []
    n = len(lines)
    for i in range(n):
        line = lines[i]
        m = re.match(r"^\s*(?:aliases|alias)\s*:\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest:
            # Inline value: either [a, b, c] or a single scalar.
            if rest.startswith("[") and rest.endswith("]"):
                inner = rest[1:-1]
                for part in inner.split(","):
                    val = _strip_yaml_scalar(part)
                    if val:
                        aliases.append(val)
            else:
                val = _strip_yaml_scalar(rest)
                if val:
                    aliases.append(val)
        else:
            # Block list: subsequent "- item" lines.
            for j in range(i + 1, n):
                item = lines[j]
                if not item.strip():
                    continue
                lm = re.match(r"^\s*-\s+(.*)$", item)
                if lm:
                    val = _strip_yaml_scalar(lm.group(1))
                    if val:
                        aliases.append(val)
                elif not (item.startswith(" ") or item.startswith("\t")):
                    # A non-list, non-indented line ends the aliases block.
                    break
        # Only honor the first aliases key encountered.
        break
    return aliases


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
_JUNK_DIR_NAMES = {"node_modules", "site-packages", ".venv"}


def build(vault, out, embed=False, embed_model=DEFAULT_EMBED_MODEL,
          ollama_host=DEFAULT_OLLAMA_HOST):
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

        # Parse links and urls from the raw post-frontmatter text, BEFORE
        # clean_obsidian strips the wikilink syntax away.
        link_targets = extract_wikilink_targets(body_no_fm)
        urls = extract_urls(body_no_fm)

        # Structure-aware chunking: section on headings, merge tiny sections,
        # split oversized ones on sentence boundaries, and prepend the heading
        # breadcrumb to each chunk (see chunk_note).
        by_source.setdefault(rel_source, [])
        for chunk_text in chunk_note(body_no_fm, title, fallback_title):
            toks = tokenize(chunk_text)
            if not toks:
                # Nothing tokenizable (punctuation-only); skip so that
                # re-tokenizing this chunk at load can never yield an empty
                # doc. The token list itself is NOT stored (see below).
                continue
            chunks.append({
                "id": next_id,
                "text": chunk_text,
                "source": rel_source,
                "title": title,
            })
            by_source[rel_source].append(next_id)
            next_id += 1

        # Register link_map keys for this note. First writer wins.
        stem = os.path.splitext(os.path.basename(rel_source))[0]
        rel_no_ext = os.path.splitext(rel_source)[0]
        aliases = parse_aliases(frontmatter)
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
    dense = build_dense(chunks, embed_model, ollama_host) if embed else None

    # 'tokenized' is intentionally omitted: it is fully derivable from each
    # chunk's text via tokenize(), so storing it roughly doubled the pickle.
    # load_index() re-tokenizes at load. version 4 signals an optional dense
    # block may be present; brain.py reads it only when present, so a v3 (no
    # dense) and a v4-without-dense index behave identically.
    index = {
        "version": 4,
        "tokenizer_version": TOKENIZER_VERSION,
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
    args = parser.parse_args()
    return build(args.vault, args.out, embed=args.embed,
                 embed_model=args.embed_model, ollama_host=args.ollama_host)


if __name__ == "__main__":
    sys.exit(main())
