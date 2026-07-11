#!/usr/bin/env python3
"""brain.py - interactive local RAG agent for the ThesisVault on a Jetson Orin Nano.

Loads a BM25 index built by build_index.py, routes each question through the
local vault first, and optionally falls out to a web search when the model
signals it needs external or current information. Answers stream to the
terminal. Everything runs locally against Ollama except the optional web fetch.

v2 adds three interlocking features on top of the v1 behavior, all additive:
  1. Wikilink graph expansion: notes linked from the retrieved notes get a
     representative chunk pulled in as extra "linked" context.
  2. Reference URLs: external links found in the retrieved notes are offered to
     the model, which may answer with a single 'FETCH: <url>' line to pull one.
  3. Answer cache: exact and fuzzy question caching for instant repeat answers.
None of the v1 behavior (model picker, streaming, SEARCH web fallback, rolling
history, defensive REST fallbacks, existing commands) is removed.
"""

import os
import re
import sys
import atexit
import pickle
import unicodedata

# Make sure the sibling answer_cache module is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from answer_cache import AnswerCache
except Exception:
    AnswerCache = None
try:
    from brain_ui import UI
except Exception:
    UI = None

# Paths (fixed install layout on the Jetson).
BRAIN_ROOT = "/mnt/nvme/brain"
INDEX_PATH = os.path.join(BRAIN_ROOT, "index", "bm25.pkl")
CACHE_DIR = "/mnt/nvme/brain/cache"

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://127.0.0.1:11434"

# Two-tier models. FAST is the default: small, resident, quick draft answers.
# DEEP is escalated per-question via "/deep <q>": larger, slower, more grounded.
# DEFAULT_MODEL (above) is the single source of truth for the fast/default tier;
# FAST_MODEL is an alias of it, so swapping the small model means editing
# DEFAULT_MODEL only and both stay in lockstep.
FAST_MODEL = DEFAULT_MODEL
DEEP_MODEL = "qwen2.5:7b"

# Generation options threaded into every Ollama chat call: bounded output +
# context window and a low temperature for grounded answers.
GEN_OPTIONS = {"num_predict": 200, "num_ctx": 3072, "temperature": 0.3}

# keep_alive per tier (8GB cannot hold both large models resident): the fast
# model stays pinned (-1) for warm repeats; the deep 7B loads on demand and is
# released a couple minutes after use so it never fights the fast model for VRAM.
KEEP_ALIVE_FAST = -1
KEEP_ALIVE_DEEP = "2m"

# Retrieval depth per tier: fast stays lean for speed, deep pulls more context
# for grounding. TOP_K_VAULT is the default depth for the retrieve() helper
# (a compatibility shim; handle_question does its own tiered retrieval inline).
TOP_K_FAST = 3
TOP_K_DEEP = 6
TOP_K_VAULT = TOP_K_DEEP

# Retrieval section boost: when a query names a top-level vault folder
# (e.g. "CRPD"), add this to the BM25 score of chunks in that folder so
# canonical primary notes (CRPD/Article 2 Definitions) are not buried under
# dense secondary notes that merely repeat the query terms more often.
FOLDER_BOOST = 4.0
MAX_WEB_RESULTS = 5
MAX_WEB_FETCH = 3
WEB_FETCH_TIMEOUT = 8
PAGE_CHAR_CAP = 2000
HISTORY_TURNS = 6  # keep the last N user+assistant exchanges

MAX_LINK_EXPAND = 2  # cap on extra chunks pulled in via wikilink expansion.
# Lowered 4->3->2: each linked chunk is ~230 LLM prefill tokens on the (now single)
# vault call, and prefill is the dominant per-query cost on this device. Trades a
# little answer breadth for speed; raise it back toward 4-6 for max completeness.
MAX_DOC_URLS = 6     # cap on reference URLs offered to the model

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Terminal input hygiene.
#
# In a plain line REPL (input() over readline) the terminal is on the MAIN
# screen buffer, so many terminals (macOS Terminal.app, iTerm2, and anything
# that turned on "alternate scroll" or an xterm mouse-tracking mode) translate
# a mouse wheel tick or a two-finger trackpad scroll into arrow-key escape
# sequences (ESC [ A / ESC [ B). readline then treats those as history
# navigation, so scrolling silently walks through past commands -- the bug
# where "scrolling registered keystrokes when it shouldn't."
#
# The fix is to tell the terminal to stop reporting the mouse to us and to stop
# converting scroll into arrow keys, so a scroll moves the terminal's own
# scrollback buffer again and never reaches the program. We reset every mouse
# mode defensively; unsupported ones are ignored by the terminal.
# ---------------------------------------------------------------------------
# ESC [ ? 1000 l  disable X10/normal mouse click reporting
# ESC [ ? 1002 l  disable button-event (drag) tracking
# ESC [ ? 1003 l  disable any-event (motion) tracking
# ESC [ ? 1006 l  disable SGR extended mouse mode
# ESC [ ? 1015 l  disable urxvt extended mouse mode
# ESC [ ? 1007 l  disable ALTERNATE SCROLL -- the one that turns wheel into arrows
_DISABLE_MOUSE = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?1007l"


def quiet_scroll():
    """Stop the terminal from turning mouse/trackpad scrolls into keystrokes.

    Emits the reset sequences once at startup and re-emits them on exit so a
    program that ran before us (and left a mouse mode on) does not leak into the
    next shell. No-op when stdout is not a TTY (piped or logged). Never raises.
    """
    try:
        if not sys.stdout.isatty():
            return
        sys.stdout.write(_DISABLE_MOUSE)
        sys.stdout.flush()
        atexit.register(_restore_scroll)
    except Exception:
        pass


def _restore_scroll():
    try:
        sys.stdout.write(_DISABLE_MOUSE)
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tokenizer. MUST be identical to the one in build_index.py.
# ---------------------------------------------------------------------------
def _s_stem(w):
    # Conservative S-stemmer: fold only regular English plurals so a query and
    # the corpus agree on "implant"/"implants", "study"/"studies". MUST be
    # byte-identical to build_index.py._s_stem(). Guards keep it from mangling
    # acronyms (len<4), numbers (not isalpha), and -us/-ss/-is words. Irregular
    # plurals (child/children) are intentionally NOT folded.
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
    # Shared tokenizer. MUST be byte-identical to build_index.py.tokenize();
    # any change requires rebuilding the index (bump EXPECTED_TOKENIZER_VERSION
    # and TOKENIZER_VERSION together). Order is load-bearing: NFKD accent-fold
    # (also lowercases via .lower()), collapse dotted acronyms
    # (A.S.L. -> asl, U.S. -> us), split, then S-stem each token.
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), s)
    return [_s_stem(t) for t in re.findall(r"[a-z0-9]+", s)]


# The tokenizer_version a loaded index pickle must carry. Mismatch means the
# pickle was built by a different tokenizer than this file queries with, which
# silently breaks retrieval -- load_index refuses rather than degrade quietly.
EXPECTED_TOKENIZER_VERSION = "tok-nfkd-acr-sstem"


# Grammatical filler dropped from the QUERY before BM25 scoring (query-side
# only -- the corpus tokenization and pickle are untouched). rank_bm25 floors
# the negative IDF of near-ubiquitous words to a small positive epsilon, so
# leaving "what/is/the" in a query adds a spurious bump to nearly every chunk
# and flattens the top-vs-rest gap the insufficiency gate depends on.
# Stemmed with _s_stem so the set matches the (now stemmed) tokens query_tokens
# filters -- e.g. tokenize("does") -> "doe", so "doe" must be the stored form.
QUERY_STOPWORDS = frozenset(
    _s_stem(w) for w in (
        "a an the of to in on for and or is are was were be been what which who "
        "whom when where why how do does did can could would should will this that "
        "these those with as at by from about it its into".split()
    )
)


def query_tokens(question):
    """Tokenize a question and drop filler, but never return an empty query
    (fall back to the full token list so an all-stopword question still scores)."""
    toks = tokenize(question)
    kept = [t for t in toks if t not in QUERY_STOPWORDS]
    return kept or toks


# ---------------------------------------------------------------------------
# Index loading.
# ---------------------------------------------------------------------------
def load_index():
    """Load bm25.pkl and rebuild a BM25Okapi over the tokenized corpus.

    Returns (chunks, bm25, meta) where meta carries the v2 graph maps:
        meta = {
            "by_source": {source: [chunk_id, ...]},
            "link_map":  {normalized_key: source},
            "links_out": {source: [target_source, ...]},
            "urls_out":  {source: [external_url, ...]},
            "by_id":     {chunk_id: chunk_dict},
        }
    Backward compatible: an old (v1) pickle without the new keys is upgraded on
    the fly by backfilling by_source and link_map from the chunks, with
    links_out and urls_out left empty. Exits the process with a clear message
    if the index is missing or unreadable.
    """
    if not os.path.exists(INDEX_PATH):
        print("No index found at " + INDEX_PATH + ".")
        print("Build it first with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    try:
        with open(INDEX_PATH, "rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print("Could not read the index at " + INDEX_PATH + ": " + str(exc))
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    if not isinstance(data, dict):
        print("The index has an unexpected format (not a dict).")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    chunks = data.get("chunks") or []
    tokenized = data.get("tokenized")
    if not tokenized:
        # v3+ indexes omit the derivable 'tokenized' array to roughly halve the
        # pickle. Rebuild it from chunk text with the same tokenizer as build.
        # (build_index skips empty-token chunks, so no doc is empty here.)
        tokenized = [tokenize(c.get("text") or "") for c in chunks]
    if not chunks or len(chunks) != len(tokenized):
        print("The index looks empty or corrupt.")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    # Refuse a pickle built by a different tokenizer than this file queries with:
    # index tokens and query tokens must come from the same tokenize(), or
    # retrieval silently degrades. An old pre-versioning pickle has no field and
    # also fails closed here, prompting a rebuild.
    idx_tok_ver = data.get("tokenizer_version")
    if idx_tok_ver != EXPECTED_TOKENIZER_VERSION:
        print("Index tokenizer mismatch: index=%r, expected=%r." % (
            idx_tok_ver, EXPECTED_TOKENIZER_VERSION))
        print("The index was built by a different tokenizer version.")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:
        print("The rank_bm25 package is not available: " + str(exc))
        print("Install it with:  python3 -m pip install --user rank_bm25")
        sys.exit(1)

    try:
        bm25 = BM25Okapi(tokenized)
    except Exception as exc:
        print("Could not rebuild the BM25 index: " + str(exc))
        sys.exit(1)

    meta = _build_meta(data, chunks)
    return chunks, bm25, meta


def _normalize_key(key):
    """Lowercase, strip, and collapse internal whitespace to single spaces."""
    try:
        return " ".join(str(key).strip().lower().split())
    except Exception:
        return ""


def _build_meta(data, chunks):
    """Assemble the v2 meta maps from the pickle, backfilling for old indexes."""
    by_source = data.get("by_source")
    link_map = data.get("link_map")
    links_out = data.get("links_out")
    urls_out = data.get("urls_out")

    need_backfill = not isinstance(by_source, dict) or not isinstance(link_map, dict) \
        or not by_source or not link_map

    if need_backfill:
        by_source = {}
        link_map = {}
        for c in chunks:
            src = c.get("source")
            cid = c.get("id")
            if src is None or cid is None:
                continue
            by_source.setdefault(src, []).append(cid)
            # filename stem without .md
            stem = os.path.splitext(os.path.basename(src))[0]
            skey = _normalize_key(stem)
            if skey and skey not in link_map:
                link_map[skey] = src
            # full rel path without .md
            rel = src[:-3] if src.endswith(".md") else src
            rkey = _normalize_key(rel)
            if rkey and rkey not in link_map:
                link_map[rkey] = src

    if not isinstance(links_out, dict):
        links_out = {}
    if not isinstance(urls_out, dict):
        urls_out = {}

    by_id = {}
    for c in chunks:
        cid = c.get("id")
        if cid is not None:
            by_id[cid] = c

    return {
        "by_source": by_source,
        "link_map": link_map,
        "links_out": links_out,
        "urls_out": urls_out,
        "by_id": by_id,
    }


def _apply_section_boost(scores, chunks, qtok):
    """Add FOLDER_BOOST to any chunk whose top-level vault folder is named
    in the query tokens, so canonical primary notes (e.g. CRPD/Article 2
    Definitions) are not buried under dense secondary notes. Returns scores
    unchanged when the query names no folder. Shared by retrieve() and
    handle_question so the ranking the model sees matches what any inspector
    or measurement sees."""
    if not len(scores):
        return scores
    qset = set(qtok)
    boosted = None
    for i, c in enumerate(chunks):
        src = str(c.get("source") or "")
        top = src.split("/", 1)[0] if "/" in src else ""
        if top and (set(tokenize(top)) & qset):
            if boosted is None:
                boosted = list(scores)
            boosted[i] += FOLDER_BOOST
    return boosted if boosted is not None else scores


def retrieve(bm25, chunks, query, k=TOP_K_VAULT):
    """Return up to k top-scoring chunks for the query. Never raises.

    Kept for compatibility. handle_question computes scores directly so it can
    reuse them for wikilink expansion, but this helper is still valid.
    """
    qtok = query_tokens(query)
    try:
        scores = bm25.get_scores(qtok)
    except Exception:
        return []
    scores = _apply_section_boost(scores, chunks, qtok)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked[:k]:
        if scores[i] <= 0:
            continue
        out.append(chunks[i])
    return out


# ---------------------------------------------------------------------------
# Ollama helpers. Prefer the python client, fall back to REST via requests.
# ---------------------------------------------------------------------------
def list_models():
    """Return a list of installed model names. Best effort, never raises."""
    names = []
    try:
        import ollama
        data = ollama.list()
        models = getattr(data, "models", None)
        if models is None and isinstance(data, dict):
            models = data.get("models", [])
        for m in models or []:
            name = getattr(m, "model", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        if names:
            return names
    except Exception:
        pass

    # REST fallback.
    try:
        import requests
        resp = requests.get(OLLAMA_HOST + "/api/tags", timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        for m in payload.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
    except Exception:
        pass
    return names


def chat_once(model, messages):
    """Non-streaming chat. Returns the full text, or an error string prefixed
    with '[error]' so callers can detect failure without a crash."""
    try:
        import ollama
        _ka = KEEP_ALIVE_DEEP if model == DEEP_MODEL else KEEP_ALIVE_FAST
        resp = ollama.chat(model=model, messages=messages, stream=False, options=GEN_OPTIONS, keep_alive=_ka)
        msg = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
        content = getattr(msg, "content", None)
        if content is not None:
            return content
    except Exception:
        pass

    # REST fallback.
    try:
        import requests
        resp = requests.post(
            OLLAMA_HOST + "/api/chat",
            json={"model": model, "messages": messages, "stream": False, "options": GEN_OPTIONS, "keep_alive": (KEEP_ALIVE_DEEP if model == DEEP_MODEL else KEEP_ALIVE_FAST)},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""
    except Exception as exc:
        return "[error] could not reach the model: " + str(exc)


def chat_stream(model, messages, sink=None):
    """Stream a chat reply and return the accumulated text.

    Each token is passed to `sink` when given (the UI answer emitter); otherwise
    it is written straight to stdout. On failure prints a short note and returns
    what it has (possibly empty).
    """
    collected = []

    def _emit(piece):
        if sink is not None:
            sink(piece)
        else:
            sys.stdout.write(piece)
            sys.stdout.flush()

    try:
        import ollama
        _ka = KEEP_ALIVE_DEEP if model == DEEP_MODEL else KEEP_ALIVE_FAST
        for part in ollama.chat(model=model, messages=messages, stream=True, options=GEN_OPTIONS, keep_alive=_ka):
            piece = ""
            msg = part.get("message") if isinstance(part, dict) else getattr(part, "message", None)
            if isinstance(msg, dict):
                piece = msg.get("content", "") or ""
            elif msg is not None:
                piece = getattr(msg, "content", "") or ""
            if piece:
                _emit(piece)
                collected.append(piece)
        if sink is None:
            print()
        return "".join(collected)
    except Exception:
        pass

    # REST streaming fallback.
    try:
        import json
        import requests
        with requests.post(
            OLLAMA_HOST + "/api/chat",
            json={"model": model, "messages": messages, "stream": True, "options": GEN_OPTIONS, "keep_alive": (KEEP_ALIVE_DEEP if model == DEEP_MODEL else KEEP_ALIVE_FAST)},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                piece = (obj.get("message") or {}).get("content", "") or ""
                if piece:
                    _emit(piece)
                    collected.append(piece)
        if sink is None:
            print()
        return "".join(collected)
    except Exception as exc:
        msg = "\n[the model could not stream a reply: " + str(exc) + "]"
        if sink is not None:
            sink(msg)
        else:
            print(msg)
        return "".join(collected)


# ---------------------------------------------------------------------------
# Web search + fetch.
# ---------------------------------------------------------------------------
def web_search(query):
    """Run a DuckDuckGo text search. Returns a list of result dicts or []."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=MAX_WEB_RESULTS)
        return list(results) if results else []
    except Exception:
        return []


def fetch_page(url):
    """Fetch a URL and return cleaned readable text capped at PAGE_CHAR_CAP.
    Returns '' on any failure."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:PAGE_CHAR_CAP]
    except Exception:
        return ""


def gather_web_context(query):
    """Search, fetch the top few hrefs, and return a list of (url, excerpt)."""
    results = web_search(query)
    excerpts = []
    for r in results:
        if len(excerpts) >= MAX_WEB_FETCH:
            break
        href = r.get("href") or r.get("url") or r.get("link")
        if not href:
            continue
        text = fetch_page(href)
        if not text:
            # Fall back to the search snippet so the model still has something.
            text = (r.get("body") or "").strip()
        if text:
            excerpts.append((href, text))
    return excerpts


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------
FIRST_PASS_SYSTEM = (
    "You are a careful research assistant grounded in a local thesis vault. "
    "You are given retrieved snippets from the vault, each labeled with its "
    "source filename. Some snippets may be labeled '(linked)' because they come "
    "from notes that the top notes link to. You may also be given a list of "
    "reference links found in those notes.\n"
    "First try to answer from the vault context and cite the sources by "
    "filename when you use them.\n"
    "If the vault context is not enough, YOU decide how to reach the web and "
    "reply with EXACTLY ONE line and nothing else, choosing between:\n"
    "- 'FETCH: <url>' when the answer lives on ONE specific web page: the "
    "question names or clearly implies a particular website, document, paper, "
    "or organization, or one of the reference links above is the right source. "
    "Give the single most relevant full URL (it does not have to be one of the "
    "reference links; use any specific URL you are confident points at the "
    "answer).\n"
    "- 'SEARCH: <query>' when no single page is obvious and you need to discover "
    "sources, compare several results, or get current or general information. "
    "Give a good web search query.\n"
    "Rule of thumb: FETCH only when you are confident of the exact page (a "
    "reference link above or a well known stable URL); SEARCH for anything "
    "current or time sensitive (weather, prices, latest news, recent events) or "
    "whenever you are not sure the exact URL exists. When you go to the web, "
    "output only that one FETCH or SEARCH line with no other text."
)

# Slim routing prompt for Phase 2 (Q1). Unlike FIRST_PASS_SYSTEM it does NOT
# reference vault snippets, because the routing call no longer receives them:
# Phase 1 already judged the vault insufficient. It only needs the question and
# any reference URLs to pick FETCH vs SEARCH.
ROUTE_SYSTEM = (
    "The user's question could not be answered from a local thesis vault, so "
    "you must reach the web. You are given the question and possibly a list of "
    "reference links found in the vault notes. Reply with EXACTLY ONE line and "
    "nothing else, choosing between:\n"
    "- 'FETCH: <url>' when the answer lives on ONE specific web page: the "
    "question names or clearly implies a particular website, document, paper, "
    "or organization, or one of the reference links is the right source. Give "
    "the single most relevant full URL.\n"
    "- 'SEARCH: <query>' when no single page is obvious, or for anything current "
    "or time sensitive (weather, prices, latest news, recent events). Give a "
    "good web search query.\n"
    "Output only that one FETCH or SEARCH line with no other text."
)

WEB_SYSTEM = (
    "You are a careful research assistant. You are given excerpts fetched from "
    "the web, each labeled with its source URL, plus the user's question. "
    "Answer the question using these excerpts and cite the URLs you relied on."
)

DOC_SYSTEM = (
    "You are a careful research assistant. You are given the readable text of a "
    "single reference page fetched from a URL, plus the user's question. Answer "
    "the question from this fetched reference and cite the URL you relied on."
)

NO_WEB_SYSTEM = (
    "You are a careful research assistant grounded in a local thesis vault. "
    "Web search is disabled, so answer the question as best you can from the "
    "vault snippets provided and your own knowledge. Cite vault sources by "
    "filename when you use them, and note plainly if you are unsure."
)

# Phase 1: the vault is ALWAYS consulted for an answer first. The model may only
# reach the web after this phase reports the vault cannot answer.
VAULT_FIRST_SYSTEM = (
    "You are a careful research assistant grounded in a local thesis vault. "
    "You are given retrieved snippets from the vault, each labeled with its "
    "source filename. Some snippets may be labeled '(linked)' because they come "
    "from notes that the top notes link to.\n"
    "Answer the question USING ONLY the vault snippets provided. Cite the "
    "sources by filename when you use them.\n"
    "If, and only if, the vault snippets do not contain enough information to "
    "answer the question, reply with EXACTLY the single word INSUFFICIENT and "
    "nothing else. Do not guess, do not use outside knowledge, and do not "
    "mention the web. Only emit INSUFFICIENT when the vault genuinely cannot "
    "answer; if the snippets let you answer even partially, answer instead. "
    "Never supply article numbers, statutes, dates, statistics, author-year "
    "citations, filenames, or URLs from your own memory: include such "
    "specifics ONLY if they appear verbatim in the snippets. If the snippets "
    "discuss the topic but lack the exact fact asked for, state what the "
    "snippets DO support and say plainly that the specific detail is not in "
    "the vault. "
    "Answer in 3 to 4 concise sentences. Use only the provided context."
)


# Free-form refusals a small model emits INSTEAD of the exact INSUFFICIENT
# sentinel that VAULT_FIRST_SYSTEM asks for. llama3.2:3b cannot be reliably
# forced to emit a bare sentinel, so it paraphrases ("I can't answer this
# question."), which the old literal-token check let through as a real answer.
# Phrases are curated to be specific enough that a genuine substantive vault
# answer is very unlikely to contain them, and are only matched inside the
# reply's opening clause (see _is_insufficient) so a long, real answer that
# merely quotes one deep in its body is not misclassified.
_REFUSAL_PHRASES = (
    "i can't answer",
    "i cannot answer",
    "i can not answer",
    "can't answer this",
    "cannot answer this",
    "unable to answer",
    "not able to answer",
    "cannot be answered",
    "can't be answered",
    "i don't know",
    "i do not know",
    "i'm unable to",
    "i am unable to",
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "there is no information",
    "no relevant information",
    "does not contain enough",
    "doesn't contain enough",
    "do not contain enough",
    "don't contain enough",
    "cannot provide an answer",
    "can't provide an answer",
)


def _is_insufficient(text):
    """True when the vault-first pass could not answer from the vault.

    Recognizes BOTH the exact INSUFFICIENT sentinel from VAULT_FIRST_SYSTEM and
    the free-form refusals small models emit instead ("I can't answer this
    question.", "I don't know", "the vault does not contain enough ..."). The
    refusal phrases are only searched inside the reply's opening ~200 chars,
    where a genuine refusal always sits, so a long real answer that happens to
    contain such a phrase later is not treated as a refusal."""
    if not text:
        return True
    # Normalize: collapse internal whitespace, strip wrapping quotes/punct,
    # lowercase. re is already imported at module top.
    t = re.sub(r"\s+", " ", text.strip()).strip(".!?\"'` ").lower()
    if not t:
        return True
    # Exact sentinel (case-insensitive), tolerant of a trailing clause like
    # "INSUFFICIENT - the snippets do not ...". Preserves original semantics.
    if t == "insufficient" or t.startswith("insufficient"):
        return True
    # Free-form refusal: only look at the opening clause.
    if any(p in t[:200] for p in _REFUSAL_PHRASES):
        return True
    return False


_CITE_URL_RE = re.compile(r"https?://\S+")
_CITE_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Za-z]?\b")


def faithfulness_flags(answer, context_text):
    """Citation-like tokens asserted in `answer` that do NOT appear in the
    retrieved vault context. High precision by design: only URLs and
    'Article N' references, the two fabrication types small local models most
    reliably invent. Surfaced as a caution so unverifiable specifics are
    flagged rather than silently trusted."""
    if not answer:
        return []
    ctx = (context_text or "").lower()
    flags, seen = [], set()
    for m in _CITE_URL_RE.findall(answer):
        tok = m.rstrip(".,);]")
        k = tok.lower()
        if k not in ctx and k not in seen:
            seen.add(k)
            flags.append(tok)
    for m in _CITE_ARTICLE_RE.findall(answer):
        k = m.lower()
        if k not in ctx and k not in seen:
            seen.add(k)
            flags.append(m)
    return flags


def format_vault_context(chunks, linked_ids=None):
    if not chunks:
        return "(no relevant vault snippets were found)"
    linked_ids = linked_ids or set()
    blocks = []
    for c in chunks:
        src = c.get("source") or c.get("title") or "unknown"
        label = src
        if c.get("id") in linked_ids:
            label = src + " (linked)"
        text = c.get("text") or ""
        blocks.append("[source: " + label + "]\n" + text)
    return "\n\n".join(blocks)


def format_web_context(excerpts):
    blocks = []
    for url, text in excerpts:
        blocks.append("[url: " + url + "]\n" + text)
    return "\n\n".join(blocks)


def build_first_pass_user(question, chunks, urls=None, linked_ids=None):
    parts = ["Vault context:\n" + format_vault_context(chunks, linked_ids)]
    if urls:
        parts.append("Reference links found in these notes:\n" + "\n".join(urls))
    parts.append("Question: " + question)
    return "\n\n".join(parts)


def normalize_url(raw):
    """Pull a single URL out of a FETCH line and normalize it. Adds an https://
    scheme when it is missing so the model can name a bare domain. Returns '' if
    the text does not look like a URL."""
    raw = (raw or "").strip().strip("<>\"'")
    if not raw:
        return ""
    # Take the first whitespace-delimited token in case of trailing text.
    url = raw.split()[0].strip("<>\"'").rstrip(".,;:!?)]}")
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    # Must look like host.tld to be worth a fetch.
    if not re.search(r"[a-z0-9][a-z0-9\-]*\.[a-z]{2,}", url, re.IGNORECASE):
        return ""
    return url


def answer_from_web(model, query, question, history):
    """Search the web for query, fetch the top pages, and stream an answer to
    the original question. Returns (answer, urls), or (None, []) when there
    were no usable results so the caller can fall back."""
    excerpts = gather_web_context(query)
    if not excerpts:
        return None, []
    web_messages = [{"role": "system", "content": WEB_SYSTEM}]
    web_messages.extend(history)
    web_messages.append(
        {
            "role": "user",
            "content": (
                "Web excerpts:\n"
                + format_web_context(excerpts)
                + "\n\nQuestion: "
                + question
            ),
        }
    )
    answer = chat_stream(model, web_messages)
    urls = [u for u, _ in excerpts]
    if urls:
        print("sources: " + ", ".join(urls))
    return answer, urls


def vault_sources(chunks):
    """Ordered, de-duplicated list of source names for the given chunks."""
    sources = []
    for c in chunks:
        src = c.get("source") or c.get("title")
        if src and src not in sources:
            sources.append(src)
    return sources


# ---------------------------------------------------------------------------
# Model menu.
# ---------------------------------------------------------------------------
def choose_model():
    """Show a numbered menu of installed models and return one name.
    Enter with no input selects DEFAULT_MODEL. Single selection, no looping."""
    models = list_models()
    if not models:
        print("Could not list installed models. Falling back to " + DEFAULT_MODEL + ".")
        return DEFAULT_MODEL

    print("\nInstalled models:")
    for idx, name in enumerate(models, start=1):
        marker = "  (default)" if name == DEFAULT_MODEL else ""
        print("  " + str(idx) + ". " + name + marker)
    if DEFAULT_MODEL not in models:
        print("  (Enter selects the default " + DEFAULT_MODEL + ", even though it is not listed above.)")

    try:
        raw = input("Pick a model number, or press Enter for the default: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return DEFAULT_MODEL

    if raw == "":
        return DEFAULT_MODEL
    if raw.isdigit():
        i = int(raw)
        if 1 <= i <= len(models):
            return models[i - 1]
    print("Unrecognized selection, using the default " + DEFAULT_MODEL + ".")
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# History management.
# ---------------------------------------------------------------------------
def trim_history(history):
    """Keep only the last HISTORY_TURNS user+assistant exchanges."""
    max_msgs = HISTORY_TURNS * 2
    if len(history) > max_msgs:
        del history[: len(history) - max_msgs]


# ---------------------------------------------------------------------------
# Retrieval + wikilink expansion.
# ---------------------------------------------------------------------------
def _score_at(scores, cid):
    """Safe lookup into the score array by chunk id."""
    try:
        if 0 <= cid < len(scores):
            return scores[cid]
    except Exception:
        pass
    return 0


def expand_with_links(retrieved, scores, meta):
    """Pull in a representative chunk for each note linked from the retrieved
    notes. Returns (added_chunks, linked_ids). Capped at MAX_LINK_EXPAND and
    de-duplicated by chunk id. Never raises."""
    added = []
    linked_ids = set()
    try:
        by_source = meta.get("by_source") or {}
        by_id = meta.get("by_id") or {}
        links_out = meta.get("links_out") or {}

        represented = set()
        seen_ids = set()
        for c in retrieved:
            src = c.get("source")
            if src:
                represented.add(src)
            cid = c.get("id")
            if cid is not None:
                seen_ids.add(cid)

        for c in list(retrieved):
            if len(added) >= MAX_LINK_EXPAND:
                break
            src = c.get("source")
            if not src:
                continue
            for target in links_out.get(src, []):
                if len(added) >= MAX_LINK_EXPAND:
                    break
                if not target or target in represented:
                    continue
                ids = by_source.get(target) or []
                if not ids:
                    continue
                # Pick the target note's best-scoring chunk; if all zero, first.
                best_id = None
                best_score = None
                for cid in ids:
                    sc = _score_at(scores, cid)
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_id = cid
                if best_id is None or best_score is None or best_score <= 0:
                    best_id = ids[0]
                chunk = by_id.get(best_id)
                if chunk is None:
                    continue
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                added.append(chunk)
                represented.add(target)
                seen_ids.add(cid)
                if cid is not None:
                    linked_ids.add(cid)
    except Exception:
        return added, linked_ids
    return added, linked_ids


def collect_reference_urls(chunks, meta):
    """De-duplicated list of external URLs found in the given notes, capped."""
    urls = []
    try:
        urls_out = meta.get("urls_out") or {}
        for c in chunks:
            src = c.get("source")
            if not src:
                continue
            for u in urls_out.get(src, []):
                if u and u not in urls:
                    urls.append(u)
                if len(urls) >= MAX_DOC_URLS:
                    return urls
    except Exception:
        return urls
    return urls


# ---------------------------------------------------------------------------
# One question, routed. Returns (answer_text, sources_list).
# ---------------------------------------------------------------------------
def _short_srcs(sources, n=3):
    """First n source basenames, comma-joined, for a compact tree branch."""
    names = []
    for s in sources[:n]:
        try:
            names.append(os.path.splitext(os.path.basename(str(s)))[0])
        except Exception:
            names.append(str(s))
    tail = ", …" if len(sources) > n else ""
    return ", ".join(names) + tail


def _finish(run, history, question, answer, sources):
    """Common tail: append to history, print the sources branch, return."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    trim_history(history)
    if run is not None:
        run.sources(sources)
    return answer, sources


def handle_question(model, question, history, bm25, chunks, meta, web_enabled, run):
    """Route a single question and stream the final answer through the run's UI.
    Updates history and returns (answer_text, sources_list). On error paths
    returns ("", []). `run` drives the ultracode-style progress tree; it always
    exists (it degrades to plain output when stdout is not a TTY)."""
    # --- Retrieval: BM25 scoring -----------------------------------------
    run.running("bm25")
    qtok = query_tokens(question)
    try:
        scores = bm25.get_scores(qtok)
    except Exception:
        scores = []
    scores = _apply_section_boost(scores, chunks, qtok)

    retrieved = []
    top_score = 0.0
    if len(scores):
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        for i in ranked[:(TOP_K_DEEP if model == DEEP_MODEL else TOP_K_FAST)]:
            if scores[i] <= 0:
                continue
            retrieved.append(chunks[i])
        if ranked:
            try:
                top_score = float(scores[ranked[0]])
            except Exception:
                top_score = 0.0
    vsrc = vault_sources(retrieved)
    run.done("bm25", "%d chunks · top score %.1f · %s" % (
        len(retrieved), top_score, _short_srcs(vsrc)) if retrieved
        else "no lexical matches")

    # --- Retrieval: wikilink expansion -----------------------------------
    run.running("expand")
    added, linked_ids = expand_with_links(retrieved, scores, meta)
    context_chunks = retrieved + added
    if added:
        run.done("expand", "+%d linked chunks (%d total)" % (
            len(added), len(context_chunks)))
    else:
        run.skip("expand", "no wikilinks to follow")

    # Reference URLs offered to the model for optional FETCH.
    ref_urls = collect_reference_urls(context_chunks, meta)

    # === PHASE 1: vault-first answer attempt (ALWAYS) ====================
    # The thesis vault is consulted for an actual answer before any web access.
    # Strict grounding (VAULT_FIRST_SYSTEM): answer only from the snippets, or
    # emit INSUFFICIENT. This keeps the common vault-answerable case at ONE
    # grounded call. (A merged answer-or-route call was tried and rejected: it
    # made the small model leave the vault for the web on questions the vault
    # answers well -- slower and less grounded.)
    run.running("model")
    vault_messages = [{"role": "system", "content": VAULT_FIRST_SYSTEM}]
    vault_messages.extend(history)
    vault_messages.append(
        {"role": "user", "content": build_first_pass_user(
            question, context_chunks, None, linked_ids)}
    )
    first = chat_once(model, vault_messages).strip()

    if first.startswith("[error]"):
        run.fail("model", first[:80])
        run.skip("web")
        if run is not None:
            run.finish()
        return "", []

    vault_ok = bool(context_chunks) and not _is_insufficient(first)

    # Case C: the vault answered. Stream the answer we already computed.
    if vault_ok:
        run.done("model", "%d tokens · from vault" % len(tokenize(first)))
        run.skip("web", "vault answer sufficient")
        _flags = faithfulness_flags(
            first, format_vault_context(context_chunks, linked_ids))
        if _flags:
            first = first.rstrip() + (
                "\n\n[grounding check \u2014 not found in the retrieved "
                "vault sources, treat as unverified: "
                + ", ".join(_flags) + "]")
        sink = run.begin_answer()
        sink(first)
        return _finish(run, history, question, first,
                       vault_sources(context_chunks))

    # Vault could not answer. With web off, answer best-effort without it.
    if not web_enabled:
        run.done("model", "vault insufficient")
        run.skip("web", "web search is off")
        nw_messages = [{"role": "system", "content": NO_WEB_SYSTEM}]
        nw_messages.extend(history)
        nw_messages.append(
            {"role": "user", "content": build_first_pass_user(
                question, context_chunks, None, linked_ids)}
        )
        answer = chat_stream(model, nw_messages, sink=run.begin_answer())
        return _finish(run, history, question, answer,
                       vault_sources(context_chunks))

    # === PHASE 2: web routing (only because the vault was insufficient) ===
    # Q1: the routing call does NOT re-send the vault context. Phase 1 already
    # judged it insufficient, and the FETCH-vs-SEARCH choice only needs the
    # question plus any reference URLs from the notes. Dropping the ~1.5-2k
    # vault-context tokens here removes most of this call's prefill cost.
    route_user = "Question: " + question
    if ref_urls:
        route_user = ("Reference links found in the vault notes:\n"
                      + "\n".join(ref_urls) + "\n\n" + route_user)
    route_messages = [{"role": "system", "content": ROUTE_SYSTEM}]
    route_messages.extend(history)
    route_messages.append({"role": "user", "content": route_user})
    route = chat_once(model, route_messages).strip()
    if route.startswith("[error]"):
        route = "SEARCH: " + question

    wants_fetch = route.startswith("FETCH:")
    # Anything that is not an explicit FETCH becomes a web SEARCH.
    first = route if wants_fetch or route.startswith("SEARCH:") \
        else ("SEARCH: " + question)
    wants_search = not wants_fetch
    run.done("model", "vault insufficient · going to web")

    # Case F: model asked to fetch one specific reference page.
    if wants_fetch:
        url = normalize_url(first[len("FETCH:"):])
        if url:
            run.running("web", "fetching " + url)
            text = fetch_page(url)
            if text:
                run.done("web", "fetched " + url)
                doc_messages = [{"role": "system", "content": DOC_SYSTEM}]
                doc_messages.extend(history)
                doc_messages.append(
                    {"role": "user", "content": (
                        "Reference page (" + url + "):\n" + text
                        + "\n\nQuestion: " + question)}
                )
                run.done("model")
                answer = chat_stream(model, doc_messages, sink=run.begin_answer())
                return _finish(run, history, question, answer, [url])
        # Direct fetch failed: fall back to a web search before the vault.
        if web_enabled:
            run.running("web", "direct fetch failed, searching")
            excerpts = gather_web_context(question)
            if excerpts:
                run.done("web", "DuckDuckGo · %d results" % len(excerpts))
                web_messages = [{"role": "system", "content": WEB_SYSTEM}]
                web_messages.extend(history)
                web_messages.append(
                    {"role": "user", "content": (
                        "Web excerpts:\n" + format_web_context(excerpts)
                        + "\n\nQuestion: " + question)}
                )
                run.done("model")
                answer = chat_stream(model, web_messages, sink=run.begin_answer())
                return _finish(run, history, question, answer,
                               [u for u, _ in excerpts])
            run.skip("web", "no usable web results")
        # No web result either: answer from the vault.
        run.done("model")
        fb_messages = [{"role": "system", "content": NO_WEB_SYSTEM}]
        fb_messages.extend(history)
        fb_messages.append(
            {"role": "user", "content": build_first_pass_user(
                question, context_chunks, None, linked_ids)}
        )
        answer = chat_stream(model, fb_messages, sink=run.begin_answer())
        return _finish(run, history, question, answer, vault_sources(context_chunks))

    # Case A: model wants to search and web is on.
    if wants_search and web_enabled:
        query = first[len("SEARCH:"):].strip() or question
        run.running("web", "searching: " + query[:40])
        excerpts = gather_web_context(query)
        if not excerpts:
            run.skip("web", "no usable web results")
            run.done("model")
            fallback_messages = [{"role": "system", "content": NO_WEB_SYSTEM}]
            fallback_messages.extend(history)
            fallback_messages.append(
                {"role": "user", "content": build_first_pass_user(
                    question, context_chunks, None, linked_ids)}
            )
            answer = chat_stream(model, fallback_messages, sink=run.begin_answer())
            return _finish(run, history, question, answer,
                           vault_sources(context_chunks))
        run.done("web", "DuckDuckGo · %d results" % len(excerpts))
        web_messages = [{"role": "system", "content": WEB_SYSTEM}]
        web_messages.extend(history)
        web_messages.append(
            {"role": "user", "content": (
                "Web excerpts:\n" + format_web_context(excerpts)
                + "\n\nQuestion: " + question)}
        )
        run.done("model")
        answer = chat_stream(model, web_messages, sink=run.begin_answer())
        return _finish(run, history, question, answer, [u for u, _ in excerpts])

    # Safety net: phase 2 always yields a FETCH or SEARCH handled above, so this
    # is normally unreachable. If we ever get here, answer from the vault context
    # rather than dropping the question.
    run.done("model")
    run.skip("web", "no web route")
    fb_messages = [{"role": "system", "content": NO_WEB_SYSTEM}]
    fb_messages.extend(history)
    fb_messages.append(
        {"role": "user", "content": build_first_pass_user(
            question, context_chunks, None, linked_ids)}
    )
    answer = chat_stream(model, fb_messages, sink=run.begin_answer())
    return _finish(run, history, question, answer, vault_sources(context_chunks))


# ---------------------------------------------------------------------------
# REPL.
# ---------------------------------------------------------------------------
HELP_TEXT = """Commands:
  /deep <q>      re-ask one question with the deep 7B model (slower, more grounded)
  /model         re-pick the model
  /web on|off    allow / disallow web search
  /cache         show the knowledge-bank stats (entries, reuses, balance, size)
  /cache clear   delete all cached answers
  /cache on|off  enable / disable the answer cache
  /clear         forget the conversation history
  /help          show this help
  /exit          quit (Ctrl-D and Ctrl-C also quit)
Just type a question to ask the vault."""


def parse_args(argv):
    model = None
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        else:
            i += 1
    return model


def _cache_stats_line(cache, cache_enabled):
    """Human-readable one-liner for /cache including the knowledge-bank balance."""
    try:
        st = cache.stats()
        kb = st.get("bytes", 0) / 1024.0
        return (
            "Knowledge bank: %d entries · %d reuses · balance %.1f · %.0f KB on disk · %s"
            % (st.get("count", 0), st.get("hits", 0), st.get("bank", 0.0), kb,
               "ON" if cache_enabled else "OFF")
        )
    except Exception as exc:
        return "[could not read cache stats: " + str(exc) + "]"


class _NullRun:
    """No-op stand-in for a UI Run, used only when brain_ui failed to import
    (ui is None). handle_question and the cache path call run.* unconditionally;
    without this they would crash the plain no-UI fallback with AttributeError.
    Stage methods do nothing (there is no tree to render); the answer sink and
    sources() write plainly to stdout so the fallback still produces output."""

    def running(self, *args, **kwargs):
        pass

    def done(self, *args, **kwargs):
        pass

    def skip(self, *args, **kwargs):
        pass

    def fail(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def _emit(self, piece):
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()

    def begin_answer(self):
        sys.stdout.write("\n")
        return self._emit

    def sources(self, srcs):
        if srcs:
            sys.stdout.write(
                "\nsources: " + ", ".join(str(s) for s in srcs) + "\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    model = parse_args(sys.argv[1:])

    # Stop mouse-wheel / trackpad scrolls from leaking in as arrow keys.
    quiet_scroll()

    ui = UI() if UI is not None else None

    if ui is None or ui.plain:
        print("Loading the vault index...")
    chunks, bm25, meta = load_index()

    if not model:
        model = choose_model()

    # Answer cache. Degrade gracefully if the module is missing or unusable.
    cache = None
    cache_enabled = True
    if AnswerCache is not None:
        try:
            cache = AnswerCache(CACHE_DIR)
        except Exception:
            cache = None
    if cache is None:
        cache_enabled = False

    # Welcome banner (Claude Code style) or a plain header.
    cache_count = 0
    if cache is not None:
        try:
            cache_count = cache.stats().get("count", 0)
        except Exception:
            cache_count = 0
    if ui is not None:
        ui.welcome(model, len(chunks), cache_count, "ThesisVault", CACHE_DIR)
        if cache is None:
            ui.note("[answer cache unavailable, running without it]")
    else:
        print("Loaded %d chunks. Using model: %s" % (len(chunks), model))
        print("Type /help for commands. Ctrl-D or /exit to quit.")

    web_enabled = True
    history = []

    while True:
        try:
            line = ui.read_prompt() if ui is not None else input("\nyou> ")
        except EOFError:
            print("\nbye")
            break
        except KeyboardInterrupt:
            print("\nbye")
            break

        question = line.strip()
        if not question:
            continue

        # Per-question tier: default fast; "/deep <q>" escalates to the deep model.
        deep_once = False

        # Commands.
        if question in ("/exit", "/quit"):
            print("bye")
            break
        if question == "/help":
            print(HELP_TEXT)
            continue
        if question in ("/clear",):
            history = []
            print("History cleared.")
            continue
        if question == "/model":
            model = choose_model()
            print("Using model: " + model)
            continue
        if question == "/web on":
            web_enabled = True
            print("Web search is ON.")
            continue
        if question == "/web off":
            web_enabled = False
            print("Web search is OFF.")
            continue
        if question == "/cache":
            if cache is None:
                print("The answer cache is unavailable.")
            else:
                print(_cache_stats_line(cache, cache_enabled))
            continue
        if question == "/cache clear":
            if cache is None:
                print("The answer cache is unavailable.")
            else:
                try:
                    n = cache.clear()
                    print("Cleared " + str(n) + " cached answers.")
                except Exception as exc:
                    print("[could not clear the cache: " + str(exc) + "]")
            continue
        if question == "/cache on":
            if cache is None:
                print("The answer cache is unavailable, cannot enable it.")
            else:
                cache_enabled = True
                print("Answer cache is ON.")
            continue
        if question == "/cache off":
            cache_enabled = False
            print("Answer cache is OFF.")
            continue
        if question == "/deep" or question.startswith("/deep "):
            rest = question[len("/deep"):].strip()
            if not rest:
                print("Usage: /deep <question>  (re-ask with the "
                      + DEEP_MODEL + " model for a more careful, grounded answer)")
                continue
            question = rest
            deep_once = True
        elif question.startswith("/"):
            print("Unknown command. Type /help for the list.")
            continue

        # Per-question tier selection.
        q_model = DEEP_MODEL if deep_once else model

        # Start a run so the whole question renders like an ultracode pipeline.
        # When brain_ui is unavailable (ui is None), fall back to a no-op run so
        # the plain path still answers instead of crashing on run.* calls.
        run = ui.start_run(q_model) if ui is not None else _NullRun()

        # Fast path: serve from the answer cache before retrieval or a model call.
        # Deep (/deep) answers bypass the cache so they never collide with a fast draft.
        if cache_enabled and cache is not None and not deep_once:
            try:
                run.running("cache")
                hit = cache.lookup(question)
            except Exception:
                hit = None
            if hit:
                cached_answer = hit.get("answer", "") or ""
                age = str(hit.get("created", "?"))
                run.done("cache", "hit — served from cache (saved " + age + ")")
                for k in ("bm25", "expand", "model", "web"):
                    run.skip(k, "cache hit")
                sink = run.begin_answer()
                sink(cached_answer)
                run.sources(hit.get("sources") or [])
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": cached_answer})
                trim_history(history)
                run.finish()
                continue
            run.done("cache", "miss — computing fresh")
        elif run is not None:
            run.skip("cache", "cache disabled")

        try:
            answer, sources = handle_question(
                q_model, question, history, bm25, chunks, meta, web_enabled, run
            )
        except KeyboardInterrupt:
            if run is not None:
                run.finish()
            print("\n[interrupted, back to the prompt]")
            continue
        except Exception as exc:
            if run is not None:
                run.finish()
            print("[something went wrong handling that: " + str(exc) + "]")
            continue

        if run is not None:
            run.finish()

        # Store fresh answers in the cache (triggers the storage-optimization
        # pass and compounds the knowledge bank).
        if cache_enabled and cache is not None and answer and not deep_once:
            try:
                cache.store(question, answer, sources, model)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
