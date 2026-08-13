#!/usr/bin/env python3
"""brain_skills.py - a bounded skill loop for the jetson-brain INSUFFICIENT branch.

Today brain.py's phase 2 makes ONE routing call (FETCH/SEARCH), goes to the web,
and stops. It cannot retry retrieval with better wording, pull a named note whole,
or fall back on the model's own knowledge. Measured on the held-out set, most
vault misses are phrasing misses rather than content gaps (the retrieval eval
showed the gold chunk sitting just outside keep_k for a differently worded
query), so "search the vault again, worded better" is the highest-value recovery
and it is the first skill offered here.

WHY THIS FILE IS STANDALONE (no `import brain`): escalate.py:1-10 set the
discipline and it holds for the same two reasons. brain.py is edited constantly
and a top-level import would couple this module's importability to whatever
state that 2,900-line file is in; and the tests below must run with no Ollama,
no network, and no BM25 index, which only dependency injection allows. Everything
this module needs arrives in a SkillContext.

WHY NOT JSON TOOL CALLS: llama3.2:3b cannot reliably emit JSON. This codebase
already proved the working shape - a single-line prefix protocol with a
permissive fallback (brain.py ROUTE_SYSTEM at :1668 and its parse at :2556-2558,
where anything that is not an explicit FETCH becomes a plain SEARCH). The
protocol here is the same idea with five verbs:

    SKILL: search_vault | <reformulated query>
    SKILL: get_note     | <note title or alias>
    SKILL: web_search   | <query>
    SKILL: fetch_url    | <url>
    SKILL: reason

Exactly one line, exactly one decision per call. Anything unparseable becomes
web_search on the original question (or search_vault on it when the web is
off), which is the safe default action - the same "never let a parse failure
become a crash or a dead end" rule as the routing call.

WHY THE SELECTION CALL IS TINY: num_ctx is 6144 and Ollama truncates from the
TOP, which silently eats the system prompt and with it the grounding contract.
brain.py:2538-2541 deliberately drops the vault context from its routing call to
save ~1.5-2k prefill tokens; the same reasoning applies to every selection call
here, so the selection prompt carries only the menu plus hard-capped one-line
observations. The vault context is only ever sent on an actual grounded ANSWER
call, exactly once, and never accumulates.

PROVENANCE: an answer that did not come from the vault is labelled, always, in
the same footer style as the grounding check at brain.py:2510-2517. A reasoned
answer must never be presentable as vault-grounded.

Default OFF: set BRAIN_SKILLS=1 to enable. With it off, answer_with_skills()
returns an empty result without making a single call, so brain.py's behaviour is
byte-identical to today.
"""

import os
import re
import time
from dataclasses import dataclass, field

# ---- Enable flag + budgets -------------------------------------------------


def _flag(name, default="0"):
    # Same truthiness convention as escalate.py:49-50, so one mental model
    # covers every env switch in this repo.
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


SKILLS_ENABLED = _flag("BRAIN_SKILLS", "0")
SKILLS_VERBOSE = _flag("BRAIN_SKILLS_VERBOSE", "0")

# 3 steps, not 5: each step costs one selection call plus one answer call, and
# on the Jetson a 3B round trip is ~1-3s with a warm model. Past three steps the
# loop is spending more wall clock than the web path it is meant to beat, and
# the observation list starts crowding num_ctx.
MAX_STEPS = int(os.environ.get("BRAIN_SKILLS_MAX_STEPS", "3"))

# Wall clock, checked BETWEEN steps: once before the selection call and once
# after it, before dispatch (test_brain_skills.py:368-388 pins both). A step
# that has already been dispatched runs to completion - nothing here can abort a
# web fetch or a generation mid-flight - so the effective ceiling is DEADLINE_S
# plus one handler's cost, and a web_search handler is the expensive case.
# Deliberately not called a hard wall clock: bounding one server request's hold
# on the GPU lock belongs in brain.py's phase-2 web path and in the shim/web
# handlers, not in a loop that cannot see inside its own injected callables.
# The step cap alone is not a time bound: a cold 7B load plus a slow fetch can
# blow past any step budget.
DEADLINE_S = float(os.environ.get("BRAIN_SKILLS_DEADLINE", "45"))

# Observation caps, sized against num_ctx=6144. ~4 chars/token puts the whole
# context at roughly 24k chars; the selection prompt gets a small slice of it so
# the menu (the part that must survive top-truncation) is never at risk.
OBS_CHAR_CAP = 240        # per observation line
OBS_TOTAL_CAP = 1200      # all observations together

# A note pulled whole is the one place a single blob can be large. 3000 chars is
# about a fifth of the context, in line with what TOP_K_FAST=6 chunks occupy.
NOTE_CHAR_CAP = int(os.environ.get("BRAIN_SKILLS_NOTE_CHARS", "3000"))

# The whole point of a search_vault retry is a WIDER window than the phase-1
# keep_k (TOP_K_FAST=6 / MULTI_TOP_K): the reformulated query is a second
# opinion, and narrowing it to the same width would mostly re-fetch the same
# chunks under different wording.
WIDE_KEEP_K = int(os.environ.get("BRAIN_SKILLS_KEEP_K", "10"))

# Argument sanity cap: a 3B model sometimes trails a whole paragraph after the
# pipe. Truncating is friendlier than rejecting, since the head of the argument
# is nearly always the usable query.
ARG_CHAR_CAP = 400

# Generation options for the SELECTION call only. temperature 0, not the 0.2
# answer default in brain.GEN_OPTIONS: this call emits one label, and sampling
# noise on a label is pure loss (the same reason eval runs pin
# BRAIN_TEMPERATURE=0, brain.py:83-88). num_predict is tiny because the protocol
# is one line, which also caps how long a runaway reply can stall the loop.
# brain.py builds ctx.chat_select by merging this over GEN_OPTIONS; when it is
# not injected the loop degrades to ctx.chat rather than failing.
SELECT_GEN_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 48,
    "num_ctx": 6144,
}

# ---- Provenance labels -----------------------------------------------------
# Footer style copied from brain.py:2510-2517 ("[grounding check - ...]") so the
# terminal, the web UI, and the eval harness all render one visual convention.
# These are module constants precisely so tests (and any future UI styling) can
# assert on the exact text instead of a paraphrase.

LABEL_VAULT = ""   # a vault-grounded answer carries no extra label, as today.
LABEL_WEB_SEARCH = (
    "\n\n[provenance - not from the vault: answered from live web search "
    "results, treat as unverified]")
LABEL_WEB_FETCH = (
    "\n\n[provenance - not from the vault: answered from one web page fetched "
    "live, treat as unverified]")
LABEL_REASONED = (
    "\n\n[provenance - not from the vault: answered from the model's own "
    "general knowledge with no source, treat as unverified]")

PROVENANCE_LABELS = {
    "search_vault": LABEL_VAULT,
    "get_note": LABEL_VAULT,
    "web_search": LABEL_WEB_SEARCH,
    "fetch_url": LABEL_WEB_FETCH,
    "reason": LABEL_REASONED,
}

# ---- Protocol --------------------------------------------------------------

SKILL_NAMES = ("search_vault", "get_note", "web_search", "fetch_url", "reason")
WEB_SKILLS = ("web_search", "fetch_url")

# Preference order used when the model repeats itself: cheapest and most likely
# to actually help first. Mirrors the loop's whole thesis - re-retrieval beats
# the web, and the web beats unsourced recall.
ESCALATION_ORDER = ("search_vault", "web_search", "reason")

_MENU_LINES = {
    "search_vault": "SKILL: search_vault | <the same question worded differently for a vault search>",
    "get_note": "SKILL: get_note | <the exact title of one vault note>",
    "web_search": "SKILL: web_search | <a short web search query>",
    "fetch_url": "SKILL: fetch_url | <one full https url>",
    "reason": "SKILL: reason | answer from your own general knowledge",
}


def build_skill_system(web_enabled=True):
    """The selection prompt: the menu, one example line each, nothing else.

    Deliberately does NOT include the vault snippets, the grounding contract, or
    the conversation history. This call decides one word; every token spent
    beyond that is prefill the Jetson pays for on a 6144-token window."""
    names = [n for n in SKILL_NAMES if web_enabled or n not in WEB_SKILLS]
    lines = [
        "The local vault could not answer the user's question. Pick the ONE "
        "next action most likely to answer it.",
        "Reply with EXACTLY ONE line and nothing else, in this form:",
    ]
    lines.extend(_MENU_LINES[n] for n in names)
    lines.append(
        "Prefer search_vault first when the question sounds like it belongs to "
        "a personal research vault about deafness, sign language, or education "
        "and the earlier wording may simply have missed.")
    lines.append("Output only that one line. No explanation, no extra lines.")
    return "\n".join(lines)


# Smart quotes and dashes a chat model emits instead of ASCII. Normalized before
# parsing so 'SKILL: web_search | "who is X"' parses the same as the plain form.
_SMART = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "｜": "|", "│": "|",
}

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_LABEL_RE = re.compile(r"^\s*(?:skill|action|tool)\s*[:\-]\s*", re.IGNORECASE)


def _clean_line(line):
    for bad, good in _SMART.items():
        line = line.replace(bad, good)
    line = _FENCE_RE.sub("", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = _BULLET_RE.sub("", line)
    return line.strip().strip("\"'").strip()


def _norm_name(token):
    """'Search Vault', 'search-vault', 'SEARCH_VAULT' -> 'search_vault'."""
    t = re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")
    return t


def _parse_one_line(line):
    """(skill, arg, piped) from one candidate line, or None when it names no skill.

    Permissive on purpose: the 'SKILL:' label is optional, the pipe may have any
    spacing or be missing entirely, and trailing prose after the argument is
    kept (and later truncated) rather than rejected. `piped` records whether the
    line actually carried a '|' so parse_skill can prefer a well-formed line
    over a chatty one that happened to come first."""
    line = _clean_line(line)
    if not line:
        return None
    line = _LABEL_RE.sub("", line).strip()
    if not line:
        return None

    head, sep, tail = line.partition("|")
    name = _norm_name(head)
    if name not in SKILL_NAMES:
        # No pipe and extra words ("use search_vault now"): find the skill name
        # anywhere in the head, which is how a chattier reply degrades.
        found = None
        for cand in SKILL_NAMES:
            if cand == "reason":
                # `reason` is the only name with no underscore, so this loose
                # scan would degenerate to \breason\b over arbitrary prose and
                # route any chatty reply containing the bare English word to
                # unsourced recall ("The main reason was ..."). Measured on the
                # 3B's real replies this fired far more often on prose than on
                # an actual pick. It stays reachable through the exact
                # head-token path above (_norm_name), the _next_untried
                # substitution, and the web-off fallback.
                continue
            if re.search(r"\b" + cand.replace("_", "[ _-]?") + r"\b", head,
                         re.IGNORECASE):
                if found is not None and found != cand:
                    return None    # two skills in one line: ambiguous
                found = cand
        if found is None:
            return None
        name = found
        if not sep:
            pre, tail = re.split(r"\b" + name.replace("_", "[ _-]?") + r"\b",
                                 head, maxsplit=1, flags=re.IGNORECASE)[:2]
            # Only trust the tail when the line STARTS with the skill name
            # ("search_vault dpn 1988", "fetch_url https://..."). When the name
            # is buried in prose ("I will use the search_vault skill to ...")
            # the tail is narration, not an argument - taking it sent BM25 and
            # DuckDuckGo the string "skill to find information about this."
            # marked parsed=True. Rejected alternative: a stopword list of
            # narration verbs; position is exact where a word list is a guess.
            if pre.strip():
                tail = ""
    arg = tail.strip().strip("\"'").strip()
    # Strip a restated placeholder ("<a web search query>") - the model echoing
    # the menu's angle brackets means it gave no real argument.
    if arg.startswith("<") and arg.endswith(">"):
        arg = ""
    return name, arg[:ARG_CHAR_CAP].strip(), bool(sep)


def parse_skill(text, question, web_enabled=True):
    """Parse a selection reply into (skill, arg, parsed_ok).

    Fallback rule, mirroring brain.py:2556-2558 where anything that is not an
    explicit FETCH becomes a SEARCH: anything unparseable becomes web_search on
    the ORIGINAL question when the web is on, and search_vault on it when the
    web is off. A bad parse is never an error and never a dead end."""
    # Web off: search_vault, not reason. ESCALATION_ORDER[0] is search_vault and
    # the module's whole thesis is that re-retrieval beats unsourced recall, so
    # a non-decision on an offline turn must not land on the one skill that
    # produces no sources. reason stays reachable through _next_untried once
    # search_vault is spent.
    fallback = ("web_search", question, False) if web_enabled \
        else ("search_vault", question, False)
    if not text:
        return fallback

    candidates = []
    for raw in text.splitlines():
        got = _parse_one_line(raw)
        if got:
            candidates.append(got)
    if not candidates:
        return fallback

    distinct = {name for name, _, _ in candidates}
    if len(distinct) >= 3:
        # The model restated the whole menu instead of choosing. That is a
        # non-decision, so it takes the safe default rather than the first item
        # (which would always be search_vault). The loop's own repeat guard is
        # what prevents spinning; this rule is only about not mistaking a menu
        # echo for a choice.
        return fallback

    # Prefer the first line that actually carried a pipe. A 3B reply often
    # narrates first and complies second ("I will use the search_vault skill.\n
    # SKILL: search_vault | deaf president now 1988"); taking candidates[0]
    # threw away the good argument on line 2.
    piped = [c for c in candidates if c[2]]
    name, arg, _sep = (piped or candidates)[0]
    if not web_enabled and name in WEB_SKILLS:
        return fallback
    if name != "reason" and not arg:
        # A bare skill name with no argument is still a real choice; the
        # original question is the obvious argument.
        arg = question
    return name, arg, True


# ---- Context + result ------------------------------------------------------


@dataclass
class SkillContext:
    """Everything the loop needs, injected by brain.py at the INSUFFICIENT branch.

    Every field is optional so a test (or a degraded runtime) can stub exactly
    the parts it exercises; a skill whose dependency is missing simply reports
    that as an observation and the loop picks something else.

    chat(model, messages) -> str            brain.chat_once
    chat_select(model, messages) -> str     same, but temperature 0 (see below)
    retrieve(query, keep_k) -> [chunk]      wraps brain.retrieve_context; brain
                                            passes order_full sliced to keep_k so
                                            the retry really is wider
    gather_web(query) -> [(url, excerpt)]   brain.gather_web_context
    fetch_page(url) -> str                  brain.fetch_page
    meta                                    brain's index meta (by_source, by_id,
                                            link_map)
    vault_system / web_system / doc_system  brain's prompts, passed VERBATIM so
                                            the INSUFFICIENT contract that
                                            _is_insufficient parses is preserved
    is_insufficient(text) -> bool           brain._is_insufficient
    format_vault_context(chunks, ids) -> str brain.format_vault_context
    vault_sources(chunks) -> [str]          brain.vault_sources
    clock() -> float                        monotonic seconds, injectable so the
                                            deadline is testable without sleeping
    prev_turn                               PRE-RENDERED text of the last
                                            exchange, hard-capped by brain.py.
                                            A plain string, not a message list,
                                            so it can never become extra chat
                                            turns that displace the system
                                            prompt in a 6144-token window.
    """
    model: str = "llama3.2:3b"
    chat: object = None
    chat_select: object = None
    retrieve: object = None
    gather_web: object = None
    fetch_page: object = None
    meta: dict = field(default_factory=dict)
    web_enabled: bool = True
    vault_system: str = ""
    web_system: str = ""
    doc_system: str = ""
    reason_system: str = ""
    is_insufficient: object = None
    format_vault_context: object = None
    vault_sources: object = None
    clock: object = None
    prev_turn: str = ""
    max_steps: int = 0
    deadline_s: float = 0.0


@dataclass
class SkillResult:
    """Never an exception. `answer == ""` means "no answer" and brain.py falls
    through to its existing web path exactly as if this module were off."""
    answer: str = ""
    sources: list = field(default_factory=list)
    skill: str = ""
    provenance: str = ""
    steps: int = 0
    reason: str = "disabled"
    trace: list = field(default_factory=list)

    @property
    def ok(self):
        return bool(self.answer)


# Used only when the caller does not inject a grounding contract (tests, or a
# partially wired runtime). Production ALWAYS passes VAULT_FIRST_SYSTEM, because
# _is_insufficient, faithfulness_flags and chat_stream_holdback all parse the
# model's text against that exact contract.
FALLBACK_VAULT_SYSTEM = (
    "Answer the question USING ONLY the vault snippets provided, and cite the "
    "sources by filename. If the snippets do not contain enough information, "
    "reply with EXACTLY the single word INSUFFICIENT and nothing else."
)
FALLBACK_WEB_SYSTEM = (
    "Answer the question using the web excerpts provided and cite the URLs.")
FALLBACK_DOC_SYSTEM = (
    "Answer the question from the fetched reference page and cite its URL.")
DEFAULT_REASON_SYSTEM = (
    "Answer the question from your own general knowledge, briefly and plainly. "
    "You have no sources for this answer, so state any uncertainty outright and "
    "never invent a citation, a URL, a statute, a date, or a statistic."
)


def _log(msg):
    if SKILLS_VERBOSE:
        try:
            print("[skills] " + msg, flush=True)
        except Exception:
            pass


def _default_insufficient(text):
    # Only the exact sentinel: the rich refusal list lives in brain.py and is
    # injected. This bare version exists so the module is importable and
    # testable on its own.
    t = re.sub(r"\s+", " ", (text or "").strip()).strip(".!?\"'` ").lower()
    return (not t) or t.startswith("insufficient")


def _default_format_context(chunks, linked_ids=None):
    blocks = []
    for c in chunks or []:
        src = c.get("source") or c.get("title") or "unknown"
        blocks.append("[source: " + src + "]\n" + (c.get("text") or ""))
    return "\n\n".join(blocks) if blocks else "(no relevant vault snippets were found)"


def _default_sources(chunks):
    out = []
    for c in chunks or []:
        src = c.get("source") or c.get("title")
        if src and src not in out:
            out.append(src)
    return out


# ---- get_note --------------------------------------------------------------


def _normalize_key(key):
    # Same normalization build_index.normalize_key uses for link_map keys
    # (build_index.py:433-435), so an alias written for wikilink resolution
    # resolves here too. Duplicated rather than imported: this module stays
    # standalone, and the rule is three lines.
    return re.sub(r"\s+", " ", (key or "").strip().lower())


def resolve_note(meta, title):
    """Vault source path for a note named by title, path, or YAML alias.

    brain.py has no such lookup today - expand_with_links resolves link TARGETS
    but nothing resolves a user-facing name. Built from what the index already
    carries: link_map (normalized stem/relpath/alias -> source, build_index.py
    :874-884) first, then a case-insensitive filename-stem scan over by_source
    as the backstop for notes with no link_map entry. Returns '' when nothing
    matches. Never raises."""
    try:
        name = (title or "").strip().strip("[]").strip()
        name = re.sub(r"\.md$", "", name, flags=re.IGNORECASE)
        if not name:
            return ""
        key = _normalize_key(name)
        link_map = meta.get("link_map") or {}
        if key in link_map:
            return link_map[key]
        by_source = meta.get("by_source") or {}
        base = key.split("/")[-1]
        for src in by_source:
            stem = re.sub(r"\.md$", "", str(src).split("/")[-1],
                          flags=re.IGNORECASE)
            if _normalize_key(stem) == base:
                return src
        # Last resort: a unique substring match, so "Deaf President Now" finds
        # "History/Deaf President Now 1988.md". Ambiguous matches are refused
        # rather than guessed.
        hits = [s for s in by_source if base in _normalize_key(str(s))]
        if len(hits) == 1:
            return hits[0]
    except Exception:
        return ""
    return ""


def note_text(meta, source, cap=NOTE_CHAR_CAP):
    """The full note, chunks re-joined in index order and hard-capped. Never
    raises; returns '' when the note has no retrievable chunks."""
    try:
        by_source = meta.get("by_source") or {}
        by_id = meta.get("by_id") or {}
        parts = []
        for cid in by_source.get(source) or []:
            chunk = by_id.get(cid)
            if chunk:
                parts.append(chunk.get("text") or "")
        text = "\n\n".join(p for p in parts if p)
        return text[:cap]
    except Exception:
        return ""


# ---- The loop --------------------------------------------------------------


def _observations_block(observations):
    """Newest-first observations, trimmed to OBS_TOTAL_CAP.

    Trims the OLDEST entries, not the newest: the model is choosing what to do
    next, and the most recent failure is the most informative constraint on that
    choice. Unbounded growth here is exactly how a loop silently pushes the
    system prompt out of a 6144-token window."""
    kept, total = [], 0
    for obs in reversed(observations):
        line = "- " + obs[:OBS_CHAR_CAP]
        if total + len(line) > OBS_TOTAL_CAP:
            break
        kept.append(line)
        total += len(line)
    kept.reverse()
    return "\n".join(kept)


def _select(ctx, question, observations, web_enabled):
    """One selection call. temperature 0, not the 0.2 answer default: this call
    picks a label, and sampling noise on a label is pure loss - the same reason
    eval runs pin BRAIN_TEMPERATURE=0 (brain.py:83-88)."""
    system = build_skill_system(web_enabled)
    # The prior exchange goes on the SELECTION call only. A follow-up ("what
    # about her other paper?") is unroutable without an antecedent, and this
    # call is where the antecedent pays off: it produces the reformulated
    # search_vault/web_search query, after which the retrieved chunks
    # disambiguate the pronoun for the answer call by themselves. Rejected:
    # adding history to _answer_call too - that risks top-truncating the
    # injected grounding contract at WIDE_KEEP_K=10 chunks and pollutes the
    # sourceless reason path.
    user = ""
    if ctx.prev_turn:
        user += "Earlier in this conversation:\n" + ctx.prev_turn + "\n\n"
    user += "Question: " + question
    if observations:
        user += ("\n\nAlready tried (do not repeat these):\n"
                 + _observations_block(observations))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    fn = ctx.chat_select or ctx.chat
    if fn is None:
        return ""
    try:
        text = fn(ctx.model, messages) or ""
    except Exception as exc:
        _log("selection call failed: " + str(exc))
        return ""
    if text.startswith("[error]"):
        # chat_once's failure convention (brain.py:1358). Treated as an
        # unparseable reply so the permissive fallback applies.
        return ""
    return text


def _answer_call(ctx, system, user):
    """One grounded answer call. Returns '' on any failure."""
    if ctx.chat is None:
        return ""
    try:
        text = (ctx.chat(ctx.model,
                         [{"role": "system", "content": system},
                          {"role": "user", "content": user}]) or "").strip()
    except Exception as exc:
        _log("answer call failed: " + str(exc))
        return ""
    return "" if text.startswith("[error]") else text


def _try_vault_answer(ctx, question, chunks, label):
    """Grounded answer over `chunks` using the INJECTED contract verbatim.

    Returns (answer, sources) or ('', []) when the model still says
    INSUFFICIENT. The contract is passed through untouched so the same
    _is_insufficient that guards phase 1 guards this retry."""
    fmt = ctx.format_vault_context or _default_format_context
    srcs = ctx.vault_sources or _default_sources
    insuff = ctx.is_insufficient or _default_insufficient
    system = ctx.vault_system or FALLBACK_VAULT_SYSTEM
    user = "Vault context:\n" + fmt(chunks, None) + "\n\nQuestion: " + question
    text = _answer_call(ctx, system, user)
    if not text or insuff(text):
        return "", []
    _log(label + " answered")
    return text, srcs(chunks)


def _do_search_vault(ctx, question, arg):
    if ctx.retrieve is None:
        return None, "search_vault: retrieval is not available"
    query = arg or question
    try:
        chunks = ctx.retrieve(query, WIDE_KEEP_K) or []
    except Exception as exc:
        return None, "search_vault '%s': retrieval failed (%s)" % (query, exc)
    if not chunks:
        return None, "search_vault '%s': no chunks matched" % query
    answer, sources = _try_vault_answer(ctx, question, chunks, "search_vault")
    if not answer:
        srcs = (ctx.vault_sources or _default_sources)(chunks)
        return None, ("search_vault '%s': %d chunks (%s) still did not answer"
                      % (query, len(chunks), ", ".join(srcs[:3])))
    return (answer, sources, "search_vault"), None


def _do_get_note(ctx, question, arg):
    source = resolve_note(ctx.meta or {}, arg)
    if not source:
        return None, "get_note '%s': no vault note by that name" % arg
    text = note_text(ctx.meta or {}, source)
    if not text:
        return None, "get_note '%s': note '%s' has no text" % (arg, source)
    chunk = {"source": source, "text": text}
    answer, sources = _try_vault_answer(ctx, question, [chunk], "get_note")
    if not answer:
        return None, "get_note '%s': note '%s' did not answer it" % (arg, source)
    return (answer, sources, "get_note"), None


def _do_web_search(ctx, question, arg):
    if ctx.gather_web is None:
        return None, "web_search: web access is not available"
    query = arg or question
    try:
        excerpts = ctx.gather_web(query) or []
    except Exception as exc:
        return None, "web_search '%s': failed (%s)" % (query, exc)
    if not excerpts:
        return None, "web_search '%s': no usable results" % query
    blocks = "\n\n".join("[url: %s]\n%s" % (u, t) for u, t in excerpts)
    user = "Web excerpts:\n" + blocks + "\n\nQuestion: " + question
    text = _answer_call(ctx, ctx.web_system or FALLBACK_WEB_SYSTEM, user)
    if not text:
        return None, "web_search '%s': the model could not use the results" % query
    return (text, [u for u, _ in excerpts], "web_search"), None


def _do_fetch_url(ctx, question, arg):
    if ctx.fetch_page is None:
        return None, "fetch_url: web access is not available"
    url = normalize_url(arg)
    if not url:
        return None, "fetch_url: '%s' is not a usable url" % arg
    try:
        page = ctx.fetch_page(url) or ""
    except Exception as exc:
        return None, "fetch_url %s: failed (%s)" % (url, exc)
    if not page:
        return None, "fetch_url %s: page was empty or unreachable" % url
    user = ("Reference page (" + url + "):\n" + page
            + "\n\nQuestion: " + question)
    text = _answer_call(ctx, ctx.doc_system or FALLBACK_DOC_SYSTEM, user)
    if not text:
        return None, "fetch_url %s: the model could not use the page" % url
    return (text, [url], "fetch_url"), None


def _do_reason(ctx, question, arg):
    system = ctx.reason_system or DEFAULT_REASON_SYSTEM
    text = _answer_call(ctx, system, "Question: " + question)
    if not text:
        return None, "reason: the model produced nothing"
    return (text, [], "reason"), None


_DISPATCH = {
    "search_vault": _do_search_vault,
    "get_note": _do_get_note,
    "web_search": _do_web_search,
    "fetch_url": _do_fetch_url,
    "reason": _do_reason,
}


def normalize_url(raw):
    """Same shape as brain.normalize_url (brain.py:1889-1909), duplicated to
    keep this module standalone: first token, scheme added when missing, and a
    host.tld sanity check so a sentence never becomes a fetch."""
    raw = (raw or "").strip().strip("<>\"'")
    if not raw:
        return ""
    url = raw.split()[0].strip("<>\"'").rstrip(".,;:!?)]}")
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    if not re.search(r"[a-z0-9][a-z0-9\-]*\.[a-z]{2,}", url, re.IGNORECASE):
        return ""
    return url


def _next_untried(tried_skills, web_enabled):
    """First skill in ESCALATION_ORDER not yet attempted, or ''."""
    for name in ESCALATION_ORDER:
        if not web_enabled and name in WEB_SKILLS:
            continue
        if name not in tried_skills:
            return name
    return ""


def answer_with_skills(ctx, question):
    """THE entry point. Call it at brain.py's INSUFFICIENT branch.

    Returns a SkillResult. `result.ok` false means "no answer, carry on" - the
    caller must fall through to its existing phase-2 web path unchanged. This
    function never raises: every failure inside degrades to a no-answer result,
    because a skill loop that can break the main answer path is worse than no
    skill loop at all."""
    result = SkillResult()
    if not SKILLS_ENABLED:
        return result
    try:
        return _run_loop(ctx, question)
    except Exception as exc:
        _log("loop aborted: " + str(exc))
        return SkillResult(reason="error:" + str(exc)[:80])


def _run_loop(ctx, question):
    clock = ctx.clock or time.monotonic
    started = clock()
    max_steps = ctx.max_steps or MAX_STEPS
    deadline = ctx.deadline_s or DEADLINE_S
    web_enabled = bool(ctx.web_enabled)

    observations = []
    trace = []
    attempted = set()        # (skill, normalized arg) - exact repeats
    tried_skills = set()      # skill names, for the repeat substitution
    steps = 0

    while steps < max_steps:
        if clock() - started >= deadline:
            return SkillResult(steps=steps, reason="deadline", trace=trace)

        raw = _select(ctx, question, observations, web_enabled)
        skill, arg, parsed = parse_skill(raw, question, web_enabled)

        key = (skill, _normalize_key(arg))
        if key in attempted:
            # Refuse an identical repeat. Substituting the next untried skill
            # beats burning a step on a no-op, and beats looping forever on the
            # model's favourite verb.
            sub = _next_untried(tried_skills, web_enabled)
            if not sub:
                return SkillResult(steps=steps, reason="exhausted", trace=trace)
            skill, arg = sub, question
            key = (skill, _normalize_key(arg))
            if key in attempted:
                return SkillResult(steps=steps, reason="exhausted", trace=trace)
        attempted.add(key)
        tried_skills.add(skill)
        steps += 1
        trace.append({"step": steps, "skill": skill, "arg": arg,
                      "parsed": parsed})
        _log("step %d: %s | %s (parsed=%s)" % (steps, skill, arg, parsed))

        if clock() - started >= deadline:
            return SkillResult(steps=steps, reason="deadline", trace=trace)

        handler = _DISPATCH.get(skill)
        if handler is None:
            observations.append("unknown skill '%s'" % skill)
            continue
        try:
            done, observation = handler(ctx, question, arg)
        except Exception as exc:
            done, observation = None, "%s: crashed (%s)" % (skill, exc)
        if done is None:
            observations.append(observation or (skill + ": no result"))
            trace[-1]["observation"] = observations[-1]
            continue

        answer, sources, used = done
        label = PROVENANCE_LABELS.get(used, LABEL_REASONED)
        if label:
            answer = answer.rstrip() + label
        trace[-1]["answered"] = True
        return SkillResult(answer=answer, sources=sources, skill=used,
                           provenance=label, steps=steps, reason="answered",
                           trace=trace)

    return SkillResult(steps=steps, reason="step_cap", trace=trace)


# ---- CLI (parse-only; no Ollama, no index) ---------------------------------

if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python3 brain_skills.py --parse '<a model reply>' "
              "[--no-web]")
        print("       python3 brain_skills.py --menu [--no-web]")
        print("enable in brain.py with BRAIN_SKILLS=1")
        raise SystemExit(0)

    web = "--no-web" not in argv
    argv = [a for a in argv if a != "--no-web"]

    if argv[0] == "--menu":
        print(build_skill_system(web))
        raise SystemExit(0)

    if argv[0] == "--parse":
        reply = " ".join(argv[1:])
        skill, arg, parsed = parse_skill(reply, "the original question", web)
        print("skill=%s parsed=%s" % (skill, parsed))
        print("arg=%r" % arg)
        print("provenance=%r" % PROVENANCE_LABELS.get(skill, ""))
        raise SystemExit(0)

    print("unrecognized arguments; try --help")
    raise SystemExit(1)
