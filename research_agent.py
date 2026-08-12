#!/usr/bin/env python3
"""research_agent.py - the scheduled deaf-education literature agent.

One run: discover candidate papers (research_sources), drop everything the
brain already knows (state, queue, vault DOIs/citekeys), gate on peer review,
verify full-text access (research_fetch), score relevance with the resident
llama3.2:3b, and DRAFT notes into the review queue. Nothing this agent writes
is ever indexed directly: a human approves each draft with research_review.py
before it enters vault/_Research/ (the curated-growth rule).

State lives OUTSIDE the vault so deploy.sh rsync can never delete it:

    /mnt/nvme/brain/research/
        state.json          last_run per source, seen/rejected keys,
                            cached OpenAlex author ids, Clerc URLs seen
        queue/<stem>.json   full candidate record (the machine half)
        queue/<stem>.md     draft note awaiting review (the human half)
        decisions.jsonl     append-only audit log (research_review.py writes)

Failure tolerance: each source sweep and each candidate is individually
guarded, state is written atomically after discovery and again at the end,
and an unreachable Ollama just queues candidates unscored (score null); the
human gate still works. Runs weekly from research-agent.timer (Mon 04:00),
offset from brain-learn.timer (Sun 03:30) so they never contend for Ollama.

    python3 research_agent.py --dry-run --limit 5 --since 2024-01-01
    python3 research_agent.py --limit 25
    python3 research_agent.py --source eric

Python 3.9 compatible. Standard library plus the sibling modules. No em dashes.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
import datetime
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import research_sources as rs
import research_fetch as rf
import research_note as rn
from web_cache import _atomic_write_text

BRAIN_ROOT = os.environ.get("BRAIN_ROOT", "/mnt/nvme/brain")
RESEARCH_DIR = os.environ.get("RESEARCH_DIR",
                              os.path.join(BRAIN_ROOT, "research"))
QUEUE_DIR = os.path.join(RESEARCH_DIR, "queue")
STATE_PATH = os.path.join(RESEARCH_DIR, "state.json")
VAULT_DIR = os.environ.get("BRAIN_VAULT", os.path.join(BRAIN_ROOT, "vault"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
LLM_MODEL = "llama3.2:3b"        # resident fast model; the 7B would evict it
LLM_TIMEOUT = 120
SCORE_THRESHOLD = 4              # deliberately low: the human gate is the filter
DEFAULT_SINCE = "2015-01-01"
OVERLAP_DAYS = 7                 # re-sweep window absorbs indexing lag
DEFAULT_LIMIT = 25               # keeps the review queue human-sized

SOURCES = ("openalex", "eric", "clerc")

SCORE_SYSTEM = (
    "You screen research papers for a thesis library on language deprivation, "
    "sign language acquisition, and language support for deaf children. "
    "Output only JSON.")
SCORE_TEMPLATE = (
    "Rate how relevant this paper is to language support for deaf children "
    "(language deprivation, ASL/sign language acquisition, early intervention, "
    "bilingual-bimodal deaf education, family decision-making). 0 = unrelated, "
    "10 = squarely on topic.\n\nTITLE: {title}\nVENUE: {venue}\n"
    "ABSTRACT: {abstract}\n\nRespond with ONLY this JSON: "
    '{{"score": <0-10>, "reason": "<one short sentence>"}}')
NOTE_TEMPLATE = (
    "In 3 to 5 plain sentences, say what this paper found and why it matters "
    "for language support for deaf children. Use only the abstract below; do "
    "not invent findings.\n\nTITLE: {title}\nABSTRACT: {abstract}\n\n"
    'Respond with ONLY this JSON: {{"note": "<3-5 sentences>"}}')


def _log(msg):
    print("[research-agent] " + msg, flush=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        if not isinstance(state, dict):
            state = {}
    except Exception:
        state = {}
    state.setdefault("last_run", {})
    state.setdefault("seen", [])
    state.setdefault("rejected", [])
    state.setdefault("discards", [])
    state.setdefault("author_ids", {})
    state.setdefault("clerc_seen_urls", [])
    return state


def save_state(state):
    try:
        data = json.dumps(state, ensure_ascii=False, indent=1)
    except Exception:
        return False
    return _atomic_write_text(STATE_PATH, data)


def since_for(state, source, cli_since):
    """The from-date for one source's sweep: the CLI override, else the
    stored last_run minus the overlap window, else the backfill default."""
    if cli_since:
        return cli_since
    last = (state.get("last_run") or {}).get(source)
    if not last:
        return DEFAULT_SINCE
    try:
        d = datetime.date.fromisoformat(last[:10])
        return (d - datetime.timedelta(days=OVERLAP_DAYS)).isoformat()
    except Exception:
        return DEFAULT_SINCE


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
_DOI_RE_FM = None   # compiled lazily


def vault_dedup_keys(vault_dir=VAULT_DIR):
    """Scan every vault note's leading frontmatter block for doi: and
    citekey: fields. Returns a set of 'doi:...'/'citekey:...' keys. Cheap
    (first 60 lines per file) and run once per agent run."""
    import re
    keys = set()
    doi_re = re.compile(r"^doi:\s*[\"']?([^\"'\s]+)", re.I)
    cite_re = re.compile(r"^citekey:\s*[\"']?([^\"'\s]+)", re.I)
    for root, dirs, files in os.walk(vault_dir):
        dirs[:] = [d for d in dirs if d not in
                   (".obsidian", ".git", ".trash", "node_modules")]
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8",
                          errors="ignore") as fh:
                    head = [next(fh, "") for _ in range(60)]
            except Exception:
                continue
            if not head or head[0].strip() != "---":
                continue
            for line in head[1:]:
                if line.strip() == "---":
                    break
                m = doi_re.match(line.strip())
                if m:
                    doi = rs.normalize_doi(m.group(1))
                    if doi:
                        keys.add("doi:" + doi)
                m = cite_re.match(line.strip())
                if m:
                    keys.add("citekey:" + m.group(1).strip().lower())
    return keys


def queue_dedup_keys(queue_dir=QUEUE_DIR):
    """Keys of every candidate already drafted and awaiting review."""
    keys = set()
    try:
        for name in os.listdir(queue_dir):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(queue_dir, name), "r",
                          encoding="utf-8") as fh:
                    cand = json.load(fh)
                keys.update(all_keys(cand))
            except Exception:
                continue
    except Exception:
        pass
    return keys


def _mark_clerc_seen(state, cand):
    # Consumption-time marking (queued/rejected/deduped); never at discovery.
    if cand.get("discovery_source") == "clerc" and cand.get("landing_url"):
        state["clerc_seen_urls"].append(cand["landing_url"])


def all_keys(cand):
    """candidate_keys plus the derived citekey (catches the same paper found
    through different ids). The derived citekey only joins when the candidate
    has real author+year metadata: an 'anon_nd_<two words>' key would merge
    distinct grey-lit pages whose titles share their first two words, and the
    strong identifiers (doi/openalex/eric/url) already cover those."""
    keys = list(rs.candidate_keys(cand))
    if cand.get("authors") and cand.get("year"):
        keys.append("citekey:" + rn.make_citekey(cand))
    return keys


def queue_stem(cand):
    """Deterministic queue filename stem: citekey plus a hash of the
    strongest identifier, so refinds overwrite in place (web_cache idiom)."""
    ident = (rs.normalize_doi(cand.get("doi")) or cand.get("openalex_id")
             or cand.get("eric_id") or cand.get("landing_url") or
             cand.get("title") or "unknown")
    h = hashlib.sha1(str(ident).encode("utf-8", "replace")).hexdigest()[:12]
    return "%s-%s" % (rn.make_citekey(cand), h)


# ---------------------------------------------------------------------------
# LLM scoring (best effort; failure queues the candidate unscored)
# ---------------------------------------------------------------------------
def _ollama_json(prompt, system=SCORE_SYSTEM):
    """One JSON-mode chat call to the resident 3B. Returns a dict or None."""
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False, "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 2048, "num_predict": 200},
    }
    try:
        req = urllib.request.Request(
            OLLAMA_HOST + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = (payload.get("message") or {}).get("content") or ""
        return json.loads(content)
    except Exception:
        return None


def llm_screen(cand):
    """Set cand['score'] (int or None), cand['score_reason'], and fill
    cand['relevance_note'] from the abstract. Never raises."""
    abstract = (cand.get("abstract") or "")[:1500]
    title = (cand.get("title") or "")[:300]
    obj = _ollama_json(SCORE_TEMPLATE.format(
        title=title, venue=(cand.get("venue") or "")[:120], abstract=abstract))
    score = None
    reason = ""
    if isinstance(obj, dict):
        try:
            score = max(0, min(10, int(obj.get("score"))))
            reason = str(obj.get("reason") or "").strip()[:300]
        except Exception:
            score = None
    cand["score"] = score
    cand["score_reason"] = reason
    if abstract and score is not None and score >= SCORE_THRESHOLD:
        obj = _ollama_json(NOTE_TEMPLATE.format(title=title, abstract=abstract))
        if isinstance(obj, dict) and str(obj.get("note") or "").strip():
            cand["relevance_note"] = str(obj["note"]).strip()[:1200]
    return cand


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def discover(state, sources, cli_since):
    """Run the selected sweeps. Returns (candidates, ran_sources)."""
    cands = []
    ran = []
    if "openalex" in sources:
        # Resolve any focus author still missing from the cache each run, so
        # one transient lookup failure on the first run does not silently
        # disable that scholar's sweep forever.
        missing = [n for n in rs.FOCUS_AUTHOR_NAMES
                   if n not in state["author_ids"]]
        if missing:
            _log("resolving focus-author OpenAlex ids (%d missing)" % len(missing))
            state["author_ids"].update(rs.resolve_author_ids(missing))
        since = since_for(state, "openalex", cli_since)
        _log("openalex sweep since %s" % since)
        got = rs.openalex_candidates(since, state["author_ids"])
        _log("openalex: %d raw candidates" % len(got))
        cands.extend(got)
        ran.append("openalex")
    if "eric" in sources:
        since = since_for(state, "eric", cli_since)
        _log("eric sweep since %s" % since[:4])
        got = rs.eric_candidates(int(since[:4]))
        _log("eric: %d raw candidates" % len(got))
        cands.extend(got)
        ran.append("eric")
    if "clerc" in sources:
        got = rs.clerc_candidates(state.get("clerc_seen_urls"))
        if got is None:
            _log("clerc sweep skipped (curl_cffi not installed)")
        else:
            # Links are marked seen only when a candidate is actually queued,
            # rejected, or deduped in run() -- never at discovery -- so grey
            # lit that falls past --limit is rediscovered next run instead of
            # being lost forever.
            _log("clerc: %d new listing links" % len(got))
            cands.extend(got)
            ran.append("clerc")
    return cands, ran


def run(sources, cli_since, limit, dry_run=False, use_llm=True):
    os.makedirs(QUEUE_DIR, exist_ok=True)
    state = load_state()
    started = datetime.datetime.now().isoformat(timespec="seconds")

    candidates, ran_sources = discover(state, sources, cli_since)

    # ---- Dedup: state, queue, vault, and within this run -----------------
    known = set(state["seen"]) | set(state["rejected"])
    known |= queue_dedup_keys()
    known |= vault_dedup_keys()
    fresh, in_run = [], set()
    n_dup = 0
    for cand in candidates:
        keys = all_keys(cand)
        if any(k in known for k in keys) or any(k in in_run for k in keys):
            n_dup += 1
            _mark_clerc_seen(state, cand)
            continue
        in_run.update(keys)
        fresh.append(cand)

    # ---- Peer-review gate ------------------------------------------------
    gated = []
    n_gate = 0
    for cand in fresh:
        if cand.get("peer_reviewed") or cand.get("access") == "grey-lit":
            gated.append(cand)
        else:
            n_gate += 1
    # Grey lit reviews last; higher-signal scholarly work first.
    gated.sort(key=lambda c: (c.get("access") == "grey-lit",
                              -(c.get("year") or 0)))

    capped = gated[:max(1, limit)]
    _log("candidates: %d raw, %d deduped away, %d failed the peer-review "
         "gate, %d fresh, %d after --limit"
         % (len(candidates), n_dup, n_gate, len(gated), len(capped)))

    if dry_run:
        for cand in capped:
            reason = rf.dead_domain_reason(cand.get("fulltext_url") or "")
            print("  [%s] %s (%s) %s%s" % (
                cand.get("discovery_source"), cand.get("title")[:70],
                cand.get("year"), "doi:" + cand["doi"] if cand.get("doi") else
                cand.get("landing_url", "")[:60],
                ("  [dead domain: %s]" % reason) if reason else ""))
        _log("dry run: nothing written")
        return 0

    # ---- Fetch, score, queue --------------------------------------------
    n_queued = n_fulltext = n_discard = n_err = 0
    for cand in capped:
        try:
            fetched = {"body": "", "via": "", "note": "not attempted"}
            if cand.get("access") != "grey-lit":
                fetched = rf.fetch_fulltext(cand)
                if fetched["body"]:
                    n_fulltext += 1
                elif cand.get("access") == "open":
                    cand["access"] = "abstract-only"   # the OA claim failed
                cand["fetch_note"] = fetched["note"]
            if use_llm:
                llm_screen(cand)
            score = cand.get("score")
            if score is not None and score < SCORE_THRESHOLD:
                state["discards"].append({
                    "keys": all_keys(cand), "title": cand.get("title"),
                    "score": score, "reason": cand.get("score_reason"),
                    "ts": started})
                state["rejected"].extend(all_keys(cand))
                _mark_clerc_seen(state, cand)
                n_discard += 1
                continue
            stem = queue_stem(cand)
            cand["queue_stem"] = stem
            cand["fulltext"] = fetched["body"]
            cand["fulltext_via"] = fetched["via"]
            ok_json = _atomic_write_text(
                os.path.join(QUEUE_DIR, stem + ".json"),
                json.dumps(cand, ensure_ascii=False, indent=1))
            draft = rn.render_note(cand, fulltext=fetched["body"],
                                   fulltext_via=fetched["via"],
                                   status="candidate")
            ok_md = _atomic_write_text(
                os.path.join(QUEUE_DIR, stem + ".md"), draft)
            if ok_json and ok_md:
                state["seen"].extend(all_keys(cand))
                _mark_clerc_seen(state, cand)
                n_queued += 1
            else:
                n_err += 1
        except Exception as exc:
            _log("candidate error (%s): %r" % (
                (cand.get("title") or "")[:50], exc))
            n_err += 1

    # ---- State + summary -------------------------------------------------
    state["discards"] = state["discards"][-500:]
    state["clerc_seen_urls"] = state["clerc_seen_urls"][-2000:]
    # Advance the per-source date window ONLY when this run fully consumed
    # what it discovered. A --limit truncation or a write error leaves
    # last_run alone, so the next sweep re-covers the same window and the
    # dedup layers absorb the repeats; the backlog drains at --limit per run
    # instead of being silently lost past the 7-day overlap.
    if len(gated) <= max(1, limit) and n_err == 0:
        for src in ran_sources:
            state["last_run"][src] = started
    else:
        _log("window held: %d candidate(s) past --limit or %d error(s); "
             "last_run not advanced" % (max(0, len(gated) - max(1, limit)), n_err))
    save_state(state)
    _log("run summary: queued=%d fulltext=%d llm_discarded=%d errors=%d "
         "(queue at %s; review with research_review.py)"
         % (n_queued, n_fulltext, n_discard, n_err, QUEUE_DIR))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Discover deaf-education literature into the review queue.")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="max candidates fetched/scored/queued per run")
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                    help="override the incremental from-date for every source")
    ap.add_argument("--source", choices=SOURCES, action="append",
                    help="run only this source (repeatable; default all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be queued; write nothing")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip relevance scoring (queue everything found)")
    args = ap.parse_args(argv)
    return run(tuple(args.source) if args.source else SOURCES,
               args.since, args.limit, dry_run=args.dry_run,
               use_llm=not args.no_llm)


if __name__ == "__main__":
    sys.exit(main())
