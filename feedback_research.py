#!/usr/bin/env python3
"""feedback_research.py -- turn a parent's thumbs-down feedback into
reviewable research drafts.

The live parent advisor (vault-advisor-cloud/parent_advisor_server.py on the
droplet) logs every vote to /opt/advisor/logs-parent/feedback.jsonl and every
exchange to qa_log.jsonl. Nothing acted on those until this script. It runs on
the Mac (where ThesisVault, Ollama, and the advisor-vm ssh alias all live):

    --pull                pull both logs from advisor-vm, process every new
                          thumbs-down that carries a note
    --note "..."          ad hoc free-text feedback (Clayton relaying a parent)
    --feedback-id ID      one specific pulled record
    --dry-run             show the queries and candidates, write nothing

For each record it (1) asks the resident llama3.2:3b for academic + practical
search queries, (2) discovers candidates on OpenAlex (free-text sweep) and
Brave web search, (3) dedups against the whole vault and the review queue,
(4) fetches full text, (5) scores each candidate against the parent's actual
complaint, and (6) writes <stem>.json + <stem>.md drafts into the same
QUEUE_DIR research_agent.py uses. Nothing enters the vault here: approve with

    RESEARCH_NOTES_DIR=<vault>/LanguageDeprivationAgent/Feedback\\ Research \\
        research_review.py --push

Environment (the launchd plist under deploy/ sets all of these):
    BRAIN_ROOT        default $HOME/.advisor-feedback on macOS (queue, state,
                      pulled logs; deliberately outside the vault)
    BRAIN_VAULT       default $HOME/Desktop/Dream/ThesisVault on macOS
    OLLAMA_HOST_URL   default http://127.0.0.1:11434
    BRAVE_ENV_FILE    file holding BRAVE_API_KEY=...; default
                      $HOME/.config/jetfuel/search.env. Web discovery is Brave
                      only; without a key the web leg is skipped with an error
                      line, never silently downgraded to DuckDuckGo.
    ADVISOR_VM        ssh alias for the droplet (default advisor-vm)
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOME = os.path.expanduser("~")
if sys.platform == "darwin":
    os.environ.setdefault("BRAIN_ROOT", os.path.join(HOME, ".advisor-feedback"))
    os.environ.setdefault("BRAIN_VAULT",
                          os.path.join(HOME, "Desktop", "Dream", "ThesisVault"))


def _load_env_file(path):
    """Export KEY=VALUE lines from a dotenv-style file into os.environ
    (existing values win). Returns True when the file was read."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return True
    except Exception:
        return False


BRAVE_ENV_FILE = os.environ.get(
    "BRAVE_ENV_FILE", os.path.join(HOME, ".config", "jetfuel", "search.env"))
if not os.environ.get("BRAVE_API_KEY"):
    _load_env_file(BRAVE_ENV_FILE)
# Open-access resolver keys (OA_MAILTO, CORE_API_KEY, S2_API_KEY); optional.
OA_ENV_FILE = os.environ.get(
    "OA_ENV_FILE", os.path.join(HOME, ".config", "jetfuel", "oa.env"))
_load_env_file(OA_ENV_FILE)

import research_sources as rs          # noqa: E402
import research_fetch as rf            # noqa: E402
import research_note as rn             # noqa: E402
import research_agent as ra            # noqa: E402
from research_agent import (QUEUE_DIR, BRAIN_ROOT, SCORE_THRESHOLD,  # noqa: E402
                            _ollama_json, all_keys, queue_stem,
                            queue_dedup_keys, vault_dedup_keys,
                            load_state, save_state)
from web_cache import _atomic_write_text   # noqa: E402
import web_paths                            # noqa: E402

ADVISOR_VM = os.environ.get("ADVISOR_VM", "advisor-vm")
REMOTE_LOG_DIR = os.environ.get("ADVISOR_REMOTE_LOG_DIR",
                                "/opt/advisor/logs-parent")
FEEDBACK_DIR = os.path.join(BRAIN_ROOT, "feedback")
FEEDBACK_LOG = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
QA_LOG = os.path.join(FEEDBACK_DIR, "qa_log.jsonl")
PROCESSED_PATH = os.path.join(FEEDBACK_DIR, "processed.json")
RUNS_LOG = os.path.join(FEEDBACK_DIR, "feedback_research_runs.jsonl")

MAX_ACADEMIC_PER_QUERY = 8
MAX_WEB_PER_QUERY = 5
WEB_ABSTRACT_CHARS = 1500
MAX_PER_RECORD = 12            # drafts per feedback record; keeps review human-sized

QUERY_SYSTEM = (
    "You turn a parent's complaint about an AI advisor for families of deaf "
    "and hard of hearing children into web and literature search queries. "
    "Output only JSON.")
QUERY_TEMPLATE = (
    "A parent asked the advisor a question, got an answer, and said the answer "
    "was not good enough. Write search queries that would find what the parent "
    "actually needed.\n\n"
    "QUESTION: {question}\n\nADVISOR ANSWER (may be empty): {answer}\n\n"
    "PARENT FEEDBACK: {note}\n\n"
    "Return 1 to 2 academic queries (for a research database: short keyword "
    "phrases, no quotes) and 1 to 2 practical queries (for a web search: "
    "programs, services, organizations, state agencies, how-to guides; include "
    "any state or place the parent named).\n"
    'Respond with ONLY this JSON: {{"topic": "<3-6 word label>", '
    '"academic": ["<query>", ...], "practical": ["<query>", ...]}}')

SCORE_SYSTEM = (
    "You screen sources for a knowledge base that answers hearing parents of "
    "deaf and hard of hearing children. Output only JSON.")
SCORE_TEMPLATE = (
    "A parent complained that the advisor's answer was missing something. "
    "Rate 0 to 10 how directly this source supplies what the parent asked for. "
    "0 = unrelated, 10 = exactly what was missing.\n\n"
    "PARENT QUESTION: {question}\nPARENT FEEDBACK: {note}\n\n"
    "SOURCE TITLE: {title}\nSOURCE VENUE OR SITE: {venue}\n"
    "SOURCE TEXT: {abstract}\n\n"
    'Respond with ONLY this JSON: {{"score": <0-10>, "reason": "<one short '
    'sentence>", "note": "<2-4 plain sentences on what this source offers the '
    'parent, using only the text above>"}}')


def _log(msg):
    print("[feedback-research] " + msg, flush=True)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Feedback records
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def pull_logs():
    """scp feedback.jsonl + qa_log.jsonl from the droplet. Returns True when
    at least feedback.jsonl arrived."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    ok = True
    for name in ("feedback.jsonl", "qa_log.jsonl"):
        src = "%s:%s/%s" % (ADVISOR_VM, REMOTE_LOG_DIR, name)
        dest = os.path.join(FEEDBACK_DIR, name)
        try:
            rc = subprocess.call(["scp", "-q", "-o", "ConnectTimeout=15",
                                  src, dest + ".tmp"])
            if rc == 0:
                os.replace(dest + ".tmp", dest)
            else:
                _log("scp failed for %s (rc=%d)" % (name, rc))
                ok = ok and name != "feedback.jsonl"
        except Exception as exc:
            _log("scp error for %s: %r" % (name, exc))
            ok = ok and name != "feedback.jsonl"
    return ok


def load_processed():
    try:
        with open(PROCESSED_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_processed(ids):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    _atomic_write_text(PROCESSED_PATH, json.dumps(sorted(ids), indent=1))


def feedback_records(only_id=None, include_processed=False):
    """Thumbs-down records with a note, joined to the original exchange."""
    qa = {}
    for rec in _read_jsonl(QA_LOG):
        if rec.get("id"):
            qa[rec["id"]] = rec
    processed = load_processed()
    out = []
    for rec in _read_jsonl(FEEDBACK_LOG):
        fid = str(rec.get("id") or "").strip()
        note = str(rec.get("note") or "").strip()
        if only_id:
            if fid != only_id:
                continue
        else:
            if rec.get("vote") != "down" or not note:
                continue
            if fid in processed and not include_processed:
                continue
        ex = qa.get(fid, {})
        out.append({
            "feedback_id": fid or ("adhoc-" + str(int(time.time()))),
            "note": note,
            "question": str(ex.get("question") or "").strip(),
            "answer": str(ex.get("answer") or "").strip(),
            "ts": rec.get("ts"),
        })
    return out


# ---------------------------------------------------------------------------
# Query extraction + discovery
# ---------------------------------------------------------------------------
def extract_queries(record):
    """Ask the 3B for search queries. Returns {"topic", "academic",
    "practical"}; falls back to the raw note as one query of each kind."""
    obj = _ollama_json(QUERY_TEMPLATE.format(
        question=record["question"][:600] or "(not recorded)",
        answer=record["answer"][:800],
        note=record["note"][:800]), system=QUERY_SYSTEM)
    academic, practical, topic = [], [], ""
    if isinstance(obj, dict):
        topic = str(obj.get("topic") or "").strip()[:80]
        for key, dest in (("academic", academic), ("practical", practical)):
            vals = obj.get(key)
            if isinstance(vals, str):
                vals = [vals]
            for v in (vals or [])[:2]:
                v = str(v or "").strip()
                if 3 <= len(v) <= 200:
                    dest.append(v)
    fallback = (record["note"] or record["question"])[:200]
    if not academic:
        academic = [fallback]
    if not practical:
        practical = [fallback]
    return {"topic": topic, "academic": academic, "practical": practical,
            "llm_ok": isinstance(obj, dict)}


def web_candidates(query):
    """Brave web search -> page fetch -> candidate dicts shaped like the
    academic ones (grey-lit, web-feedback). Returns [] without a Brave key."""
    if not os.environ.get("BRAVE_API_KEY"):
        _log("no BRAVE_API_KEY (looked in %s); skipping web query %r"
             % (BRAVE_ENV_FILE, query))
        return []
    import brain
    results = brain.brave_search(query)
    if not results:
        _log("brave returned nothing for %r" % query)
        return []
    out = []
    for r in results[:MAX_WEB_PER_QUERY]:
        href = r.get("href") or ""
        if not href or web_paths.is_denied(href):
            continue
        text = ""
        try:
            text = brain.fetch_page(href) or ""
        except Exception:
            text = ""
        snippet_only = False
        if not text:
            text = str(r.get("body") or "").strip()
            snippet_only = True
        if not text:
            continue
        c = rs._blank_candidate()
        c["title"] = str(r.get("title") or href).strip()[:300]
        c["landing_url"] = href
        c["fulltext_url"] = href
        try:
            from urllib.parse import urlparse
            c["venue"] = urlparse(href).netloc.lower().replace("www.", "", 1)
        except Exception:
            c["venue"] = ""
        c["access"] = "grey-lit"
        c["peer_reviewed"] = False
        c["discovery_source"] = "web-feedback"
        c["abstract"] = text[:WEB_ABSTRACT_CHARS]
        c["_web_text"] = "" if snippet_only else text
        c["fetch_note"] = ("search snippet only (page fetch failed)"
                           if snippet_only else "page fetched via brave")
        out.append(c)
    return out


def discover(queries):
    cands = []
    for q in queries["academic"]:
        got = rs.openalex_candidates_for_query(q)
        _log("openalex %r: %d" % (q, len(got)))
        cands.extend(got[:MAX_ACADEMIC_PER_QUERY])
    for q in queries["practical"]:
        got = web_candidates(q)
        _log("brave %r: %d" % (q, len(got)))
        cands.extend(got)
    return cands


# ---------------------------------------------------------------------------
# Scoring against the complaint
# ---------------------------------------------------------------------------
def score_against_feedback(cand, record):
    obj = _ollama_json(SCORE_TEMPLATE.format(
        question=record["question"][:500] or "(not recorded)",
        note=record["note"][:600],
        title=(cand.get("title") or "")[:300],
        venue=(cand.get("venue") or "")[:120],
        abstract=(cand.get("abstract") or "")[:1500]), system=SCORE_SYSTEM)
    score, reason, note = None, "", ""
    if isinstance(obj, dict):
        try:
            score = max(0, min(10, int(obj.get("score"))))
        except Exception:
            score = None
        reason = str(obj.get("reason") or "").strip()[:300]
        note = str(obj.get("note") or "").strip()[:1200]
    cand["score"] = score
    cand["score_reason"] = reason
    if note:
        cand["relevance_note"] = note
    return cand


# ---------------------------------------------------------------------------
# One feedback record
# ---------------------------------------------------------------------------
def process_record(record, dry_run=False, use_llm=True):
    started = _now()
    _log("feedback %s: %r" % (record["feedback_id"], record["note"][:100]))
    queries = extract_queries(record) if use_llm else {
        "topic": "", "academic": [record["note"][:200]],
        "practical": [record["note"][:200]], "llm_ok": False}
    if use_llm and not queries["llm_ok"]:
        _log("query extraction fell back to the raw note (Ollama unreachable "
             "or bad JSON)")
    _log("queries: academic=%r practical=%r" % (
        queries["academic"], queries["practical"]))

    candidates = discover(queries)

    known = queue_dedup_keys() | vault_dedup_keys()
    fresh, in_run, n_dup = [], set(), 0
    for cand in candidates:
        keys = all_keys(cand)
        if any(k in known for k in keys) or any(k in in_run for k in keys):
            n_dup += 1
            continue
        in_run.update(keys)
        fresh.append(cand)
    fresh = fresh[:MAX_PER_RECORD]
    _log("candidates: %d raw, %d deduped away, %d to consider"
         % (len(candidates), n_dup, len(fresh)))

    summary = {"feedback_id": record["feedback_id"], "ts": started,
               "topic": queries["topic"], "queries": {
                   "academic": queries["academic"],
                   "practical": queries["practical"]},
               "n_candidates": len(candidates), "n_dedup": n_dup,
               "n_queued": 0, "n_discarded": 0, "n_errors": 0,
               "queue_stems": [], "dry_run": dry_run}

    if dry_run:
        for cand in fresh:
            print("  [%s] %s (%s) %s" % (
                cand.get("discovery_source"), (cand.get("title") or "")[:70],
                cand.get("year") or "web",
                ("doi:" + cand["doi"]) if cand.get("doi")
                else (cand.get("landing_url") or "")[:70]))
        _log("dry run: nothing written")
        return summary

    os.makedirs(QUEUE_DIR, exist_ok=True)
    state = load_state()
    for cand in fresh:
        try:
            fetched = {"body": cand.pop("_web_text", "") or "",
                       "via": "brave" if cand.get("discovery_source") ==
                       "web-feedback" else "", "note": ""}
            if cand.get("access") != "grey-lit":
                fetched = rf.fetch_fulltext(cand)
                if not fetched["body"] and cand.get("access") == "open":
                    cand["access"] = "abstract-only"
                cand["fetch_note"] = fetched["note"]
            if use_llm:
                score_against_feedback(cand, record)
            score = cand.get("score")
            if score is not None and score < SCORE_THRESHOLD:
                state["discards"].append({
                    "keys": all_keys(cand), "title": cand.get("title"),
                    "score": score, "reason": cand.get("score_reason"),
                    "feedback_id": record["feedback_id"], "ts": started})
                state["rejected"].extend(all_keys(cand))
                summary["n_discarded"] += 1
                continue
            cand["feedback_id"] = record["feedback_id"]
            cand["feedback_note"] = record["note"]
            cand["feedback_question"] = record["question"]
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
                summary["n_queued"] += 1
                summary["queue_stems"].append(stem)
            else:
                summary["n_errors"] += 1
        except Exception as exc:
            _log("candidate error (%s): %r" % (
                (cand.get("title") or "")[:50], exc))
            summary["n_errors"] += 1
    state["discards"] = state["discards"][-500:]
    save_state(state)
    _log("feedback %s: queued=%d discarded=%d errors=%d"
         % (record["feedback_id"], summary["n_queued"],
            summary["n_discarded"], summary["n_errors"]))
    return summary


def _append_run(summary):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    try:
        with open(RUNS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except Exception as exc:
        _log("could not append run log: %r" % exc)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Research a parent's feedback into the review queue.")
    ap.add_argument("--pull", action="store_true",
                    help="scp the advisor logs from %s and process every new "
                         "thumbs-down with a note" % ADVISOR_VM)
    ap.add_argument("--note", help="ad hoc feedback text")
    ap.add_argument("--question", default="",
                    help="the parent's question, for --note")
    ap.add_argument("--feedback-id", help="process one pulled record by id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ollama: raw note as the query, nothing scored")
    ap.add_argument("--limit", type=int, default=10,
                    help="max feedback records per run (default 10)")
    args = ap.parse_args(argv)

    records = []
    if args.note:
        records = [{"feedback_id": "adhoc-" + datetime.datetime.now()
                    .strftime("%Y%m%d-%H%M%S"), "note": args.note.strip(),
                    "question": args.question.strip(), "answer": "",
                    "ts": time.time()}]
    elif args.feedback_id:
        records = feedback_records(only_id=args.feedback_id,
                                   include_processed=True)
        if not records:
            _log("no record with id %s in %s" % (args.feedback_id, FEEDBACK_LOG))
            return 2
    elif args.pull:
        if not pull_logs():
            _log("pull failed; nothing processed")
            return 1
        records = feedback_records()
        _log("%d unprocessed thumbs-down record(s) with notes" % len(records))
    else:
        ap.print_help()
        return 2

    if not records:
        return 0
    records = records[:max(1, args.limit)]

    # Fail fast when Ollama is down: every record would otherwise degrade to
    # unscored raw-note queries and flood the queue.
    if not args.no_llm and _ollama_json(
            'Reply with {"ok": true}', system="Output only JSON.") is None:
        _log("Ollama not answering at %s; rerun when it is up, or pass "
             "--no-llm" % ra.OLLAMA_HOST)
        return 1

    processed = load_processed()
    for rec in records:
        summary = process_record(rec, dry_run=args.dry_run,
                                 use_llm=not args.no_llm)
        _append_run(summary)
        if not args.dry_run and summary["n_errors"] == 0 \
                and not rec["feedback_id"].startswith("adhoc-"):
            processed.add(rec["feedback_id"])
    if not args.dry_run:
        save_processed(processed)
        _log("done; review with: RESEARCH_NOTES_DIR=<vault>/LanguageDeprivation"
             "Agent/Feedback\\ Research research_review.py --push  (queue at %s)"
             % QUEUE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
