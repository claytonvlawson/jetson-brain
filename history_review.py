#!/usr/bin/env python3
"""history_review.py - nightly review of the brain REPL's own Q&A transcript.

Every answered turn in the REPL already lands in cache/transcript.jsonl (see
brain_transcript.py; that is what /history reads). Nothing reviews it for
material worth adding to ThesisVault, and nothing produces an away-from-the-
Jetson summary of a day's usage. This script closes both gaps, once nightly:

  1. Read every transcript turn since the last run (state.json's cursor),
     across transcript.jsonl and, if the log rotated, transcript.jsonl.1.
  2. Classify each turn with the resident LLM: a RETRIEVAL GAP (the sources
     were empty/thin -- the vault likely lacks material on this), NOVEL INFO
     (the answer said something not evidently covered by its own cited
     source titles), or neither.
  3. Draft one candidate note per gap/novel turn, plus one deterministic
     (no LLM) daily-summary note per calendar day covered, into
     history_review/queue/ -- OUTSIDE the vault, same curated-growth rule
     research_agent.py and web_review.py already enforce: automation drafts,
     a human promotes with history_review_approve.py.
  4. Refresh a rolling digest note in vault/_PendingReview/ so the queue is
     visible from Obsidian without SSH (mirrors web_review_digest.py).
  5. Render the same gaps/novel-info/summary content as a .docx report
     (JetFuel/<date>.docx) and best-effort push it to the Mac's
     ~/Desktop/JetFuel/ over scp, so it can be enriched and emailed from
     there. A failed push is logged, never fatal: the .docx still exists on
     the Jetson for a later manual copy.

Run: python3 history_review.py
Meant to run nightly via history-review.timer (23:59, see that file for the
install steps -- same one-time sudo dance as brain-learn.timer).

Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import os
import sys
import json
import time
import datetime
import subprocess
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import history_note as hn
from web_paths import VAULT_ROOT
from web_cache import _atomic_write_text

BRAIN_ROOT = os.environ.get("BRAIN_ROOT", "/mnt/nvme/brain")
CACHE_DIR = os.path.join(BRAIN_ROOT, "cache")
TRANSCRIPT_PATH = os.path.join(CACHE_DIR, "transcript.jsonl")

HISTORY_DIR = os.environ.get("HISTORY_REVIEW_DIR",
                             os.path.join(BRAIN_ROOT, "history_review"))
STATE_PATH = os.path.join(HISTORY_DIR, "state.json")
QUEUE_DIR = os.path.join(HISTORY_DIR, "queue")
JETFUEL_DIR = os.path.join(HISTORY_DIR, "JetFuel")

PENDING_REVIEW_DIR = os.path.join(VAULT_ROOT, "_PendingReview")
DIGEST_PATH = os.path.join(PENDING_REVIEW_DIR, "pending-history-review.md")

SSH_HOST = "user@192.168.55.1"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
# Resident fast model, matching research_agent.py's choice: a background
# nightly job must not evict the model a live REPL session is using.
LLM_MODEL = os.environ.get("HISTORY_REVIEW_MODEL", "llama3.2:3b")
LLM_TIMEOUT = 120
ANSWER_EXCERPT_CAP = 1500

CLASSIFY_SYSTEM = (
    "You review one turn from a Q&A transcript produced by a personal "
    "research assistant grounded in an Obsidian vault (ThesisVault, about "
    "language deprivation and deaf education, plus general topics the "
    "owner asks about). Output only JSON.")
CLASSIFY_TEMPLATE = (
    "QUESTION: {q}\n\nANSWER: {a}\n\nSOURCES CITED (titles only): {sources}\n\n"
    "Decide ONE verdict:\n"
    '  "gap"    - the sources are empty or clearly thin for this question, '
    "so the vault likely lacks real material on this topic and a note "
    "should be drafted describing the gap.\n"
    '  "novel"  - the answer states something substantive that is not '
    "evidently covered by the titles of its own cited sources (new "
    "information worth adding to the vault).\n"
    '  "neither" - routine, already well covered, or nothing worth '
    "drafting a note about.\n\n"
    "If gap or novel, also give a short note title (under 12 words) and a "
    "2-4 sentence summary a human could use to decide whether to write a "
    "real vault note.\n\n"
    'Respond with ONLY this JSON: {{"verdict": "gap|novel|neither", '
    '"title": "<short title or empty>", "summary": "<2-4 sentences or empty>"}}'
)

JUNK_DIR_NAMES = ("_WebCache", "_Research", "_PendingReview", "_Advisor",
                  "Templates")


def _log(msg):
    print("[history-review] " + msg, flush=True)


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
    state.setdefault("last_ts", 0.0)
    return state


def save_state(state):
    try:
        data = json.dumps(state, ensure_ascii=False, indent=1)
    except Exception:
        return False
    return _atomic_write_text(STATE_PATH, data)


# ---------------------------------------------------------------------------
# Reading the transcript
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except Exception:
        pass
    return out


def read_new_entries(state):
    """Every transcript turn with ts > state['last_ts'], oldest first, across
    both the live file and one rotated generation (transcript.jsonl.1) so a
    missed night straddling a rotation does not silently drop entries."""
    last_ts = float(state.get("last_ts") or 0.0)
    entries = _read_jsonl(TRANSCRIPT_PATH + ".1") + _read_jsonl(TRANSCRIPT_PATH)
    fresh = [e for e in entries if float(e.get("ts") or 0) > last_ts]
    fresh.sort(key=lambda e: float(e.get("ts") or 0))
    return fresh


# ---------------------------------------------------------------------------
# LLM classification (best effort; failure just skips that turn)
# ---------------------------------------------------------------------------
def _ollama_json(prompt, system=CLASSIFY_SYSTEM):
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False, "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300},
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


def classify_turn(entry):
    """Returns a candidate dict (kind gap/novel) or None. Never raises."""
    q = str(entry.get("q") or "")[:800]
    a = str(entry.get("a") or "")[:ANSWER_EXCERPT_CAP]
    sources = entry.get("sources") or []
    titles = ", ".join(str(s).rsplit("/", 1)[-1].replace(".md", "")
                       for s in sources) or "(none cited)"
    if not q or not a:
        return None
    obj = _ollama_json(CLASSIFY_TEMPLATE.format(q=q, a=a, sources=titles))
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict") or "neither").strip().lower()
    if verdict not in ("gap", "novel"):
        return None
    title = str(obj.get("title") or "").strip()
    summary = str(obj.get("summary") or "").strip()
    if not title or not summary:
        return None
    return {
        "kind": verdict,
        "title": title[:120],
        "summary": summary[:1200],
        "questions": [q],
        "source_ts": [float(entry.get("ts") or 0)],
        "discovered_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Deterministic daily summaries (no LLM: a plain log, not a judgment call)
# ---------------------------------------------------------------------------
def _date_str(ts):
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except Exception:
        return "unknown-date"


def build_daily_summaries(entries):
    by_date = {}
    for e in entries:
        d = _date_str(e.get("ts") or 0)
        by_date.setdefault(d, []).append(e)

    summaries = []
    for date, day_entries in sorted(by_date.items()):
        lines = []
        for e in day_entries:
            q = str(e.get("q") or "").strip()
            mode = str(e.get("mode") or "fast")
            model = str(e.get("model") or "")
            n_src = len(e.get("sources") or [])
            lines.append("- [%s/%s, %d source(s)] %s" % (
                mode, model or "?", n_src, q[:200]))
        summaries.append({
            "kind": "summary",
            "title": "%s session summary (%d question(s))" % (date, len(day_entries)),
            "summary": "\n".join(lines),
            "questions": [],
            "source_ts": [float(day_entries[0].get("ts") or 0)],
            "discovered_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
    return summaries


# ---------------------------------------------------------------------------
# Digest note (mirrors web_review_digest.py's rolling/overwritten pattern)
# ---------------------------------------------------------------------------
def _queue_stems():
    try:
        return sorted(n[:-3] for n in os.listdir(QUEUE_DIR) if n.endswith(".md"))
    except Exception:
        return []


def write_digest():
    stems = _queue_stems()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        "---",
        "title: Pending history review",
        "generator: history_review.py",
        "tags: [history-review-digest]",
        "updated_at: %s" % now,
        "---",
        "",
        "# Pending history review",
        "",
        "Generated %s by history_review.py from your own brain REPL "
        "transcript. Items disappear from here as soon as they are promoted "
        "or rejected via history_review_approve.py." % now,
        "",
        "## Pending (%d)" % len(stems),
        "",
    ]
    if not stems:
        lines.append("Nothing pending review.")
        lines.append("")
    else:
        for stem in stems:
            lines.append("- `%s`" % stem)
            lines.append("  ```")
            lines.append('  ssh %s "cd /mnt/nvme/brain && '
                         'python3 history_review_approve.py --promote %s"'
                         % (SSH_HOST, stem))
            lines.append('  ssh %s "cd /mnt/nvme/brain && '
                         'python3 history_review_approve.py --reject %s"'
                         % (SSH_HOST, stem))
            lines.append("  ```")
        lines.append("")
    text = "\n".join(lines)
    ok = _atomic_write_text(DIGEST_PATH, text)
    if not ok:
        _log("FAILED to write digest %s" % DIGEST_PATH)
    return ok


# ---------------------------------------------------------------------------
# .docx report
# ---------------------------------------------------------------------------
def render_docx(gaps, novels, summaries, date, dest_path):
    try:
        import docx
        from docx.shared import Pt
    except Exception as exc:
        _log("python-docx not available (%r); skipping .docx report" % exc)
        return ""

    try:
        doc = docx.Document()
        doc.add_heading("JetFuel: history review for %s" % date, level=1)
        doc.add_paragraph(
            "Generated by history_review.py from your brain REPL transcript. "
            "Every item below is a DRAFT: nothing here is in ThesisVault "
            "yet. Approve or reject via history_review_approve.py, or check "
            "vault/_PendingReview/pending-history-review.md in Obsidian.")

        def _section(title, cands, empty_msg):
            doc.add_heading(title, level=2)
            if not cands:
                doc.add_paragraph(empty_msg)
                return
            for c in cands:
                doc.add_heading(str(c.get("title") or "Untitled"), level=3)
                p = doc.add_paragraph(str(c.get("summary") or ""))
                p.style.font.size = Pt(11)
                for q in (c.get("questions") or []):
                    doc.add_paragraph("Q: " + str(q), style="List Bullet")

        _section("Retrieval gaps", gaps,
                 "No retrieval gaps flagged in this run.")
        _section("Novel information", novels,
                 "No novel information flagged in this run.")
        _section("Session summary", summaries,
                 "No new transcript entries in this run.")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        doc.save(dest_path)
        return dest_path
    except Exception as exc:
        _log("failed to render .docx (%r)" % exc)
        return ""


# ---------------------------------------------------------------------------
# Push to Mac (best effort; never fatal)
# ---------------------------------------------------------------------------
def _kick_jetfuel_chain(mac_host):
    """Fire-and-forget: tell the Mac's JetFuel orchestrator (jetfuel_chain.py
    -- enrich -> research -> litreview -> email) the report just landed,
    instead of it guessing a clock time. Best-effort only: the Mac also has
    its own 4:30am safety-net LaunchAgent, so a failure here (Mac asleep,
    SSH hiccup) just means that backstop runs the chain later instead."""
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", mac_host,
             "nohup python3 ~/scripts/jetfuel_chain.py "
             ">/dev/null 2>&1 &"],
            capture_output=True, text=True, timeout=15)
        _log("kicked jetfuel_chain.py on %s" % mac_host)
    except Exception as exc:
        _log("failed to kick jetfuel_chain.py on %s (%r); the 4:30am "
             "safety-net timer will pick it up instead." % (mac_host, exc))


def push_to_mac(local_path):
    if not local_path or not os.path.exists(local_path):
        return False
    mac_host = os.environ.get("JETFUEL_MAC_HOST", "")
    mac_dest = os.environ.get("JETFUEL_MAC_DEST", "~/Desktop/JetFuel/")
    if not mac_host:
        _log("JETFUEL_MAC_HOST not set; leaving the report on the Jetson "
             "at %s" % local_path)
        return False
    try:
        result = subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             local_path, "%s:%s" % (mac_host, mac_dest)],
            capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            _log("pushed %s -> %s:%s" % (local_path, mac_host, mac_dest))
            _kick_jetfuel_chain(mac_host)
            return True
        _log("scp to %s failed (rc=%d): %s" % (
            mac_host, result.returncode, (result.stderr or "").strip()[:300]))
        return False
    except Exception as exc:
        _log("scp to %s raised %r; report stays on the Jetson" % (mac_host, exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)
    os.makedirs(JETFUEL_DIR, exist_ok=True)
    os.makedirs(PENDING_REVIEW_DIR, exist_ok=True)

    state = load_state()
    entries = read_new_entries(state)
    _log("%d new transcript entry(ies) since last run" % len(entries))

    gaps, novels = [], []
    for entry in entries:
        cand = classify_turn(entry)
        if not cand:
            continue
        if cand["kind"] == "gap":
            gaps.append(cand)
        elif cand["kind"] == "novel":
            novels.append(cand)

    summaries = build_daily_summaries(entries)

    written = 0
    for cand in gaps + novels + summaries:
        path = hn.write_note(cand, QUEUE_DIR)
        if path:
            written += 1
        else:
            _log("failed to write queue note for %r" % cand.get("title"))
    _log("drafted %d queue note(s) (%d gap, %d novel, %d summary)" % (
        written, len(gaps), len(novels), len(summaries)))

    write_digest()

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    docx_path = os.path.join(JETFUEL_DIR, "%s.docx" % today)
    rendered = render_docx(gaps, novels, summaries, today, docx_path)
    if rendered:
        _log("wrote report %s" % rendered)
        push_to_mac(rendered)

    if entries:
        state["last_ts"] = max(float(e.get("ts") or 0) for e in entries)
        save_state(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
