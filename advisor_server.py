#!/usr/bin/env python3
"""advisor_server.py -- public HTTP wrapper around the jetson-brain RAG.

Exposes a single endpoint the "Ask the advisor" widget on learnwithclay.ai
speaks natively:

    POST /chat      {"question": "..."}          ->  {"id": "...", "answer": "...", "sources": [...]}
    POST /feedback  {"id", "vote", "note"?}      ->  {"ok": true}
    GET  /health                                 ->  {"ok": true, "chunks": N, "model": "..."}

Every answered exchange is appended to qa_log.jsonl and every vote to
feedback.jsonl (both under ADVISOR_LOG_DIR). review_qa.py turns the good ones
into vault notes. Visitor input never touches the vault or index directly.

Design notes:
- Zero extra dependencies (stdlib http.server) so it runs anywhere brain.py runs;
  nothing new to install on the Jetson.
- Binds to 127.0.0.1 ONLY. Reach it from the public internet through cloudflared
  (or Tailscale Funnel) plus the Vercel /api proxy. Never bind 0.0.0.0.
- One answer at a time: this is a single 8GB device, so a semaphore serializes
  inference and concurrent visitors queue behind one in-flight answer instead of
  thrashing / OOM-ing the GPU. Past a timeout they get a friendly 503.
- Web search is OFF by default for the public endpoint (predictable, faster,
  vault-grounded). Flip ADVISOR_WEB=1 to allow it.

Run locally to test:
    ADVISOR_TOKEN=$(openssl rand -hex 32) python3 advisor_server.py
"""
import functools
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brain  # safe to import: main() sits behind an __main__ guard

HOST = os.environ.get("ADVISOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("ADVISOR_PORT", "8000"))
TOKEN = os.environ.get("ADVISOR_TOKEN", "")            # shared secret; "" disables the check
MODEL = os.environ.get("ADVISOR_MODEL", brain.FAST_MODEL)
WEB_ENABLED = os.environ.get("ADVISOR_WEB", "0") == "1"
MAX_Q = int(os.environ.get("ADVISOR_MAX_Q", "500"))    # cap question length
MAX_NOTE = int(os.environ.get("ADVISOR_MAX_NOTE", "1000"))  # cap feedback note length
QUEUE_TIMEOUT = float(os.environ.get("ADVISOR_QUEUE_TIMEOUT", "45"))  # seconds to wait for the GPU

# Where the synced vault lives (deploy.sh rsyncs ~/Desktop/Dream/ThesisVault here,
# *.md only). Cited source paths from brain.vault_sources() are relative to it, so
# we read each note's frontmatter + "## Full citation" block straight off disk at
# answer time to hand the widget a full, clickable academic citation. No reindex,
# no extra deps.
VAULT_ROOT = os.path.normpath(os.path.join(brain.BRAIN_ROOT, "vault"))
MAX_CITATIONS = int(os.environ.get("ADVISOR_MAX_CITATIONS", "6"))

# Capture layer: every answered exchange and every piece of visitor feedback is
# appended as one JSON object per line. These logs are the raw material for the
# curated-growth loop (review_qa.py): a human reviews them and promotes the good
# answers into the vault. Nothing here writes to the vault or the index directly,
# so a visitor can never poison what the advisor knows. Logs live OFF the indexed
# vault tree on purpose.
LOG_DIR = os.environ.get("ADVISOR_LOG_DIR", os.path.join(brain.BRAIN_ROOT, "advisor"))
QA_LOG = os.path.join(LOG_DIR, "qa_log.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading the vault index...", flush=True)
CHUNKS, BM25, META = brain.load_index()
print("Loaded %d chunks. Model: %s. Web: %s" % (len(CHUNKS), MODEL, WEB_ENABLED), flush=True)
print("Logging exchanges to %s" % QA_LOG, flush=True)

# Single-flight: the whole point is one grounded answer at a time on one device.
_gpu = threading.BoundedSemaphore(1)

# Serializes appends so concurrent request threads never interleave a JSON line.
_log_lock = threading.Lock()


def _append_jsonl(path, record):
    """Append one record as a single JSON line. Best-effort: a logging failure
    must never break answering, so we swallow and print rather than raise."""
    try:
        line = json.dumps(record, ensure_ascii=False)
        with _log_lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # disk full, permissions, anything
        print("[advisor] could not write log %s: %s" % (path, exc), flush=True)


# ---------------------------------------------------------------------------
# Citations: turn a cited note's rel-path into a full academic citation with a
# clickable link, read straight from the synced vault. Regex + file read only
# (no YAML dep) so this stays stdlib-clean like the rest of the server.
# ---------------------------------------------------------------------------
def _split_frontmatter(text):
    """(frontmatter, body) split on a leading `---` YAML block. Mirrors
    build_index.split_frontmatter so both see notes the same way."""
    if not text.startswith("---"):
        return "", text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return "", text


def _fm_value(fm, key):
    """First top-level `key: value` from a frontmatter block, else ""."""
    for line in fm.split("\n"):
        m = re.match(r"^" + re.escape(key) + r"\s*:\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def _clean_md(s):
    """Strip Obsidian wikilinks and `*` emphasis so citation text reads clean."""
    s = re.sub(r"!\[\[[^\]]*\]\]", "", s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    return s.replace("*", "").strip()


def _first_h1(body):
    for line in body.split("\n"):
        m = re.match(r"^#\s+(.*)$", line)
        if m:
            return _clean_md(m.group(1))
    return ""


def _extract_full_citation(body):
    """Text under the note's `## Full citation` heading, up to the next heading,
    collapsed to one clean line. "" when the note has no such section."""
    out, capturing = [], False
    for line in body.split("\n"):
        if re.match(r"^#{1,6}\s", line):
            if capturing:
                break  # next heading ends the section
            if re.match(r"^#{1,6}\s+Full citation\s*$", line, re.IGNORECASE):
                capturing = True
            continue
        if capturing:
            out.append(line)
    text = _clean_md(" ".join(l.strip() for l in out if l.strip()))
    return re.sub(r"\s+", " ", text).strip()


def _extract_callout_citation(body):
    """Citation text from an Obsidian `> [!type] Citation` callout, which some
    source notes use instead of a `## Full citation` heading. Grabs the callout's
    `>` lines up to where it ends, collapsed to one clean line. "" when absent.
    Used only for the citation TEXT (not for link scraping), so it never revives
    the misleading-link behavior the `## Full citation` gate guards against."""
    out, capturing = [], False
    for line in body.split("\n"):
        if not capturing:
            m = re.match(r"^\s*>\s*\[!\w+\]\s*Citation\b[:]?(.*)$", line, re.IGNORECASE)
            if m:
                capturing = True
                if m.group(1).strip():
                    out.append(m.group(1).strip())
            continue
        # Inside the callout: every wrapped line still starts with ">". The first
        # line that does not ends the callout.
        if re.match(r"^\s*>", line):
            out.append(re.sub(r"^\s*>\s?", "", line))
        else:
            break
    text = _clean_md(" ".join(l.strip() for l in out if l.strip()))
    return re.sub(r"\s+", " ", text).strip()


def _safe_url(u):
    """Return u only if it is a plain http(s) URL. Blocks javascript:/data: and
    other schemes since it demands an http(s) prefix. Trims trailing punctuation."""
    u = (u or "").strip().strip("<>").strip().rstrip(").,;")
    return u if re.match(r"^https?://\S+$", u, re.IGNORECASE) else ""


def _safe_doi(d):
    """A DOI always starts with `10.`; reject anything else so we never build a
    doi.org link from junk."""
    d = (d or "").strip()
    return d if re.match(r"^10\.\S+$", d) else ""


def _first_body_url(body):
    m = re.search(r"https?://[^\s)>\]]+", body)
    return _safe_url(m.group(0)) if m else ""


@functools.lru_cache(maxsize=512)
def citation_for(rel_source):
    """Resolve one cited note (rel-path under VAULT_ROOT) to
    {title, citation, url, doi}. Cached (lru_cache is thread-safe under the GIL)
    since the same handful of notes recur across answers. Degrades to the note's
    basename with no link if the file is missing, unreadable, or citation-poor."""
    basename = os.path.splitext(os.path.basename(str(rel_source)))[0]
    fallback = {"title": basename, "citation": basename, "url": "", "doi": ""}
    path = os.path.normpath(os.path.join(VAULT_ROOT, str(rel_source)))
    # Path-traversal guard: a crafted "../.." rel_source must not escape the vault.
    if path != VAULT_ROOT and not path.startswith(VAULT_ROOT + os.sep):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return fallback
    fm, body = _split_frontmatter(raw)
    title = _fm_value(fm, "name") or _first_h1(body) or basename
    doi = _safe_doi(_fm_value(fm, "doi"))
    url = _safe_url(_fm_value(fm, "url"))
    if not url and doi:
        url = "https://doi.org/" + doi
    full_citation = _extract_full_citation(body)
    # Only scrape a body URL when the note is a real single source (it carries a
    # `## Full citation` block). Index/concept notes list many links; grabbing the
    # first one would render a misleading "View Source" for the whole note.
    if not url and full_citation:
        url = _first_body_url(body)
    # Some source notes carry their citation as a `> [!type] Citation` callout
    # rather than a `## Full citation` heading; fall back to it for the text so
    # those notes read as a real citation instead of a bare title. No link is
    # scraped from these (URL logic above is untouched), which is fine: they
    # carry no DOI/URL of their own.
    citation = full_citation or _extract_callout_citation(body) or title
    return {"title": title, "citation": citation, "url": url, "doi": doi}


def build_citations(sources):
    """Ordered, de-duplicated citations for the cited sources: url-bearing first
    (so the clickable ones lead), then capped at MAX_CITATIONS."""
    seen, resolved = set(), []
    for src in sources:
        key = str(src)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(citation_for(key))
    with_url = [c for c in resolved if c["url"]]
    without = [c for c in resolved if not c["url"]]
    return (with_url + without)[:MAX_CITATIONS]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (or the tunnel) hung up; nothing to do

    def log_message(self, *args):
        pass  # quiet the default per-request stderr line

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "chunks": len(CHUNKS), "model": MODEL})
        else:
            self._send(404, {"error": "not found"})

    def _read_json(self, max_len=10000):
        """Read and parse a JSON body, or return None and send the error itself."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > max_len:
            self._send(400, {"error": "bad request"})
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send(400, {"error": "invalid json"})
            return None

    def _authorized(self):
        return not TOKEN or self.headers.get("Authorization", "") == "Bearer " + TOKEN

    def do_POST(self):
        if self.path == "/chat":
            return self._handle_chat()
        if self.path == "/feedback":
            return self._handle_feedback()
        return self._send(404, {"error": "not found"})

    def _handle_chat(self):
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})

        data = self._read_json()
        if data is None:
            return  # _read_json already sent the error
        question = str(data.get("question", "")).strip()
        if not question:
            return self._send(400, {"error": "empty question"})
        question = question[:MAX_Q]

        # If the brain is busy past the timeout, tell the widget to retry rather
        # than piling more work onto the device.
        if not _gpu.acquire(timeout=QUEUE_TIMEOUT):
            return self._send(503, {"error": "The advisor is busy. Please try again in a moment."})
        try:
            answer, sources = brain.handle_question(
                MODEL, question, [], BM25, CHUNKS, META, WEB_ENABLED, brain._NullRun())
        except Exception as exc:  # never let one bad question take the server down
            print("[advisor] error:", exc, flush=True)
            return self._send(502, {"error": "The advisor hit an error answering that."})
        finally:
            _gpu.release()

        if not answer:
            return self._send(502, {"error": "The advisor could not answer that from the vault."})

        # Mint an id and log the exchange so review_qa.py can later promote the
        # good ones into the vault. The id lets a later /feedback call attach a
        # visitor's thumbs vote to this exact answer.
        qid = uuid.uuid4().hex[:12]
        sources = [str(s) for s in (sources or [])]
        # Full academic citations with clickable links, read from the vault. Kept
        # alongside (not replacing) `sources` so review_qa.py stays back-compatible.
        citations = build_citations(sources)
        _append_jsonl(QA_LOG, {
            "id": qid,
            "ts": time.time(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "citations": citations,
        })
        return self._send(200, {
            "id": qid,
            "answer": answer,
            "sources": sources,
            "citations": citations,
        })

    def _handle_feedback(self):
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})

        data = self._read_json(max_len=4000)
        if data is None:
            return  # _read_json already sent the error
        qid = str(data.get("id", "")).strip()[:32]
        vote = str(data.get("vote", "")).strip().lower()
        note = str(data.get("note", "")).strip()[:MAX_NOTE]
        if vote not in ("up", "down"):
            return self._send(400, {"error": "vote must be 'up' or 'down'"})

        _append_jsonl(FEEDBACK_LOG, {
            "id": qid,
            "ts": time.time(),
            "vote": vote,
            "note": note,
        })
        return self._send(200, {"ok": True})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("advisor listening on http://%s:%d  (POST /chat, POST /feedback)" % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
