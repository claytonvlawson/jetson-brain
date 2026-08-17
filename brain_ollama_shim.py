#!/usr/bin/env python3
r"""brain_ollama_shim.py -- speak Ollama's wire protocol so Obsidian's Smart Chat
Pro plugin talks to the jetson-brain RAG with zero plugin patching.

Smart Chat ships an "Ollama (Local)" provider whose only user-facing setting is
"Ollama host". Point that at this server and every chat message goes through
brain.handle_question (BM25 + dense hybrid retrieval, wikilink expansion, the
grounding contract) instead of through a raw model. The plugin never learns it
is not talking to Ollama.

    GET  /                -> "Ollama is running"   (connectivity probe)
    GET  /api/version     -> {"version": "..."}
    GET  /api/tags        -> the two advertised models
    POST /api/show        -> per-model metadata (the plugin calls this per model)
    POST /api/chat        -> a LIVE NDJSON stream of Ollama chat frames, one
                             frame per token as the model generates it, or one
                             JSON object when {"stream": false}

Streaming is real, not a replay. brain.py already streams every token through
the callable returned by run.begin_answer() (chat_stream_holdback), so the shim
hands handle_question a _CaptureRun whose sink pushes onto a queue, runs the
answer on a worker thread, and writes chunked NDJSON from the request thread as
tokens arrive. Nothing in brain.py changes. One caveat worth knowing before
judging the latency: chat_stream_holdback buffers the head of the generation
until the INSUFFICIENT-refusal decision is safe, so the first ~210 characters
still land in one burst. What this buys is that the REST of a long answer
renders progressively instead of the whole thing appearing at the end.

Everything below was verified against the plugin source, not memory
(.obsidian/plugins/smart-chat/main.js, class SmartChatModelOllamaAdapter). The
four things that actually constrain this file:

1. Model listing is TWO calls, not one. get_models() does GET /api/tags, then
   for EVERY entry in `models` it does POST /api/show {"model": <name>} and
   merges the result with {name}. parse_model_data() then reads only `name`
   (entries whose name contains "embed" are dropped) and the first key of
   `model_info` containing ".context_length" for max_input_tokens. So /api/show
   is mandatory - without it the dropdown renders "Failed to fetch models".

2. End-of-stream is detected by a SUBSTRING, not by the HTTP body ending:
   is_end_of_stream(e) { return e.data.includes('"done_reason"') }. The final
   frame must literally contain the characters "done_reason" (with quotes), and
   no earlier frame may contain them, or the plugin truncates the answer at
   that frame. _content_frames() below splits answer text so the substring can
   never appear inside a content frame.

3. The stream reader is a hand-rolled XMLHttpRequest wrapper (SmartStreamer)
   that splits the accumulated responseText on /(\r\n|\n|\r)/ and dispatches
   each non-blank piece as one message. That is exactly NDJSON, so one JSON
   object per line, each terminated by "\n". It also treats any status other
   than 200 as a stream error, so error paths must still answer 200 on /api/chat
   or the user sees a bare "Stream error" in the console.

4. XHR from Obsidian's renderer is a cross-origin request (origin
   app://obsidian.md), so CORS headers and an OPTIONS preflight are required.
   Real Ollama allows app://* by default; we allow "*" because this server is
   bound to localhost or a tailnet address, both already private.

Auth (SHIM_TOKEN) is unusual here for a reason: the plugin's
SmartChatModelOllamaRequestAdapter.to_platform() returns {url, method, body}
with NO headers, so the streaming request and /api/show carry no Authorization
at all (only /api/tags gets one, a hardcoded `Bearer na`). A header-only bearer
check would therefore break the plugin outright. So the token is ALSO accepted
as a leading path segment - set the plugin's Ollama host to
http://<ip>:11435/t/<token> and the plugin's own string concatenation
(`${host}${endpoint}`) puts it in front of every request for free. Rejected
alternative: a ?token= query string, which cannot work because the plugin
appends the endpoint path AFTER the host string.

Run locally to test:
    python3 brain_ollama_shim.py      # then: curl http://127.0.0.1:11435/api/tags
"""
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brain  # safe to import: main() sits behind an __main__ guard

HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "11435"))   # never 11434: real Ollama owns that
TOKEN = os.environ.get("SHIM_TOKEN", "")           # shared secret; "" disables the check
WEB_ENABLED = os.environ.get("SHIM_WEB", "0") == "1"
MAX_Q = int(os.environ.get("SHIM_MAX_Q", "500"))   # cap question length
MAX_BODY = int(os.environ.get("SHIM_MAX_BODY", "200000"))  # cap POST body bytes
QUEUE_TIMEOUT = float(os.environ.get("SHIM_QUEUE_TIMEOUT", "120"))  # deep model is slow
# Live token streaming. SHIM_STREAM_LIVE=0 falls back to the original behaviour
# (answer fully, then replay it as frames), which is one env var away if a
# client ever turns out to dislike a chunked response.
STREAM_LIVE = os.environ.get("SHIM_STREAM_LIVE", "1") not in ("0", "false", "False", "no")
# Retrieval runs for seconds before the first token exists. An empty-content
# frame every HEARTBEAT_S keeps the connection warm through that silence and is
# what makes a client disconnect visible while nothing is being generated.
HEARTBEAT_S = float(os.environ.get("SHIM_HEARTBEAT", "15"))
# Hard ceiling on one streamed answer. A generation wedged inside Ollama would
# otherwise hold the socket (and the GPU permit) forever.
STREAM_DEADLINE = float(os.environ.get("SHIM_STREAM_DEADLINE", "600"))
# Append the cited notes as an Obsidian [[wikilink]] block so an answer copied
# into a note is already linked into the graph. SHIM_LINKS=0 turns it off.
LINKS_ENABLED = os.environ.get("SHIM_LINKS", "1") not in ("0", "false", "False", "no")

# Model names the plugin's dropdown shows. Keeping them ":latest"-suffixed makes
# them look like ordinary Ollama tags, which is what the plugin (and the user
# reading the dropdown) expects.
FAST_NAME = os.environ.get("SHIM_FAST_NAME", "jetson-brain:latest")
DEEP_NAME = os.environ.get("SHIM_DEEP_NAME", "jetson-brain-deep:latest")

# Chunk size for replaying a finished answer as stream frames. Small enough that
# the plugin's renderer shows text arriving, large enough that a 2,000-character
# answer is ~25 frames rather than 2,000.
FRAME_CHARS = 80

# History cap. Smart Chat sends the whole thread every turn; brain.py's own
# HISTORY_TURNS is the authority on how much survives, but the prompt still gets
# built from what we pass, and Ollama truncates a prompt over num_ctx from the
# TOP (dropping the system prompt and the grounding contract). So hard-cap both
# the number of messages and their characters here, before brain.py ever sees
# them.
MAX_HISTORY_MSGS = brain.HISTORY_TURNS * 2
MAX_HISTORY_CHARS = int(os.environ.get("SHIM_MAX_HISTORY_CHARS", "4000"))

# Single-flight: one grounded answer at a time on one 8GB device. Concurrent
# requests queue behind the in-flight answer instead of thrashing / OOM-ing the
# GPU; past the timeout the caller gets a friendly message.
_gpu = threading.BoundedSemaphore(1)

# Index state. Loaded once in main() rather than at import time (advisor_server.py
# loads at import) so test_ollama_shim.py can import this module with `brain`
# stubbed out and no index on disk.
CHUNKS, BM25, META, INDEX_MTIME = [], None, {}, 0.0


def load_state():
    """Load the vault index once at startup."""
    global CHUNKS, BM25, META, INDEX_MTIME
    print("Loading the vault index...", flush=True)
    CHUNKS, BM25, META = brain.load_index()
    INDEX_MTIME = brain.index_mtime()
    print("Loaded %d chunks. fast=%s deep=%s web=%s" % (
        len(CHUNKS), brain.FAST_MODEL, brain.DEEP_MODEL, WEB_ENABLED), flush=True)


# ---------------------------------------------------------------------------
# Request shaping.
# ---------------------------------------------------------------------------
def resolve_model(name):
    """Map an advertised model name onto a real Ollama model.

    Permissive on purpose, exactly like brain.py's route parser: anything that
    is not recognizably the deep tier falls back to the safe default (fast). An
    unknown or missing model name must answer, not 404."""
    if "deep" in str(name or "").lower():
        return brain.DEEP_MODEL
    return brain.FAST_MODEL


def _flatten_content(content):
    """Smart Chat sends content as a string, but its own request adapter can
    build the OpenAI-style [{type:"text", text:...}] array (that is what
    _transform_content_to_ollama collapses). Accept both; ignore image parts,
    which this text-only RAG cannot use."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content or "")


def extract_question_and_history(messages):
    """(question, history) from an Ollama /api/chat messages array.

    The LAST user message is the question. Everything before it becomes history
    in brain.py's format (a list of {"role", "content"} with only user and
    assistant turns). System messages are dropped: brain.py builds its own
    system prompt and the grounding contract lives there, so letting the
    plugin's system message through would dilute a load-bearing prompt."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    if last_user < 0:
        return "", []
    question = _flatten_content(msgs[last_user].get("content"))[:MAX_Q].strip()

    history = []
    for m in msgs[:last_user]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _flatten_content(m.get("content")).strip()
        if text:
            history.append({"role": role, "content": text[:MAX_HISTORY_CHARS]})
    # Newest turns are the ones that resolve follow-ups ("what about her other
    # paper?"), so trim from the FRONT when over budget.
    history = history[-MAX_HISTORY_MSGS:]
    while history and sum(len(m["content"]) for m in history) > MAX_HISTORY_CHARS:
        history.pop(0)
    return question, history


# ---------------------------------------------------------------------------
# Answer shaping.
# ---------------------------------------------------------------------------
def wikilink_footer(sources):
    """An Obsidian-native sources block for the cited notes, so a pasted answer
    lands already linked into the graph. Vault notes become [[wikilinks]] built
    from the note name (Obsidian resolves the shortest unique path, so the
    folder prefix is noise); web fallback sources are bare URLs, which are not
    notes and would render as broken links. "" when there is nothing to cite."""
    lines, seen = [], set()
    for src in sources or []:
        s = str(src).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        if s.lower().startswith(("http://", "https://")):
            lines.append("- " + s)
            continue
        name = os.path.splitext(os.path.basename(s))[0]
        # "]]" inside a link target would close the wikilink early and corrupt
        # the rest of the note.
        name = " ".join(name.replace("[", "").replace("]", "").split())
        if name:
            lines.append("- [[" + name + "]]")
    if not lines:
        return ""
    return "\n\n**Sources**\n" + "\n".join(lines) + "\n"


def _safe_pieces(piece):
    r"""Split one piece of answer text so that no returned piece contains the
    substring `done_reason`.

    The plugin ends the stream at the first frame containing '"done_reason"'
    (SmartChatModelOllamaAdapter.is_end_of_stream). An answer that quotes that
    string - which a note about this very shim would - is JSON-encoded as
    \\"done_reason\\", and the substring check does not care about the escaping
    backslashes, so it would truncate the answer. Cutting the token across two
    frames makes that impossible without altering a single character of the
    answer the user sees.

    No carry buffer across calls is needed, and that is deliberate. Each NDJSON
    line is dispatched to the plugin as ONE message (SmartStreamer only
    dispatches a piece once it has seen the newline after it, which is why
    ndjson() insists on the trailing "\n"), and is_end_of_stream tests that
    single message. So a token ending in "done_" followed by a token starting
    with "reason" lands in two separate messages and cannot end the stream. The
    only real hazard is one frame carrying the whole substring, which is exactly
    what this function prevents."""
    out = []
    while True:
        at = piece.find("done_reason")
        if at < 0:
            break
        cut = at + len("done_")  # split inside the token, keep every character
        out.append(piece[:cut])
        piece = piece[cut:]
    if piece:
        out.append(piece)
    return out


def _content_frames(text, size=FRAME_CHARS):
    """Split answer text into stream-frame-sized pieces, each safe to send.

    Used for text that is already complete when we frame it: the blocking
    reply, the busy/error messages, and the post-stream delta. Live tokens go
    straight through _safe_pieces instead, one frame per token."""
    out = []
    for i in range(0, len(text), size):
        out.extend(_safe_pieces(text[i:i + size]))
    return out


def _now_iso():
    """RFC3339-ish timestamp. The plugin only stores created_at (it becomes the
    completion id), so exact nanosecond formatting does not matter."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _frame(model_name, created, content, done=False, **extra):
    """One Ollama chat frame. Shared by the live writer and chat_frames so the
    two can never drift apart in shape."""
    frame = {
        "model": model_name,
        "created_at": created,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    frame.update(extra)
    return frame


def _done_extra(answer, elapsed_ns, stats=None):
    """The final frame's counters, which the plugin turns into OpenAI usage
    (prompt_eval_count -> prompt_tokens, eval_count -> completion_tokens).

    `stats` is brain.LAST_CHAT_STATS, the REAL prefill/generation counters
    Ollama reported for this turn. It is only populated on paths that actually
    reached the model, so every field falls back to the old estimate (4
    characters per token, the same estimate the plugin's own count_tokens
    fallback uses) when it is missing."""
    stats = stats or {}
    return {
        "done_reason": "stop",
        "total_duration": stats.get("total_duration") or elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": stats.get("prompt_eval_count", 0),
        "prompt_eval_duration": 0,
        "eval_count": stats.get("eval_count") or max(1, len(answer) // 4),
        "eval_duration": elapsed_ns,
    }


def chat_frames(model_name, answer, elapsed_ns, stats=None):
    """The full NDJSON body for a /api/chat response whose answer is already
    complete: the blocking reply, and the SHIM_STREAM_LIVE=0 fallback."""
    created = _now_iso()
    frames = [_frame(model_name, created, piece)
              for piece in _content_frames(answer)]
    if not frames:
        # An empty stream would leave the plugin waiting on a message that never
        # carries a role; send one empty content frame so res_adapter has a role.
        frames.append(_frame(model_name, created, ""))
    frames.append(_frame(model_name, created, "", done=True,
                         **_done_extra(answer, elapsed_ns, stats)))
    return frames


def ndjson(frames):
    r"""Frames as newline-delimited JSON. The trailing "\n" matters: SmartStreamer
    only dispatches a piece once it has seen the separator after it."""
    return "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in frames)


def blocking_response(model_name, answer, elapsed_ns, stats=None):
    """The single-object body for {"stream": false}. The plugin's complete()
    path parses this with the same response adapter, so it is the final frame
    with the content filled in."""
    body = chat_frames(model_name, answer, elapsed_ns, stats)[-1]
    body["message"] = {"role": "assistant", "content": answer}
    return body


# ---------------------------------------------------------------------------
# Answering.
# ---------------------------------------------------------------------------
class _CaptureRun(brain._NullRun):
    """A UI Run that captures the answer instead of printing it.

    brain.handle_question drives a progress tree through this object, calling
    running/done/skip/fail/finish/sources/begin_answer on it unconditionally.
    Subclassing _NullRun rather than reimplementing the interface is what keeps
    that complete: every stage method it does not need already exists as a
    no-op, so a new stage added to brain.py cannot break the shim.

    Two overrides matter. begin_answer() is THE streaming seam: brain.py hands
    every generated token to the callable it returns (see chat_stream_holdback),
    so pushing those onto a queue is the whole of live streaming. sources() is
    overridden purely to stop _NullRun writing to stdout, where it would spam
    the journal - the shim gets the source list from handle_question's return
    value instead.

    With q=None it just accumulates, which is what the blocking path wants."""

    def __init__(self, q=None):
        self.q = q
        self.parts = []

    def _emit(self, piece):
        if not piece:
            return
        self.parts.append(piece)
        if self.q is not None:
            self.q.put(("tok", piece))

    def begin_answer(self):
        return self._emit

    def sources(self, srcs, vault_name=None):
        pass

    @property
    def text(self):
        """Everything brain.py streamed, in order. Includes brain.py's own
        inline notes ("[stream interrupted...]") and the grounding footer it
        pushes through the same sink - deliberately, because this string is
        what the streaming delta is measured against and dropping anything
        from it would double-emit that text."""
        return "".join(self.parts)


def answer_question(model_name, question, history, run):
    """Answer one question. Returns (answer, sources, stats), never raises, and
    always yields a non-empty answer.

    This is the whole GPU-serialized critical section, and it runs entirely on
    ONE thread - the caller's for the blocking path, the worker's for the
    streaming path. Acquiring and releasing _gpu in the same frame is what keeps
    the single-flight lock honest: an earlier design acquired on the request
    thread and released on the worker, which is legal for a BoundedSemaphore but
    leaves a permanently wedged device one early return away.

    brain.LAST_CHAT_STATS is a process global, not per-request, so it is read
    HERE, inside the lock, while nothing else can be generating. Reading it any
    later would attribute one question's token counts to another."""
    if not _gpu.acquire(timeout=QUEUE_TIMEOUT):
        # A busy device answers "try again" rather than piling a second 3B/7B
        # generation onto 8GB of shared memory.
        return ("The brain is busy with another question. Try again in a moment.",
                [], {})
    try:
        # Auto-learn hot-reload (safe here: the GPU lock makes this the only
        # in-flight question, so swapping the globals cannot race a
        # concurrent handle_question).
        global CHUNKS, BM25, META, INDEX_MTIME
        CHUNKS, BM25, META, INDEX_MTIME, _ = brain.maybe_reload_index(
            CHUNKS, BM25, META, INDEX_MTIME)
        answer, sources = brain.handle_question(
            resolve_model(model_name), question, history, BM25, CHUNKS,
            META, WEB_ENABLED, run)
        stats = dict(getattr(brain, "LAST_CHAT_STATS", None) or {})
    except Exception as exc:  # never let one bad question take the server down
        print("[shim] error:", exc, flush=True)
        answer, sources, stats = "The brain hit an error answering that.", [], {}
    finally:
        _gpu.release()

    if not answer:
        answer, sources = "The brain could not answer that from the vault.", []
    return answer, sources, stats


# ---------------------------------------------------------------------------
# Model advertisement.
# ---------------------------------------------------------------------------
def _tag(name, param_size, size_bytes, family):
    """One /api/tags entry. `name` is the only field parse_model_data reads, but
    the rest are what a real Ollama returns and cost nothing to supply, so a
    future plugin version that reads size/details keeps working."""
    return {
        "name": name,
        "model": name,
        "modified_at": _now_iso(),
        "size": size_bytes,
        # Fixed, obviously-synthetic digests: real ones would imply these tags
        # are pullable blobs, which they are not.
        "digest": "0" * 56,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": param_size,
            "quantization_level": "Q4_K_M",
        },
    }


def tags_payload():
    return {"models": [
        _tag(FAST_NAME, "3B", 2019393189, "llama"),
        _tag(DEEP_NAME, "7B", 4683087519, "qwen2"),
    ]}


def show_payload(name):
    """Per-model metadata. parse_model_data() looks for the first model_info key
    containing ".context_length" and uses it as max_input_tokens, so report the
    REAL window brain.py generates with (GEN_OPTIONS num_ctx, 8192 today, raised
    from 6144 on measurement - so read the dict, never a literal). Claiming
    128k here would let the plugin pack a prompt that Ollama then truncates from
    the top, silently dropping the system prompt and the grounding contract."""
    deep = "deep" in str(name or "").lower()
    family = "qwen2" if deep else "llama"
    num_ctx = brain.GEN_OPTIONS.get("num_ctx", 8192)
    return {
        "license": "",
        "modelfile": "# jetson-brain RAG (brain.py), served over the Ollama API",
        "parameters": "num_ctx %d" % num_ctx,
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "7B" if deep else "3B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": family,
            "general.parameter_count": 7000000000 if deep else 3000000000,
            family + ".context_length": num_ctx,
        },
        "capabilities": ["completion"],
    }


# ---------------------------------------------------------------------------
# HTTP.
# ---------------------------------------------------------------------------
def strip_token_prefix(path):
    """("/api/chat", "<token>") for "/t/<token>/api/chat"; (path, "") otherwise.

    The plugin builds every URL as `${host}${endpoint}`, so a token parked in
    the host setting arrives as a leading path segment on every request. See the
    module docstring for why a header-only check cannot work here.

    That same concatenation is why the leading slashes are collapsed first: a
    trailing "/" in the plugin's Ollama host field (a very ordinary paste) makes
    every URL "//api/tags", which matched no route and 404'd a perfectly healthy
    shim. Normalizing here rather than at the five route comparisons keeps it to
    one place, and it only touches the host/endpoint join point."""
    path = "/" + path.lstrip("/")
    if path.startswith("/t/"):
        rest = path[3:]
        slash = rest.find("/")
        if slash >= 0:
            return "/" + rest[slash:].lstrip("/"), rest[:slash]
        return "/", rest
    return path, ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def _cors(self):
        """Obsidian's renderer is origin app://obsidian.md, so every response
        (and a preflight) needs CORS or the XHR never reaches this code. Real
        Ollama does the same via its default OLLAMA_ORIGINS allowlist."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _send_raw(self, code, body, content_type, close=False):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if close:
            # Set close on any reply sent BEFORE the request body was read.
            # protocol_version is HTTP/1.1, so the socket would otherwise stay
            # keep-alive with an undrained body in it, and handle_one_request
            # would parse that JSON as the next request line - poisoning the
            # NEXT legitimate chat on the same pooled Chromium connection.
            # Rejected: draining min(Content-Length, MAX_BODY) - that reads a
            # body we just refused as oversized. send_header() also flips
            # self.close_connection here, so this is the whole fix.
            self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (or the tunnel) hung up; nothing to do

    def _send(self, code, payload, close=False):
        self._send_raw(code, json.dumps(payload, ensure_ascii=False),
                       "application/json", close=close)

    def _send_ndjson(self, frames):
        # Content-Length is correct HERE because this path frames an answer that
        # is already complete: the empty-question reply, and the
        # SHIM_STREAM_LIVE=0 fallback. The live path below cannot know its
        # length up front and uses chunked instead. SmartStreamer reads
        # xhr.responseText, which works either way.
        self._send_raw(200, ndjson(frames), "application/x-ndjson")

    # -- live streaming ---------------------------------------------------
    def _open_stream(self):
        """Headers for a chunked NDJSON stream.

        Chunked rather than Content-Length (unknown up front) and rather than
        Connection: close (which would drop Chromium's pooled connection on
        every single chat, the very connection the close=True guard above works
        to keep clean). _send_raw cannot be reused because it commits a length,
        so _cors() has to be called BY HAND here - without it Obsidian's
        app://obsidian.md XHR fails before reading a byte, and SmartStreamer
        reports that as a bare "Stream error"."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

    def _write_frame(self, frame):
        """One frame as one HTTP chunk. False means the client is gone.

        The whole frame is encoded before being sized, so a chunk boundary can
        never fall inside a multi-byte character."""
        body = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _end_stream(self, gone):
        """The terminating zero-length chunk.

        Skipped entirely when the client is already gone: writing a terminator
        to a dead socket is how a cleanup path raises a SECOND exception and
        takes the handler thread down with it. A half-written stream also
        cannot be reused, so the connection is closed rather than pooled."""
        if gone:
            self.close_connection = True
            return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    def log_message(self, *args):
        pass  # quiet the default per-request stderr line

    def _read_json(self):
        """Read and parse a JSON body, or return None and send the error itself."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            # Answered without reading the body (there may be a whole oversized
            # POST, or a chunked one Content-Length cannot describe, still in
            # the socket), so this reply must close the connection.
            self._send(400, {"error": "bad request"}, close=True)
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send(400, {"error": "invalid json"})
            return None

    def _authorized(self, url_token):
        """Empty SHIM_TOKEN disables the check entirely (the localhost default).
        Otherwise accept the token from the URL prefix (what the plugin can
        actually send) or from a bearer header (what curl and the systemd
        health check can send)."""
        if not TOKEN:
            return True
        if url_token and url_token == TOKEN:
            return True
        return self.headers.get("Authorization", "") == "Bearer " + TOKEN

    # -- routes -----------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path, url_token = strip_token_prefix(self.path.split("?", 1)[0])
        if not self._authorized(url_token):
            return self._send(401, {"error": "unauthorized"})
        if path == "/":
            # Byte-for-byte what real Ollama answers, because that string is the
            # standard "is it up?" check people (and docs) reach for.
            return self._send_raw(200, "Ollama is running", "text/plain; charset=utf-8")
        if path == "/api/version":
            return self._send(200, {"version": "0.0.0-jetson-brain-shim"})
        if path == "/api/tags":
            return self._send(200, tags_payload())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path, url_token = strip_token_prefix(self.path.split("?", 1)[0])
        if not self._authorized(url_token):
            # Same desync as the 400 above: the POST body is still unread.
            return self._send(401, {"error": "unauthorized"}, close=True)
        if path == "/api/show":
            data = self._read_json()
            if data is None:
                return
            return self._send(200, show_payload(data.get("model", "")))
        if path == "/api/chat":
            return self._handle_chat()
        return self._send(404, {"error": "not found"}, close=True)  # body unread

    def _handle_chat(self):
        data = self._read_json()
        if data is None:
            return  # _read_json already sent the error
        model_name = str(data.get("model") or FAST_NAME)
        stream = data.get("stream", True) is not False
        question, history = extract_question_and_history(data.get("messages"))
        if not question:
            return self._reply(model_name, stream,
                               "I did not get a question to answer.", 0)

        t0 = time.perf_counter()
        if stream and STREAM_LIVE:
            return self._stream_chat(model_name, question, history, t0)

        answer, sources, stats = answer_question(
            model_name, question, history, _CaptureRun())
        if LINKS_ENABLED:
            answer = answer + wikilink_footer(sources)
        return self._reply(model_name, stream, answer,
                           int((time.perf_counter() - t0) * 1e9), stats)

    def _stream_chat(self, model_name, question, history, t0):
        """Write NDJSON frames as brain.py generates tokens.

        The answer runs on a worker thread and pushes tokens onto a queue; this
        thread does nothing but drain the queue and write. The worker owns the
        GPU permit for its whole life, so this thread must keep draining to the
        sentinel even after the client hangs up - abandoning the queue would
        leave a generation running with nobody to finish it."""
        q = queue.Queue(maxsize=1024)
        run = _CaptureRun(q)
        box = {}

        def worker():
            try:
                box["result"] = answer_question(model_name, question, history, run)
            finally:
                q.put(("done", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        created = _now_iso()
        gone = False
        wrote = False
        self._open_stream()
        try:
            deadline = time.monotonic() + STREAM_DEADLINE
            while True:
                try:
                    kind, payload = q.get(timeout=HEARTBEAT_S)
                except queue.Empty:
                    if time.monotonic() > deadline:
                        # The worker still holds the GPU permit, so every later
                        # question gets the friendly "busy" text until this
                        # process restarts. Correct, but indistinguishable from
                        # a broken shim, so say so loudly.
                        print("[shim] stream deadline hit with the worker still "
                              "running; the GPU permit is held until it ends",
                              flush=True)
                        break
                    # Silence here is retrieval, not a stall. An empty content
                    # frame costs the plugin nothing (it appends "") and is what
                    # surfaces a disconnect while nothing is generating.
                    if not gone:
                        gone = not self._write_frame(_frame(model_name, created, ""))
                    continue
                if kind == "done":
                    break
                if gone:
                    continue  # keep draining so the worker can finish
                for piece in _safe_pieces(payload):
                    if not self._write_frame(_frame(model_name, created, piece)):
                        gone = True
                        break
                    wrote = True

            answer, sources, stats = box.get("result") or (
                "The brain could not answer that from the vault.", [], {})
            footer = wikilink_footer(sources) if LINKS_ENABLED else ""
            streamed = run.text
            if not streamed:
                # Nothing streamed: the holdback refused, the blocking
                # fall-through ran, or the persona never streams (scholar). Frame
                # the whole answer, exactly as the pre-streaming shim did.
                tail = answer
            elif answer.startswith(streamed):
                tail = answer[len(streamed):]
            else:
                # The user has already watched `streamed` arrive; re-sending the
                # answer would duplicate it. Say so rather than dropping text
                # invisibly.
                print("[shim] streamed text diverged from the returned answer; "
                      "sending only the sources footer", flush=True)
                tail = ""
            for piece in _content_frames(tail + footer):
                if gone:
                    break
                if not self._write_frame(_frame(model_name, created, piece)):
                    gone = True
                    break
                wrote = True
            if not wrote and not gone:
                # The plugin needs at least one message carrying a role.
                gone = not self._write_frame(_frame(model_name, created, ""))
            if not gone:
                elapsed_ns = int((time.perf_counter() - t0) * 1e9)
                self._write_frame(_frame(model_name, created, "", done=True,
                                         **_done_extra(answer, elapsed_ns, stats)))
        except Exception as exc:
            # The status line is already sent, so this cannot become a 4xx. All
            # that is left is to close the stream cleanly.
            print("[shim] stream error:", exc, flush=True)
        finally:
            self._end_stream(gone)

    def _reply(self, model_name, stream, answer, elapsed_ns, stats=None):
        """Always 200, even for the busy/error paths. SmartStreamer treats any
        other status as a stream error and the user sees nothing but a console
        message, so the friendly text has to ride the normal answer channel."""
        if stream:
            return self._send_ndjson(
                chat_frames(model_name, answer, elapsed_ns, stats))
        return self._send(200,
                          blocking_response(model_name, answer, elapsed_ns, stats))


def main():
    load_state()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("ollama shim listening on http://%s:%d  (GET /api/tags, POST /api/chat)"
          % (HOST, PORT), flush=True)
    print("Smart Chat -> Ollama host: http://%s:%d%s"
          % (HOST, PORT, ("/t/" + TOKEN) if TOKEN else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
