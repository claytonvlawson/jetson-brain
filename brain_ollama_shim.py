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
    POST /api/chat        -> NDJSON stream of Ollama chat frames, or one JSON
                             object when {"stream": false}

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
# built from what we pass, and num_ctx is 6144 where Ollama truncates from the
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


def _content_frames(text, size=FRAME_CHARS):
    r"""Split answer text into stream-frame-sized pieces such that no piece
    contains the substring `done_reason`.

    The plugin ends the stream at the first frame containing '"done_reason"'
    (SmartChatModelOllamaAdapter.is_end_of_stream). An answer that quotes that
    string - which a note about this very shim would - is JSON-encoded as
    \\"done_reason\\", and the substring check does not care about the escaping
    backslashes, so it would truncate the answer. Cutting the token across two
    frames makes that impossible without altering a single character of the
    answer the user sees."""
    out = []
    for i in range(0, len(text), size):
        piece = text[i:i + size]
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


def _now_iso():
    """RFC3339-ish timestamp. The plugin only stores created_at (it becomes the
    completion id), so exact nanosecond formatting does not matter."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def chat_frames(model_name, answer, elapsed_ns):
    """The full NDJSON body for a streaming /api/chat response.

    Frame shape mirrors real Ollama: content frames carry
    message.content + done:false, and one final frame carries done_reason,
    done:true and the eval counters the plugin turns into OpenAI usage
    (prompt_eval_count -> prompt_tokens, eval_count -> completion_tokens).
    Token counts are approximated at 4 characters per token, the same estimate
    the plugin's own count_tokens fallbacks use."""
    created = _now_iso()
    frames = []
    for piece in _content_frames(answer):
        frames.append({
            "model": model_name,
            "created_at": created,
            "message": {"role": "assistant", "content": piece},
            "done": False,
        })
    if not frames:
        # An empty stream would leave the plugin waiting on a message that never
        # carries a role; send one empty content frame so res_adapter has a role.
        frames.append({
            "model": model_name,
            "created_at": created,
            "message": {"role": "assistant", "content": ""},
            "done": False,
        })
    frames.append({
        "model": model_name,
        "created_at": created,
        "message": {"role": "assistant", "content": ""},
        "done_reason": "stop",
        "done": True,
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": max(1, len(answer) // 4),
        "eval_duration": elapsed_ns,
    })
    return frames


def ndjson(frames):
    r"""Frames as newline-delimited JSON. The trailing "\n" matters: SmartStreamer
    only dispatches a piece once it has seen the separator after it."""
    return "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in frames)


def blocking_response(model_name, answer, elapsed_ns):
    """The single-object body for {"stream": false}. The plugin's complete()
    path parses this with the same response adapter, so it is the final frame
    with the content filled in."""
    body = chat_frames(model_name, answer, elapsed_ns)[-1]
    body["message"] = {"role": "assistant", "content": answer}
    return body


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
    REAL window brain.py generates with (GEN_OPTIONS num_ctx = 6144). Claiming
    128k here would let the plugin pack a prompt that Ollama then truncates from
    the top, silently dropping the system prompt and the grounding contract."""
    deep = "deep" in str(name or "").lower()
    family = "qwen2" if deep else "llama"
    num_ctx = brain.GEN_OPTIONS.get("num_ctx", 6144)
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
        # Content-Length rather than chunked transfer encoding: the answer is
        # already complete by the time we frame it (handle_question returns the
        # whole string), so there is nothing left to stream incrementally and a
        # known length is one less thing to get wrong. SmartStreamer reads
        # xhr.responseText, which works either way.
        self._send_raw(200, ndjson(frames), "application/x-ndjson")

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
        # A busy device answers "try again" rather than piling a second 3B/7B
        # generation onto 8GB of shared memory.
        if not _gpu.acquire(timeout=QUEUE_TIMEOUT):
            return self._reply(model_name, stream,
                               "The brain is busy with another question. Try again in a moment.",
                               int((time.perf_counter() - t0) * 1e9))
        try:
            # Auto-learn hot-reload (safe here: the GPU lock makes this the only
            # in-flight question, so swapping the globals cannot race a
            # concurrent handle_question).
            global CHUNKS, BM25, META, INDEX_MTIME
            CHUNKS, BM25, META, INDEX_MTIME, _ = brain.maybe_reload_index(
                CHUNKS, BM25, META, INDEX_MTIME)
            answer, sources = brain.handle_question(
                resolve_model(model_name), question, history, BM25, CHUNKS,
                META, WEB_ENABLED, brain._NullRun())
        except Exception as exc:  # never let one bad question take the server down
            print("[shim] error:", exc, flush=True)
            answer, sources = "The brain hit an error answering that.", []
        finally:
            _gpu.release()

        if not answer:
            answer, sources = "The brain could not answer that from the vault.", []
        elif LINKS_ENABLED:
            answer = answer + wikilink_footer(sources)
        return self._reply(model_name, stream, answer,
                           int((time.perf_counter() - t0) * 1e9))

    def _reply(self, model_name, stream, answer, elapsed_ns):
        """Always 200, even for the busy/error paths. SmartStreamer treats any
        other status as a stream error and the user sees nothing but a console
        message, so the friendly text has to ride the normal answer channel."""
        if stream:
            return self._send_ndjson(chat_frames(model_name, answer, elapsed_ns))
        return self._send(200, blocking_response(model_name, answer, elapsed_ns))


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
