#!/usr/bin/env python3
"""Offline correctness test for brain_ollama_shim.py -- the Ollama-API shim that
lets Obsidian's Smart Chat Pro plugin talk to the jetson-brain RAG unpatched.

`brain` is replaced with a stub module BEFORE importing the shim, following the
monkey-patching convention the other tests here use (test_hybrid.py,
test_obsidian_expand.py), so this runs with no Ollama, no index on disk, and no
external network. The server itself is real: it is started on a loopback port
and driven with urllib, because the whole point of this module is its wire
format and a hand-rolled fake request would only test my own assumptions.
"""
import json
import sys
import threading
import types
import urllib.error
import urllib.request

sys.path.insert(0, "/Users/you.lawson/jetson-brain")

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# ---- stub `brain` before the shim imports it -------------------------------
CALLS = []           # every handle_question call, for assertions
ANSWER = ["Hall (2017) argues that language deprivation is a public health crisis."]
SOURCES = [["Sources/Hall 2017.md", "Concepts/Language Deprivation.md",
            "https://example.org/paper"]]

_brain = types.ModuleType("brain")
_brain.HISTORY_TURNS = 6
_brain.FAST_MODEL = "llama3.2:3b"
_brain.DEEP_MODEL = "qwen2.5:7b"
_brain.GEN_OPTIONS = {"num_ctx": 6144}
_brain.load_index = lambda: ([], None, {})
_brain.index_mtime = lambda: 0.0
_brain.maybe_reload_index = lambda c, b, m, t: (c, b, m, t, False)
# Per-turn Ollama counters. Real brain.py sets these from the final stream part;
# the shim reads them for the done frame's usage numbers. Present here so the
# real code path is exercised rather than only its getattr fallback.
_brain.LAST_CHAT_STATS = {}
_brain.LAST_TIMING = {}


def _handle_question(model, question, history, bm25, chunks, meta, web, run):
    CALLS.append({"model": model, "question": question, "history": history,
                  "web": web})
    return ANSWER[0], SOURCES[0]


_brain.handle_question = _handle_question


class _NullRun:
    """Matches real brain.py: every stage method is a no-op and begin_answer
    returns the token sink. The shim's _CaptureRun subclasses this, so the
    stub has to expose the same surface or the subclass is not representative."""

    def running(self, *a, **k):
        pass

    def done(self, *a, **k):
        pass

    def skip(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass

    def begin_answer(self):
        return lambda piece: None

    def sources(self, srcs, vault_name=None):
        pass


_brain._NullRun = _NullRun
sys.modules["brain"] = _brain

import brain_ollama_shim as shim  # noqa: E402  (must follow the stub install)


# ---- request parsing: question and history ---------------------------------
q, h = shim.extract_question_and_history([
    {"role": "system", "content": "You are Smart Chat."},
    {"role": "user", "content": "Who is Wyatte Hall?"},
    {"role": "assistant", "content": "A researcher."},
    {"role": "user", "content": "What did he publish?"},
])
check("question is the LAST user message", q == "What did he publish?")
check("history is the earlier user+assistant turns in brain.py's format",
      h == [{"role": "user", "content": "Who is Wyatte Hall?"},
            {"role": "assistant", "content": "A researcher."}])

q, h = shim.extract_question_and_history([
    {"role": "system", "content": "SYSTEM PROMPT THAT MUST NOT LEAK"},
    {"role": "user", "content": "hi"},
])
check("system messages are dropped (brain.py owns the grounding contract)",
      h == [] and q == "hi")

q, h = shim.extract_question_and_history([
    {"role": "user", "content": [{"type": "text", "text": "part one"},
                                 {"type": "image_url",
                                  "image_url": {"url": "data:image/png;base64,AAA"}},
                                 {"type": "text", "text": "part two"}]},
])
check("array content is flattened to text and image parts are ignored",
      q == "part one\npart two")

q, h = shim.extract_question_and_history([{"role": "assistant", "content": "orphan"}])
check("no user message yields an empty question, not a crash", q == "" and h == [])

q, h = shim.extract_question_and_history(None)
check("missing messages array yields an empty question", q == "" and h == [])

long_q = "x" * (shim.MAX_Q + 500)
q, _ = shim.extract_question_and_history([{"role": "user", "content": long_q}])
check("question is capped at MAX_Q (num_ctx 6144 truncates from the TOP)",
      len(q) == shim.MAX_Q)

many = [{"role": "user", "content": "u%d" % i} for i in range(40)]
many.append({"role": "user", "content": "final"})
q, h = shim.extract_question_and_history(many)
check("history is capped at MAX_HISTORY_MSGS, newest kept",
      len(h) == shim.MAX_HISTORY_MSGS and h[-1]["content"] == "u39")

big = [{"role": "user", "content": "y" * 3000},
       {"role": "assistant", "content": "z" * 3000},
       {"role": "user", "content": "final"}]
q, h = shim.extract_question_and_history(big)
check("history is capped in CHARACTERS too",
      sum(len(m["content"]) for m in h) <= shim.MAX_HISTORY_CHARS)


# ---- model mapping ---------------------------------------------------------
check("deep name maps to brain.DEEP_MODEL",
      shim.resolve_model("jetson-brain-deep:latest") == "qwen2.5:7b")
check("fast name maps to brain.FAST_MODEL",
      shim.resolve_model("jetson-brain:latest") == "llama3.2:3b")
check("unknown model falls back to fast, permissively",
      shim.resolve_model("some-other-model") == "llama3.2:3b")
check("missing model falls back to fast", shim.resolve_model(None) == "llama3.2:3b")


# ---- wikilink footer -------------------------------------------------------
foot = shim.wikilink_footer(["Sources/Hall 2017.md", "Sources/Hall 2017.md",
                             "https://example.org/x"])
check("footer wikilinks the note by name, no folder, no extension",
      "[[Hall 2017]]" in foot)
check("footer de-duplicates repeated sources", foot.count("[[Hall 2017]]") == 1)
check("footer leaves a web source as a bare URL, not a broken wikilink",
      "- https://example.org/x" in foot and "[[https" not in foot)
check("footer is empty when nothing was cited", shim.wikilink_footer([]) == "")
check("footer strips brackets that would close the wikilink early",
      "[[bad name]]" in shim.wikilink_footer(["bad ]] name.md"]))


# ---- frame shaping ---------------------------------------------------------
frames = shim.chat_frames("jetson-brain:latest", "hello world", 1234)
check("every content frame carries done:false",
      all(f["done"] is False for f in frames[:-1]))
check("final frame carries done:true", frames[-1]["done"] is True)
check("final frame carries done_reason (the plugin's ONLY end-of-stream signal)",
      frames[-1].get("done_reason") == "stop")
check("final frame carries the eval counters the plugin maps to usage",
      "eval_count" in frames[-1] and "prompt_eval_count" in frames[-1])
check("content frames reassemble to exactly the answer",
      "".join(f["message"]["content"] for f in frames) == "hello world")
check("every frame names the model and a role",
      all(f["model"] == "jetson-brain:latest"
          and f["message"]["role"] == "assistant" for f in frames))

body = shim.ndjson(frames)
check("NDJSON is one JSON object per line, newline-terminated",
      body.endswith("\n") and len(body.strip().split("\n")) == len(frames))
check("each NDJSON line parses on its own",
      all(json.loads(l) for l in body.strip().split("\n")))
check("no content line contains the end-of-stream substring",
      all('"done_reason"' not in l for l in body.strip().split("\n")[:-1]))

# The nasty case: an answer that quotes the sentinel would end the stream early.
tricky = 'The final frame sets "done_reason": "stop" and then stops.'
tframes = shim.chat_frames("m", tricky, 1)
tbody = shim.ndjson(tframes)
check("an answer quoting \"done_reason\" does not truncate the stream",
      all("done_reason" not in l for l in tbody.strip().split("\n")[:-1]))
check("...and the answer text survives that split byte for byte",
      "".join(f["message"]["content"] for f in tframes) == tricky)

empty = shim.chat_frames("m", "", 1)
check("an empty answer still sends a content frame plus the done frame",
      len(empty) == 2 and empty[-1]["done"] is True)

single = shim.blocking_response("m", "hello world", 7)
check("stream:false body is one object with the WHOLE answer",
      single["message"]["content"] == "hello world" and single["done"] is True)
check("stream:false body still carries done_reason", single["done_reason"] == "stop")


# ---- token path prefix -----------------------------------------------------
check("token prefix is stripped off the path",
      shim.strip_token_prefix("/t/abc/api/chat") == ("/api/chat", "abc"))
check("token-only path resolves to /",
      shim.strip_token_prefix("/t/abc") == ("/", "abc"))
check("an ordinary path is untouched",
      shim.strip_token_prefix("/api/tags") == ("/api/tags", ""))
# A trailing slash in the plugin's "Ollama host" field makes every URL
# "${host}//api/...", which used to 404 a perfectly healthy shim.
check("a doubled join slash is collapsed",
      shim.strip_token_prefix("//api/tags") == ("/api/tags", ""))
check("a doubled join slash is collapsed after a token prefix",
      shim.strip_token_prefix("/t/abc//api/chat") == ("/api/chat", "abc"))


# ---- live server on loopback ----------------------------------------------
from http.server import ThreadingHTTPServer  # noqa: E402

server = ThreadingHTTPServer(("127.0.0.1", 0), shim.Handler)
BASE = "http://127.0.0.1:%d" % server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()


def get(path, headers=None):
    req = urllib.request.Request(BASE + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode("utf-8"), r.headers


def post(path, payload, headers=None):
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"), headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8"), r.headers


status, text, hdrs = get("/")
check("GET / answers the real Ollama probe string",
      status == 200 and text == "Ollama is running")
check("responses carry CORS (Obsidian's renderer is a cross-origin caller)",
      hdrs.get("Access-Control-Allow-Origin") == "*")
check("responses carry Content-Length",
      hdrs.get("Content-Length") == str(len(text)))

status, text, _ = get("/api/version")
check("GET /api/version answers a version", status == 200 and "version" in json.loads(text))

status, text, _ = get("/api/tags")
tags = json.loads(text)
names = [m["name"] for m in tags["models"]]
check("GET /api/tags advertises both models",
      names == ["jetson-brain:latest", "jetson-brain-deep:latest"])
check("each tag has the fields a real Ollama sends",
      all({"name", "model", "size", "digest", "modified_at", "details"} <= set(m)
          for m in tags["models"]))
check("no advertised name contains 'embed' (the plugin drops those)",
      all("embed" not in n for n in names))

status, text, _ = post("/api/show", {"model": "jetson-brain:latest"})
show = json.loads(text)
ctx_keys = [k for k in show["model_info"] if ".context_length" in k]
check("POST /api/show returns a model_info *.context_length key", len(ctx_keys) == 1)
check("...reporting the REAL num_ctx, not an invented 128k",
      show["model_info"][ctx_keys[0]] == 6144)

status, text, hdrs = post("/api/chat", {
    "model": "jetson-brain-deep:latest",
    "messages": [{"role": "user", "content": "Who is Wyatte Hall?"}],
    "stream": True,
})
lines = text.strip().split("\n")
check("POST /api/chat streams NDJSON with 200", status == 200)
check("the deep model name routed to the deep Ollama model",
      CALLS[-1]["model"] == "qwen2.5:7b")
check("the question reached brain.handle_question",
      CALLS[-1]["question"] == "Who is Wyatte Hall?")
check("web fallback is off by default inside Obsidian", CALLS[-1]["web"] is False)
streamed = "".join(json.loads(l)["message"]["content"] for l in lines)
check("streamed answer starts with the model's answer",
      streamed.startswith(ANSWER[0]))
check("streamed answer ends with the wikilink footer",
      "[[Hall 2017]]" in streamed and "[[Language Deprivation]]" in streamed)
check("last streamed line is the done frame", json.loads(lines[-1])["done"] is True)

status, text, _ = post("/api/chat", {
    "model": "jetson-brain:latest",
    "messages": [{"role": "user", "content": "again"}],
    "stream": False,
})
one = json.loads(text)
check("stream:false returns a single JSON object, not NDJSON",
      "\n" not in text.strip() and one["done"] is True)
check("stream:false content is the full answer plus footer",
      one["message"]["content"].startswith(ANSWER[0]) and "[[Hall 2017]]" in one["message"]["content"])
check("the fast model name routed to the fast Ollama model",
      CALLS[-1]["model"] == "llama3.2:3b")

shim.LINKS_ENABLED = False
status, text, _ = post("/api/chat", {"messages": [{"role": "user", "content": "q"}],
                                     "stream": False})
check("SHIM_LINKS=0 suppresses the wikilink footer",
      json.loads(text)["message"]["content"] == ANSWER[0])
shim.LINKS_ENABLED = True

# A raising handle_question must still produce a well-formed 200 stream, because
# SmartStreamer turns any other status into a bare console error.
def _boom(*args, **kwargs):
    raise RuntimeError("ollama is down")


_brain.handle_question = _boom
status, text, _ = post("/api/chat", {"messages": [{"role": "user", "content": "q"}],
                                     "stream": True})
check("a raising brain still answers 200 with a valid done frame",
      status == 200 and json.loads(text.strip().split("\n")[-1])["done"] is True)
_brain.handle_question = _handle_question


# ---- live token streaming --------------------------------------------------
# The stubs above never call run.begin_answer(), so they exercise the "nothing
# streamed" fallback. These call it for real, which is the actual streaming
# path: brain.py hands every token to the sink begin_answer() returns.
#
# Every check here is wrapped, and every read is bounded by a timeout. This file
# is straight-line module-scope asserts with no isolation, so an uncaught
# exception would abort it and silently skip every later check - and a streaming
# read that blocks forever would wedge the run instead of failing it.
def stream_post(payload, timeout=20):
    req = urllib.request.Request(
        BASE + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def content_of(lines):
    return "".join(json.loads(l)["message"]["content"] for l in lines)


TOKENS = ["Hall (2017) ", "argues that ", "language deprivation ",
          "is a public health crisis."]
GATE = threading.Event()


def _streaming_question(model, question, history, bm25, chunks, meta, web, run):
    sink = run.begin_answer()
    sink(TOKENS[0])
    GATE.wait(timeout=10)   # held until the test has READ that first token
    for tok in TOKENS[1:]:
        sink(tok)
    _brain.LAST_CHAT_STATS = {"prompt_eval_count": 321, "eval_count": 77,
                              "total_duration": 12345}
    return "".join(TOKENS), SOURCES[0]


_brain.handle_question = _streaming_question
try:
    GATE.clear()
    resp = stream_post({"messages": [{"role": "user", "content": "q"}],
                        "stream": True})
    first = resp.readline().decode("utf-8")
    early = json.loads(first)
    # The proof that this is a live stream and not a replay: the first frame is
    # readable while the stub is still blocked, so the answer does not exist yet.
    check("a token frame arrives BEFORE the answer is finished",
          early["message"]["content"] == TOKENS[0] and early["done"] is False)
    GATE.set()
    lines = [first] + [l for l in resp.read().decode("utf-8").split("\n")
                       if l.strip()]
    resp.close()
    check("live frames reassemble to the streamed tokens plus the footer",
          content_of(lines) == "".join(TOKENS) + shim.wikilink_footer(SOURCES[0]))
    check("one frame per token, not one frame for the whole answer",
          len(lines) > len(TOKENS))
    check("the live stream ends with a done frame",
          json.loads(lines[-1])["done"] is True)
    check("no live content line carries the end-of-stream substring",
          all("done_reason" not in l for l in lines[:-1]))
    check("the done frame reports the REAL counters from brain.LAST_CHAT_STATS",
          json.loads(lines[-1])["eval_count"] == 77
          and json.loads(lines[-1])["prompt_eval_count"] == 321)
except Exception as exc:
    check("live token streaming works (raised %r)" % (exc,), False)
finally:
    GATE.set()
    _brain.LAST_CHAT_STATS = {}


# A single TOKEN that is the sentinel. Cross-token splits are safe by
# construction (each NDJSON line is its own message), but one frame carrying the
# whole substring would truncate the answer.
SENTINEL_TEXT = 'The final frame sets "done_reason": "stop" and stops.'


def _sentinel_question(model, question, history, bm25, chunks, meta, web, run):
    run.begin_answer()(SENTINEL_TEXT)
    return SENTINEL_TEXT, []


_brain.handle_question = _sentinel_question
try:
    status, text, _ = post("/api/chat", {"messages": [{"role": "user", "content": "q"}],
                                         "stream": True})
    lines = [l for l in text.split("\n") if l.strip()]
    check("a streamed token quoting \"done_reason\" does not truncate the stream",
          all("done_reason" not in l for l in lines[:-1]))
    check("...and that token's text survives the split byte for byte",
          content_of(lines) == SENTINEL_TEXT)
except Exception as exc:
    check("a streamed done_reason token is split safely (raised %r)" % (exc,), False)


# Tokens already went out, so the response cannot become a 4xx: the failure has
# to ride out as content plus a normal done frame.
def _boom_midstream(model, question, history, bm25, chunks, meta, web, run):
    run.begin_answer()("partial answer so far ")
    raise RuntimeError("ollama died mid-generation")


_brain.handle_question = _boom_midstream
try:
    status, text, _ = post("/api/chat", {"messages": [{"role": "user", "content": "q"}],
                                         "stream": True})
    lines = [l for l in text.split("\n") if l.strip()]
    check("a brain that raises MID-STREAM still ends 200 with a done frame",
          status == 200 and json.loads(lines[-1])["done"] is True)
    check("...and keeps the tokens the user already watched arrive",
          "partial answer so far " in content_of(lines))
    check("...without re-sending a diverged answer on top of them",
          content_of(lines).count("partial answer so far") == 1)
except Exception as exc:
    check("a mid-stream failure closes the stream cleanly (raised %r)" % (exc,), False)

_brain.handle_question = _handle_question


# The escape hatch: SHIM_STREAM_LIVE=0 restores the pre-streaming behaviour.
shim.STREAM_LIVE = False
try:
    status, text, hdrs = post("/api/chat", {"messages": [{"role": "user", "content": "q"}],
                                            "stream": True})
    lines = [l for l in text.split("\n") if l.strip()]
    check("SHIM_STREAM_LIVE=0 still answers a well-formed NDJSON stream",
          status == 200 and json.loads(lines[-1])["done"] is True
          and content_of(lines).startswith(ANSWER[0]))
    check("...with a Content-Length, not chunked",
          hdrs.get("Content-Length") is not None)
except Exception as exc:
    check("SHIM_STREAM_LIVE=0 falls back cleanly (raised %r)" % (exc,), False)
shim.STREAM_LIVE = True


# ---- auth ------------------------------------------------------------------
check("empty token disables the check entirely", shim.TOKEN == "")
shim.TOKEN = "s3cret"
try:
    get("/api/tags")
    check("a tokened server rejects an unauthenticated request", False)
except urllib.error.HTTPError as exc:
    check("a tokened server rejects an unauthenticated request", exc.code == 401)

status, _, _ = get("/api/tags", headers={"Authorization": "Bearer s3cret"})
check("bearer header authenticates (curl and health checks)", status == 200)

status, _, _ = get("/t/s3cret/api/tags")
check("URL path prefix authenticates (the only form the plugin can send)",
      status == 200)

try:
    get("/t/wrong/api/tags")
    check("a wrong path token is rejected", False)
except urllib.error.HTTPError as exc:
    check("a wrong path token is rejected", exc.code == 401)

status, _, _ = post("/t/s3cret/api/chat",
                    {"messages": [{"role": "user", "content": "q"}], "stream": False})
check("the chat endpoint accepts the path-prefixed token too", status == 200)
shim.TOKEN = ""


# ---- malformed input -------------------------------------------------------
try:
    req = urllib.request.Request(BASE + "/api/chat", data=b"not json",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    check("a non-JSON body is a 400, not a crash", False)
except urllib.error.HTTPError as exc:
    check("a non-JSON body is a 400, not a crash", exc.code == 400)

try:
    get("/api/nope")
    check("an unknown route is a 404", False)
except urllib.error.HTTPError as exc:
    check("an unknown route is a 404", exc.code == 404)

status, text, _ = post("/api/chat", {"messages": [], "stream": False})
check("an empty messages array answers rather than erroring",
      status == 200 and json.loads(text)["done"] is True)

server.shutdown()
server.server_close()

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
