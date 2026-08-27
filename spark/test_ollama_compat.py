#!/usr/bin/env python3
"""test_ollama_compat.py -- offline tests for the vLLM front door.

No network, no vLLM, no GPU. A fake OpenAI server stands in for vLLM so the
tests can assert BOTH halves of the contract:

  * what brain.py reads out of the shim (Ollama frame shape, the three
    counters _capture_chat_stats looks for, the single-vector embedding
    spelling semantic_cache._embed requires)
  * what the shim sends upstream (option mapping, model aliasing) -- the half
    a response-only test would miss entirely

Run: python3 test_ollama_compat.py
"""

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Configure the module BEFORE importing it: config is read at import time.
FAKE_PORT = 8931
SHIM_PORT = 11439
os.environ["SPARK_BASE_URL"] = "http://127.0.0.1:%d/v1" % FAKE_PORT
os.environ["SPARK_MODEL"] = "served/fast-model"
os.environ["SPARK_DEEP_MODEL"] = "served/deep-model"
os.environ["SPARK_EMBED_MODEL"] = "served/embedder"
os.environ["SPARK_PORT"] = str(SHIM_PORT)

import ollama_compat as oc  # noqa: E402

RECEIVED = []          # every payload the fake upstream was handed


class FakeVLLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        RECEIVED.append((self.path, body))
        if self.path.endswith("/embeddings"):
            return self._json({"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]})
        if body.get("stream"):
            return self._sse()
        return self._json({
            "choices": [{"message": {"role": "assistant", "content": "grounded answer"}}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 7},
        })

    def _json(self, obj):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(text):
            raw = (text + "\n\n").encode("utf-8")
            self.wfile.write(("%X\r\n" % len(raw)).encode("ascii") + raw + b"\r\n")
            self.wfile.flush()

        for piece in ("Start ", "with ", "the IFSP."):
            emit("data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}}]}))
        emit("data: " + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 41, "completion_tokens": 3}}))
        emit("data: [DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def post(path, obj, port=SHIM_PORT, raw=False):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read().decode("utf-8")
    return body if raw else json.loads(body)


def get(path, port=SHIM_PORT):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=10) as r:
        return r.read().decode("utf-8")


FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   " + name)
    else:
        print("  FAIL " + name + ("  " + detail if detail else ""))
        FAILED.append(name)


def main():
    for server, port in ((FakeVLLM, FAKE_PORT), (oc.Handler, SHIM_PORT)):
        srv = ThreadingHTTPServer(("127.0.0.1", port), server)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    print("pure functions")
    check("deep alias resolves to the deep served model",
          oc.resolve_model(oc.DEEP_ALIAS) == "served/deep-model")
    check("fast alias resolves to the fast served model",
          oc.resolve_model(oc.FAST_ALIAS) == "served/fast-model")
    check("an unknown name passes through unchanged",
          oc.resolve_model("served/other") == "served/other")

    opts = oc.map_options({"num_predict": 640, "num_ctx": 24576,
                           "temperature": 0.2, "repeat_penalty": 1.1,
                           "num_gpu": 999, "seed": 7})
    check("num_predict -> max_tokens", opts.get("max_tokens") == 640)
    check("temperature passes through", opts.get("temperature") == 0.2)
    check("repeat_penalty -> repetition_penalty",
          opts.get("repetition_penalty") == 1.1)
    check("seed is forwarded (eval reproducibility)", opts.get("seed") == 7)
    check("num_ctx is dropped, not proxied", "num_ctx" not in opts)
    check("num_gpu is dropped, not proxied", "num_gpu" not in opts)
    check("an absent seed stays absent", "seed" not in oc.map_options({}))

    print("probes")
    check("root answers the connectivity probe", get("/") == "Ollama is running")
    tags = json.loads(get("/api/tags"))
    names = [m["name"] for m in tags["models"]]
    check("tags lists both tiers brain dispatches on",
          oc.FAST_ALIAS in names and oc.DEEP_ALIAS in names, str(names))
    show = post("/api/show", {"model": oc.FAST_ALIAS})
    check("show reports a context length", ".context_length" in
          " ".join(show.get("model_info", {}).keys()))

    print("chat, non-streaming")
    RECEIVED.clear()
    out = post("/api/chat", {"model": oc.DEEP_ALIAS, "stream": False,
                             "messages": [{"role": "user", "content": "hi"}],
                             "options": {"num_predict": 640, "num_ctx": 24576,
                                         "seed": 7, "num_gpu": 999}})
    check("content lands where brain reads it",
          (out.get("message") or {}).get("content") == "grounded answer")
    check("done is true", out.get("done") is True)
    check("prompt_eval_count is captured", out.get("prompt_eval_count") == 41)
    check("eval_count is captured", out.get("eval_count") == 7)
    check("total_duration is captured", isinstance(out.get("total_duration"), int)
          and out["total_duration"] > 0)
    check("the alias is echoed back, not the served name",
          out.get("model") == oc.DEEP_ALIAS)
    sent = RECEIVED[-1][1]
    check("upstream got the DEEP served model", sent.get("model") == "served/deep-model")
    check("upstream got max_tokens", sent.get("max_tokens") == 640)
    check("upstream got the seed", sent.get("seed") == 7)
    check("upstream never saw num_ctx", "num_ctx" not in sent)
    check("upstream never saw num_gpu", "num_gpu" not in sent)

    print("chat, streaming")
    body = post("/api/chat", {"model": oc.FAST_ALIAS, "stream": True,
                              "messages": [{"role": "user", "content": "hi"}],
                              "options": {"temperature": 0.0}}, raw=True)
    frames = [json.loads(x) for x in body.strip().splitlines() if x.strip()]
    text = "".join((f.get("message") or {}).get("content", "") for f in frames)
    check("streamed text reassembles in order", text == "Start with the IFSP.", text)
    check("every frame but the last is done:false",
          all(f["done"] is False for f in frames[:-1]))
    check("the final frame is done:true", frames[-1]["done"] is True)
    check("the final frame carries the counters",
          frames[-1].get("prompt_eval_count") == 41 and
          frames[-1].get("eval_count") == 3)
    check("upstream was asked for usage in the stream",
          RECEIVED[-1][1].get("stream_options", {}).get("include_usage") is True)

    print("embeddings")
    RECEIVED.clear()
    emb = post("/api/embeddings", {"model": "nomic-embed-text", "prompt": "text"})
    check("legacy route returns ONE vector under 'embedding'",
          emb.get("embedding") == [0.1, 0.2, 0.3], json.dumps(emb)[:120])
    check("upstream got the served embedder name",
          RECEIVED[-1][1].get("model") == "served/embedder")
    import io, contextlib
    oc._EMBED_WARNED.clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        oc._resolve_embed("nomic-embed-text")
        oc._resolve_embed("nomic-embed-text")   # second time must stay quiet
        oc._resolve_embed(oc.EMBED_MODEL)       # a match must never warn
    warned = err.getvalue()
    check("a mismatched embedder warns loudly, not silently",
          "WARNING" in warned and "nomic-embed-text" in warned)
    check("the warning fires once per name, not once per call",
          warned.count("WARNING") == 1, "count=%d" % warned.count("WARNING"))

    emb2 = post("/api/embed", {"model": "nomic-embed-text", "input": "text"})
    check("newer route returns a LIST of vectors",
          emb2.get("embeddings") == [[0.1, 0.2, 0.3]])

    print("unload no-op")
    gen = post("/api/generate", {"model": oc.FAST_ALIAS, "keep_alive": 0})
    check("generate answers 200 so brain does not log an eviction failure",
          gen.get("done") is True)

    print()
    if FAILED:
        print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
