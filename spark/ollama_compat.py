#!/usr/bin/env python3
r"""ollama_compat.py -- speak Ollama's wire protocol INWARD so brain.py runs
byte-identical on the DGX Spark against a vLLM OpenAI endpoint.

This is the mirror image of brain_ollama_shim.py. That one makes brain LOOK
like Ollama to an Obsidian plugin. This one makes vLLM look like Ollama to
brain, so nothing in brain.py's four chat paths, its embedding calls, or the
eval harness has to change to move off the Jetson.

    GET  /                -> "Ollama is running"     (connectivity probe)
    GET  /api/version     -> {"version": "..."}
    GET  /api/tags        -> the aliased models brain expects to find
    POST /api/show        -> per-model metadata (the ollama client calls this)
    POST /api/chat        -> NDJSON Ollama frames, streamed from vLLM's SSE,
                             or one JSON object when {"stream": false}
    POST /api/generate    -> unload no-op (see "keep_alive" below)
    POST /api/embeddings  -> {"embedding": [...]}    (legacy, what brain uses)
    POST /api/embed       -> {"embeddings": [[...]]} (newer spelling)

Why this exists instead of a brain.py refactor: the eval harness
(eval_brain.py, parent_gold.jsonl, heldout_gold.jsonl, gold_questions.jsonl)
is the only proof that the Spark port did not regress the measured 8/8 / 28/28
/ 13/14. A refactor of the generation path invalidates that gate at exactly the
moment you need it. A shim leaves brain.py untouched, so the same suites score
the same thing on both boxes and the comparison means something.

Four things that were NOT obvious and cost real reading:

1. BRAIN TALKS TO OLLAMA TWO DIFFERENT WAYS, and they read the host from two
   different places. chat() and chat_stream() (brain.py:2154, :2214) prefer the
   `ollama` PYTHON CLIENT and only fall back to REST. The client reads the
   OLLAMA_HOST *environment variable*; the REST fallback reads brain.py's
   module constant, which at brain.py:98 is a hardcoded literal. Repoint only
   one and you get a split brain: the client hits this shim while the fallback
   quietly hits a dead 11434, and the failure only shows up on the error path.
   The fix is brain.py's _ollama_host() (brain.py:100), which reads the same
   OLLAMA_HOST env var the client reads and adds a scheme when one is missing.
   That makes one env var steer both paths, and keeps the Jetson default.

2. MODEL NAMES DO NOT MATCH AND MUST BE ALIASED. brain.py asks for
   "llama3.2:3b" (FAST_MODEL) and "qwen2.5:7b" (DEEP_MODEL) by name, and those
   names are load-bearing: _keep_alive(), _gen_options(), _before_chat() and
   _num_ctx() all dispatch on `model == DEEP_MODEL`. So the shim maps those two
   names onto whatever vLLM actually serves rather than renaming them in brain,
   which would silently change tier dispatch. Both tiers may point at the same
   served model; on a 128 GB Spark serving one Qwen3.6-35B-A3B-FP8 they do.

3. `seed` MUST BE FORWARDED OR EVERY A/B IS UNDECIDABLE. BRAIN_SEED writes
   GEN_OPTIONS["seed"] (brain.py:185) and the eval protocol depends on it:
   unseeded judged runs flip 5 of 28 questions, which makes any unpinned
   comparison noise. vLLM honours `seed` on the OpenAI route, so it is mapped
   straight through. Dropping it would not error, it would just quietly make
   the gate meaningless.

4. `num_ctx` IS NOT MAPPABLE AND THAT IS FINE, BUT IT IS ALSO A TRAP. Ollama
   takes the context window per request; vLLM fixes it at launch with
   --max-model-len. So num_ctx is dropped here. The trap is that brain BUDGETS
   prompts against _num_ctx(model) (fast 24576, deep 12288) and Ollama
   truncates over-long prompts FROM THE TOP, taking the system prompt with it.
   vLLM instead REJECTS them with a 400. That is strictly better -- a loud
   failure rather than a silent ungrounding -- but it means --max-model-len
   must be >= the largest window brain budgets, or long parent answers start
   400ing. Serve with --max-model-len 65536 and there is headroom either way.

Config (all via environment, all with Jetson-safe defaults):

    SPARK_BASE_URL     vLLM OpenAI base, default http://127.0.0.1:8000/v1
    SPARK_API_KEY      bearer for the above, default "local"
    SPARK_MODEL        served chat model name (required in practice)
    SPARK_FAST_ALIAS   brain's fast name,  default llama3.2:3b
    SPARK_DEEP_ALIAS   brain's deep name,  default qwen2.5:7b
    SPARK_DEEP_MODEL   served model for the deep tier, default = SPARK_MODEL
    SPARK_EMBED_URL    embeddings base, default = SPARK_BASE_URL.
                       THE DEFAULT IS ONLY RIGHT IF ONE BACKEND SERVES BOTH.
                       A vLLM started to serve a chat model does NOT expose
                       nomic-embed-text on /v1/embeddings, so on the Spark this
                       must point at a real embedding server -- e.g. Ollama on
                       its own port, http://127.0.0.1:11500/v1 -- or every
                       _dense_scores() and _rerank_by_embedding() call fails.
    SPARK_EMBED_MODEL  served embedding model, default nomic-embed-text
    SPARK_PORT         listen port, default 11434 (drop-in for Ollama)

Run:  python3 ollama_compat.py
Test: python3 test_ollama_compat.py     (no network, no vLLM required)
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
API_KEY = os.environ.get("SPARK_API_KEY", "local")
# FP8, not NVFP4. GB10 is SM121 and lacks the cvt.rn.satfinite.e2m1x2 PTX
# instruction that SM120/SM100 have, so NVFP4 falls back to Marlin
# decompression: measured 40.8 tok/s vs 53.8 for FP8 on this same model.
MODEL = os.environ.get("SPARK_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
FAST_ALIAS = os.environ.get("SPARK_FAST_ALIAS", "llama3.2:3b")
DEEP_ALIAS = os.environ.get("SPARK_DEEP_ALIAS", "qwen2.5:7b")
DEEP_MODEL = os.environ.get("SPARK_DEEP_MODEL", MODEL)
EMBED_URL = os.environ.get("SPARK_EMBED_URL", BASE_URL).rstrip("/")
EMBED_MODEL = os.environ.get("SPARK_EMBED_MODEL", "nomic-embed-text")
PORT = int(os.environ.get("SPARK_PORT", "11434"))

# Advertised context length per alias. Only /api/show reads these, and only the
# Obsidian-style clients care, but brain's list_models() path is happier with a
# well-formed answer than with a 404.
CTX_BY_ALIAS = {FAST_ALIAS: 65536, DEEP_ALIAS: 65536}


def resolve_model(name):
    """brain's tier name -> the model vLLM actually serves.

    Unknown names pass through unchanged so a direct curl against this shim
    with the real served name still works."""
    if name == DEEP_ALIAS:
        return DEEP_MODEL
    if name == FAST_ALIAS:
        return MODEL
    return name


_EMBED_WARNED = set()


def _resolve_embed(requested):
    """Every embedding request routes to the ONE embedder vLLM serves.

    brain names its embedder in two places -- RERANK_EMBED_MODEL
    ("nomic-embed-text") and the index manifest's dm["model"] -- and neither is
    a tier alias, so a name-match rule lets them through untranslated and vLLM
    400s on a model it does not serve.

    Rewriting silently would be worse than the 400 though. If the on-disk index
    was embedded with a different model than the one now being served, the
    rewrite makes every dense score a cosine between vectors from two different
    models: no error, no crash, just quietly wrong retrieval. That is the exact
    failure class the port is supposed to rule out. So the rewrite happens (one
    served embedder, nothing else can work) but it announces itself once per
    distinct name, on stderr, where a deploy will see it."""
    if requested and requested != EMBED_MODEL and requested not in _EMBED_WARNED:
        _EMBED_WARNED.add(requested)
        sys.stderr.write(
            "[compat] WARNING: request asked for embedder %r, serving %r.\n"
            "[compat]          If the index was built with %r, its vectors are\n"
            "[compat]          NOT comparable to these. Re-embed, or serve %r.\n"
            % (requested, EMBED_MODEL, requested, requested))
    return EMBED_MODEL


def map_options(opts):
    """Ollama `options` -> (OpenAI kwargs, vLLM extra fields).

    Deliberately explicit rather than a pass-through dict: an unknown key
    forwarded to vLLM is a 400, and brain sends several that have no OpenAI
    equivalent (num_ctx, num_gpu) which must be dropped, not proxied."""
    opts = opts or {}
    out = {}
    if "num_predict" in opts:
        out["max_tokens"] = int(opts["num_predict"])
    if "temperature" in opts:
        out["temperature"] = float(opts["temperature"])
    if "top_p" in opts:
        out["top_p"] = float(opts["top_p"])
    if "seed" in opts:                       # load-bearing, see header note 3
        out["seed"] = int(opts["seed"])
    if "stop" in opts and opts["stop"]:
        out["stop"] = opts["stop"]
    if "repeat_penalty" in opts:             # vLLM extension, not core OpenAI
        out["repetition_penalty"] = float(opts["repeat_penalty"])
    # Dropped on purpose: num_ctx (launch-time --max-model-len), num_gpu (no
    # CPU offload concept in vLLM), and anything else brain grows later.
    return out


def _post_openai(path, payload, stream=False, timeout=180):
    """One call to the vLLM OpenAI route. Returns a decoded dict, or the raw
    response object when stream=True so the caller can read SSE lines."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path if not path.startswith("http") else path,
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    with resp:
        return json.loads(resp.read().decode("utf-8"))


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _usage_counters(usage, elapsed_ns):
    """OpenAI `usage` -> the three counters brain._capture_chat_stats reads.

    brain only ever looks for prompt_eval_count, eval_count and total_duration
    (brain.py:728). The rest of Ollama's timing block is emitted as zeros so a
    client that expects the full shape does not KeyError."""
    usage = usage or {}
    return {
        "total_duration": int(elapsed_ns),
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens", 0)),
        "prompt_eval_duration": 0,
        "eval_count": int(usage.get("completion_tokens", 0)),
        "eval_duration": int(elapsed_ns),
    }


def chat_frame(alias, content, done=False, **extra):
    frame = {
        "model": alias,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": bool(done),
    }
    frame.update(extra)
    return frame


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if os.environ.get("SPARK_COMPAT_VERBOSE"):
            sys.stderr.write("[compat] " + (fmt % args) + "\n")

    # ---- plumbing -------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- GET ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/":
            return self._send(200, b"Ollama is running", "text/plain")
        if path == "/api/version":
            return self._send(200, {"version": "0.12.0-spark-compat"})
        if path == "/api/tags":
            return self._send(200, {"models": [_tag(FAST_ALIAS), _tag(DEEP_ALIAS)]})
        return self._send(404, {"error": "not found"})

    # ---- POST -----------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        body = self._body()
        try:
            if path == "/api/chat":
                return self._chat(body)
            if path == "/api/generate":
                return self._generate(body)
            if path == "/api/show":
                return self._show(body)
            if path in ("/api/embeddings", "/api/embed"):
                return self._embeddings(body, legacy=(path == "/api/embeddings"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:500]
            except Exception:
                pass
            return self._send(502, {"error": "upstream %s: %s" % (exc.code, detail)})
        except Exception as exc:
            return self._send(502, {"error": "upstream unreachable: %s" % exc})
        return self._send(404, {"error": "not found"})

    def _show(self, body):
        alias = body.get("model") or FAST_ALIAS
        ctx = CTX_BY_ALIAS.get(alias, 65536)
        return self._send(200, {
            "license": "", "modelfile": "", "parameters": "",
            "template": "", "details": _details(alias),
            "model_info": {"general.architecture": "qwen3moe",
                           "qwen3moe.context_length": ctx},
        })

    def _generate(self, body):
        """brain.py:321 only ever calls /api/generate to UNLOAD the fast model
        (keep_alive: 0) before a deep turn, because 9.4 GB does not fit in the
        Jetson's 7.4 GB. On the Spark there is nothing to evict -- vLLM holds
        one resident model and multiplexes -- so this is a successful no-op.
        Answering 404 here would make brain log an eviction failure on every
        deep question."""
        return self._send(200, chat_frame(body.get("model") or FAST_ALIAS, "",
                                          done=True, done_reason="unload",
                                          response=""))

    def _embeddings(self, body, legacy):
        text = body.get("prompt")
        if text is None:
            text = body.get("input")
        payload = {"model": _resolve_embed(body.get("model")), "input": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            EMBED_URL + "/embeddings", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + API_KEY}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        vecs = [row.get("embedding") for row in (out.get("data") or [])]
        if legacy:
            # semantic_cache._embed reads .get("embedding") and treats anything
            # else as a failure, silently. One vector, not a list of them.
            return self._send(200, {"embedding": vecs[0] if vecs else []})
        return self._send(200, {"embeddings": vecs})

    def _chat(self, body):
        alias = body.get("model") or FAST_ALIAS
        served = resolve_model(alias)
        messages = body.get("messages") or []
        payload = {"model": served, "messages": messages}
        payload.update(map_options(body.get("options")))
        if body.get("stream"):
            return self._chat_stream(alias, payload)
        started = time.time_ns()
        payload["stream"] = False
        out = _post_openai("/chat/completions", payload)
        choices = out.get("choices") or [{}]
        content = ((choices[0].get("message") or {}).get("content")) or ""
        frame = chat_frame(alias, content, done=True, done_reason="stop")
        frame.update(_usage_counters(out.get("usage"), time.time_ns() - started))
        return self._send(200, frame)

    def _chat_stream(self, alias, payload):
        """SSE in, NDJSON out, one frame per delta.

        Chunked rather than buffered: brain's chat_stream() pushes every token
        into the UI sink as it arrives, and the whole point of moving to a box
        that decodes at ~289 tok/s is that the user sees it happen."""
        started = time.time_ns()
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        upstream = _post_openai("/chat/completions", payload, stream=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write(obj):
            raw = (json.dumps(obj) + "\n").encode("utf-8")
            self.wfile.write(("%X\r\n" % len(raw)).encode("ascii") + raw + b"\r\n")
            self.wfile.flush()

        usage = None
        try:
            with upstream:
                for line in upstream:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            write(chat_frame(alias, piece))
            final = chat_frame(alias, "", done=True, done_reason="stop")
            final.update(_usage_counters(usage, time.time_ns() - started))
            write(final)
        finally:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()


def _details(alias):
    return {"parent_model": "", "format": "gguf", "family": "qwen3moe",
            "families": ["qwen3moe"], "parameter_size": "35B",
            "quantization_level": "FP8"}


def _tag(alias):
    return {"name": alias, "model": alias, "modified_at": _now_iso(),
            "size": 23400000000, "digest": "spark-compat",
            "details": _details(alias)}


def main():
    sys.stderr.write(
        "[compat] Ollama-shaped front door for vLLM\n"
        "[compat]   listening   127.0.0.1:%d\n"
        "[compat]   upstream    %s\n"
        "[compat]   chat model  %s\n"
        "[compat]   deep model  %s\n"
        "[compat]   embeddings  %s @ %s\n"
        "[compat]   aliases     %s (fast), %s (deep)\n"
        % (PORT, BASE_URL, MODEL, DEEP_MODEL, EMBED_MODEL, EMBED_URL,
           FAST_ALIAS, DEEP_ALIAS))
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
