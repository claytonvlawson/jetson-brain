#!/usr/bin/env python3
"""advisor_server.py -- public HTTP wrapper around the jetson-brain RAG.

Exposes a single endpoint the "Ask the advisor" widget on learnwithclay.ai
speaks natively:

    POST /chat   {"question": "..."}  ->  {"answer": "...", "sources": [...]}
    GET  /health                      ->  {"ok": true, "chunks": N, "model": "..."}

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
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brain  # safe to import: main() sits behind an __main__ guard

HOST = os.environ.get("ADVISOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("ADVISOR_PORT", "8000"))
TOKEN = os.environ.get("ADVISOR_TOKEN", "")            # shared secret; "" disables the check
MODEL = os.environ.get("ADVISOR_MODEL", brain.FAST_MODEL)
WEB_ENABLED = os.environ.get("ADVISOR_WEB", "0") == "1"
MAX_Q = int(os.environ.get("ADVISOR_MAX_Q", "500"))    # cap question length
QUEUE_TIMEOUT = float(os.environ.get("ADVISOR_QUEUE_TIMEOUT", "45"))  # seconds to wait for the GPU

print("Loading the vault index...", flush=True)
CHUNKS, BM25, META = brain.load_index()
print("Loaded %d chunks. Model: %s. Web: %s" % (len(CHUNKS), MODEL, WEB_ENABLED), flush=True)

# Single-flight: the whole point is one grounded answer at a time on one device.
_gpu = threading.BoundedSemaphore(1)


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

    def do_POST(self):
        if self.path != "/chat":
            return self._send(404, {"error": "not found"})

        if TOKEN and self.headers.get("Authorization", "") != "Bearer " + TOKEN:
            return self._send(401, {"error": "unauthorized"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 10000:
            return self._send(400, {"error": "bad request"})

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(data.get("question", "")).strip()
        except Exception:
            return self._send(400, {"error": "invalid json"})

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
        return self._send(200, {
            "answer": answer,
            "sources": [str(s) for s in (sources or [])],
        })


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("advisor listening on http://%s:%d  (POST /chat)" % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
