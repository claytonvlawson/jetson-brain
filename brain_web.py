#!/usr/bin/env python3
"""brain_web.py -- phone-friendly web UI for the jetson-brain RAG, over Tailscale.

A self-contained mobile chat page you open in Safari (or any browser) on a device
in your tailnet. It speaks to the SAME retrieval + answer pipeline the terminal
brain and the advisor widget use (brain.load_index + brain.handle_question), so
the answers and citations are identical.

    GET  /         -> the mobile chat page (one self-contained HTML doc)
    POST /ask      {"question": "...", "web": false, "deep": false}
                   -> {"answer": "...", "sources": [...], "model": "...", "ms": N}
    GET  /health   -> {"ok": true, "chunks": N, "fast": "...", "deep": "..."}

Design:
- Zero extra dependencies (stdlib http.server); nothing new to install.
- Binding is set by BRAIN_WEB_HOST (default 127.0.0.1, a safe localhost default).
  The systemd unit (brain-web.service) sets it to this node's Tailscale IP so a
  phone on the tailnet opens http://<tailscale-ip>:8899/ directly -- no MagicDNS
  and no `tailscale serve` (iOS Safari often will not resolve the .ts.net name).
  A Tailscale IP is only routable inside your tailnet, so it stays private to
  your own devices. (Prefer the clean https://<node>.<tailnet>.ts.net/ URL? Turn
  on HTTPS certs for the tailnet and run `brain-web.sh serve-https`.)
- One answer at a time: a single 8GB device, so a semaphore serializes inference
  and concurrent requests queue behind the in-flight answer instead of thrashing
  or OOM-ing the GPU. Past a timeout the caller gets a friendly 503. Page loads
  (GET /) and /health never take the lock, so the UI stays responsive.
- Stateless: each question is answered fresh over the vault (no server-side chat
  history). The page keeps a visual transcript; the backend does not.

Run locally to test (no sudo, binds localhost):
    python3 brain_web.py         # then: curl http://127.0.0.1:8899/health
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brain  # safe to import: main() sits behind an __main__ guard

HOST = os.environ.get("BRAIN_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("BRAIN_WEB_PORT", "8899"))
FAST_MODEL = os.environ.get("BRAIN_WEB_MODEL", brain.FAST_MODEL)
DEEP_MODEL = os.environ.get("BRAIN_WEB_DEEP_MODEL", brain.DEEP_MODEL)
MAX_Q = int(os.environ.get("BRAIN_WEB_MAX_Q", "500"))            # cap question length
QUEUE_TIMEOUT = float(os.environ.get("BRAIN_WEB_QUEUE_TIMEOUT", "120"))  # deep model can be slow

print("Loading the vault index...", flush=True)
CHUNKS, BM25, META = brain.load_index()
print("Loaded %d chunks. fast=%s deep=%s" % (len(CHUNKS), FAST_MODEL, DEEP_MODEL), flush=True)

# Single-flight: one grounded answer at a time on one device.
_gpu = threading.BoundedSemaphore(1)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Brain">
<meta name="theme-color" content="#0e0f13">
<title>Brain</title>
<style>
  :root{
    --bg:#0e0f13; --panel:#16181f; --text:#e8e9ee; --muted:#9aa0ad;
    --accent:#f2b23e; --accent2:#7aa2ff; --line:#262a35; --user:#20242f; --danger:#ff6b6b;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    display:flex; flex-direction:column;
    padding:env(safe-area-inset-top) env(safe-area-inset-right) 0 env(safe-area-inset-left);
  }
  header{
    padding:14px 16px 11px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; position:sticky; top:0; background:var(--bg); z-index:5;
  }
  header .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}
  header h1{font-size:17px;margin:0;font-weight:600;letter-spacing:.2px}
  header .sub{color:var(--muted);font-size:12px;margin-left:auto;text-align:right}
  #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;-webkit-overflow-scrolling:touch}
  .msg{max-width:88%;padding:11px 14px;border-radius:16px;white-space:pre-wrap;word-wrap:break-word}
  .q{align-self:flex-end;background:var(--user);border:1px solid var(--line);border-bottom-right-radius:5px}
  .a{align-self:flex-start;background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:5px}
  .a .srcs{margin-top:10px;padding-top:9px;border-top:1px dashed var(--line);font-size:12.5px;color:var(--muted)}
  .a .srcs b{color:var(--accent);font-weight:600}
  .a .srcs a,.a .srcs span.src{display:block;margin-top:3px;color:var(--accent2);text-decoration:none;word-break:break-all}
  .meta{font-size:11px;color:var(--muted);margin-top:8px}
  .err{color:var(--danger);align-self:flex-start;background:#201414;border:1px solid #3a1f1f}
  .think{align-self:flex-start;color:var(--muted);font-size:14px;display:flex;align-items:center;gap:9px}
  .think .spin{width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:sp .8s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
  form{
    border-top:1px solid var(--line);background:var(--bg);
    padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    display:flex;flex-direction:column;gap:8px;position:sticky;bottom:0
  }
  .row{display:flex;gap:8px;align-items:flex-end}
  textarea{
    flex:1;resize:none;background:var(--panel);color:var(--text);border:1px solid var(--line);
    border-radius:13px;padding:11px 13px;font:inherit;max-height:120px;min-height:44px;line-height:1.35
  }
  textarea:focus{outline:none;border-color:var(--accent2)}
  button.send{
    background:var(--accent);color:#20160a;border:none;border-radius:12px;
    padding:0 16px;height:44px;font-weight:700;font-size:15px;cursor:pointer;white-space:nowrap
  }
  button.send:disabled{opacity:.5;cursor:default}
  .toggles{display:flex;gap:16px;padding:0 2px;font-size:13px;color:var(--muted)}
  .toggles label{display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
  .toggles input{accent-color:var(--accent)}
  .hint{color:var(--muted);text-align:center;margin:auto;padding:30px 20px;max-width:320px}
  .hint b{color:var(--text)}
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>Brain</h1>
    <span class="sub" id="sub">connecting...</span>
  </header>
  <div id="log">
    <div class="hint" id="hint"><b>Ask your thesis vault.</b><br>Answers come from your notes, with sources. Toggle web search or the deeper model below.</div>
  </div>
  <form id="f">
    <div class="row">
      <textarea id="q" rows="1" placeholder="Ask a question..." autocomplete="off"></textarea>
      <button class="send" id="send" type="submit">Ask</button>
    </div>
    <div class="toggles">
      <label><input type="checkbox" id="web"> Web search</label>
      <label><input type="checkbox" id="deep"> Deep model (slower)</label>
    </div>
  </form>
<script>
(function(){
  var log=document.getElementById('log'), f=document.getElementById('f'),
      q=document.getElementById('q'), send=document.getElementById('send'),
      sub=document.getElementById('sub'), hint=document.getElementById('hint'),
      web=document.getElementById('web'), deep=document.getElementById('deep');
  var busy=false;

  function scroll(){ log.scrollTop=log.scrollHeight; }
  function el(cls,txt){ var d=document.createElement('div'); d.className=cls; if(txt!=null) d.textContent=txt; return d; }

  fetch('/health').then(function(r){return r.json();}).then(function(d){
    sub.textContent=(d.chunks||0)+' notes · '+(d.fast||'');
  }).catch(function(){ sub.textContent='offline'; });

  q.addEventListener('input',function(){ q.style.height='auto'; q.style.height=Math.min(q.scrollHeight,120)+'px'; });

  function addAnswer(d){
    var a=el('msg a'); a.textContent=d.answer||'';
    if(d.sources && d.sources.length){
      var s=el('srcs'); var b=document.createElement('b'); b.textContent='sources'; s.appendChild(b);
      d.sources.forEach(function(src){
        var isUrl=/^https?:/.test(src);
        var line=document.createElement(isUrl?'a':'span');
        if(isUrl){ line.href=src; line.target='_blank'; line.rel='noopener'; }
        else { line.className='src'; }
        line.textContent=src; s.appendChild(line);
      });
      a.appendChild(s);
    }
    var m=el('meta'); m.textContent=(d.model||'')+' · '+((d.ms||0)/1000).toFixed(1)+'s';
    a.appendChild(m);
    log.appendChild(a); scroll();
  }

  f.addEventListener('submit',function(e){
    e.preventDefault();
    if(busy) return;
    var text=q.value.trim(); if(!text) return;
    if(hint){ hint.remove(); hint=null; }
    log.appendChild(el('msg q',text));
    q.value=''; q.style.height='auto'; scroll();
    busy=true; send.disabled=true;

    var think=el('think'); think.appendChild(el('spin'));
    var label=document.createElement('span'); label.textContent='thinking...'; think.appendChild(label);
    log.appendChild(think); scroll();
    var t0=Date.now();
    var timer=setInterval(function(){ label.textContent='thinking... '+((Date.now()-t0)/1000).toFixed(0)+'s'; },1000);

    fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text, web:web.checked, deep:deep.checked})})
    .then(function(r){ return r.json().then(function(j){ return {ok:r.ok,status:r.status,j:j}; }); })
    .then(function(res){
      clearInterval(timer); think.remove();
      if(res.ok){ addAnswer(res.j); }
      else{ log.appendChild(el('msg err',(res.j&&res.j.error)||('Error '+res.status))); scroll(); }
    })
    .catch(function(){
      clearInterval(timer); think.remove();
      log.appendChild(el('msg err','Could not reach the brain. Is Tailscale on?')); scroll();
    })
    .then(function(){ busy=false; send.disabled=false; });
  });

  q.addEventListener('keydown',function(e){
    if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); f.requestSubmit(); }
  });
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload, ctype="application/json", close=False):
        if ctype == "application/json":
            body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = payload
        # close=True is for POST rejects that never read the request body. Under
        # HTTP/1.1 the connection is kept alive by default and the handler does
        # NOT auto-drain an unread body, so those Content-Length bytes would be
        # parsed as the *next* request and desync the connection -- which matters
        # because `tailscale serve` fronts us as a pooled reverse proxy and reuses
        # upstream connections for later, legitimate requests. Closing the socket
        # (vs. draining) also avoids reading an arbitrarily large declared body.
        if close:
            self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up; nothing to do

    def log_message(self, *args):
        pass  # quiet the default per-request stderr line

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, ctype="text/html; charset=utf-8")
        if path == "/health":
            return self._send(200, {"ok": True, "chunks": len(CHUNKS),
                                    "fast": FAST_MODEL, "deep": DEEP_MODEL})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        # These two guards reject BEFORE reading the body, so close the connection
        # to avoid an HTTP/1.1 keep-alive desync (see _send's close=True note).
        if self.path.split("?", 1)[0] != "/ask":
            return self._send(404, {"error": "not found"}, close=True)

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 10000:
            return self._send(400, {"error": "bad request"}, close=True)

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(data.get("question", "")).strip()
            web = bool(data.get("web", False))
            deep = bool(data.get("deep", False))
        except Exception:
            return self._send(400, {"error": "invalid json"})

        if not question:
            return self._send(400, {"error": "empty question"})
        question = question[:MAX_Q]
        model = DEEP_MODEL if deep else FAST_MODEL

        # If the brain is busy past the timeout, tell the caller to retry rather
        # than piling more work onto the one device.
        if not _gpu.acquire(timeout=QUEUE_TIMEOUT):
            return self._send(503, {"error": "The brain is busy with another question. Try again in a moment."})
        t0 = time.time()
        try:
            answer, sources = brain.handle_question(
                model, question, [], BM25, CHUNKS, META, web, brain._NullRun())
        except Exception as exc:  # never let one bad question take the server down
            print("[brain_web] error:", exc, flush=True)
            return self._send(502, {"error": "The brain hit an error answering that."})
        finally:
            _gpu.release()

        if not answer:
            return self._send(502, {"error": "The brain could not answer that from the vault."})
        return self._send(200, {
            "answer": answer,
            "sources": [str(s) for s in (sources or [])],
            "model": model,
            "ms": int((time.time() - t0) * 1000),
        })


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("brain_web listening on http://%s:%d  (GET / , POST /ask)" % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
