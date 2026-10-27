#!/usr/bin/env python3
"""gemma_vision.py -- live webcam view plus object identification with gemma4:e2b.

One process: a camera thread keeps the newest frame, an analyzer thread sends a
downscaled snapshot to Ollama, and a small HTTP server shows both in a page.

The video is live (MJPEG). The identification is NOT: on the 8GB Orin Nano one
vision pass takes ~20s (fast), ~45s (quality) or ~75s (reasoning), almost all of
it spent encoding the image. The page says so and shows the exact frame each
answer was based on, so a stale list is never mistaken for the current view.

Bind policy (same as Sign Studio): the page shows the camera and has no login,
so bind only loopback and the USB-C link address, never the tailnet or the LAN.
A Host-header allowlist and a JSON-only POST rule guard against DNS rebinding
and cross-site form posts reaching the loopback port.

    python3 gemma_vision.py --host 127.0.0.1,192.168.55.1 --port 8093
"""
import argparse
import base64
import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

MODEL = "gemma4:e2b-it-q4_K_M"
OLLAMA = "http://127.0.0.1:11434"

# mode -> (image width sent to the model, reasoning on, max tokens, label)
MODES = {
    "fast":      (512, False, 300,  "Fast"),
    "quality":   (768, False, 400,  "Quality"),
    "reasoning": (768, True,  1800, "Reasoning"),
}

PROMPT = (
    "List every distinct physical object visible in this image, one per line, in "
    "exactly this form: name | location\n"
    "The location is one or two words such as left, center, right, top, bottom or "
    "background. Use short common names. Only list what you can actually see. Do "
    "not list the same object twice. Do not describe or characterize any person; "
    "name only body parts, clothing and accessories. Output only the lines."
)

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "192.168.55.1"}


class Camera(threading.Thread):
    """Keeps the newest frame; re-opens the device if it drops."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.lock = threading.Lock()
        self.frame = None        # newest full-size BGR frame
        self.jpeg = None         # newest frame as a stream-sized JPEG
        self.seq = 0
        self.ok = False
        self.error = "starting"

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                self.ok, self.error = False, "cannot open /dev/video%d (another program may hold it)" % self.device
                time.sleep(2)
                continue
            last = 0.0
            while True:
                good, frame = cap.read()
                if not good or frame is None:
                    self.ok, self.error = False, "camera stopped returning frames"
                    break
                now = time.time()
                if now - last < 0.1:         # cap the stream at ~10 fps
                    continue
                last = now
                small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
                enc_ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not enc_ok:
                    continue
                with self.lock:
                    self.frame, self.jpeg, self.seq = frame, buf.tobytes(), self.seq + 1
                self.ok, self.error = True, ""
            cap.release()
            time.sleep(1)

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stream_jpeg(self):
        with self.lock:
            return self.jpeg, self.seq


def parse_objects(text):
    """Turn 'name | location' lines into [{'name','where'}]. Tolerates bullets,
    numbering and stray markdown so none of it reaches the page."""
    out, seen = [], set()
    lines = text.splitlines()
    if any("|" in ln for ln in lines):      # headings and chatter carry no pipe
        lines = [ln for ln in lines if "|" in ln]
    for line in lines:
        line = re.sub(r"[*_`#>]+", "", line).strip()
        line = re.sub(r"^(?:[-+•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        name, _, where = line.partition("|")
        name, where = name.strip(" .:"), where.strip(" .")
        if not name or len(name) > 60:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "where": where[:40]})
        if len(out) >= 30:
            break
    return out


class Analyzer(threading.Thread):
    def __init__(self, camera):
        super().__init__(daemon=True)
        self.cam = camera
        self.cv = threading.Condition()
        self.want = False
        self.auto = False
        self.mode = "quality"
        self.busy = False
        self.started = 0.0
        self.last = None         # dict describing the newest finished analysis
        self.snapshot_jpeg = None
        self.error = ""

    def request(self):
        with self.cv:
            self.want = True
            self.cv.notify()

    def configure(self, auto=None, mode=None):
        with self.cv:
            if mode in MODES:
                self.mode = mode
            if auto is not None:
                self.auto = bool(auto)
                if self.auto:
                    self.want = True
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while not (self.want or (self.auto and self.cam.ok)):
                    self.cv.wait(timeout=1.0)
                self.want = False
                mode = self.mode
            frame = self.cam.snapshot()
            if frame is None:
                self.error = "no camera frame yet"
                time.sleep(1)
                continue
            self._analyze(frame, mode)
            if self.error:
                time.sleep(5)    # back off so auto mode does not hammer a failing model

    def _analyze(self, frame, mode):
        width, think, max_tokens, label = MODES[mode]
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(h * width / w)))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg = buf.tobytes()
        body = {
            "model": MODEL, "stream": False, "think": think, "keep_alive": "5m",
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [base64.b64encode(jpg).decode()]}],
            "options": {"num_ctx": 4096, "temperature": 0.1, "num_predict": max_tokens},
        }
        self.busy, self.started, self.error = True, time.time(), ""
        try:
            req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.load(resp)
            msg = data.get("message") or {}
            content = (msg.get("content") or "").strip()
            self.snapshot_jpeg = jpg
            self.last = {
                "objects": parse_objects(content),
                "raw": content,
                "thinking": (msg.get("thinking") or "").strip(),
                "secs": round(time.time() - self.started, 1),
                "finished": time.time(),
                "mode": label,
            }
        except Exception as exc:       # keep the previous good answer on screen
            self.error = "model call failed: %s" % exc
        finally:
            self.busy = False

    def state(self):
        now = time.time()
        last = None
        if self.last:
            last = dict(self.last)
            last["age"] = round(now - last.pop("finished"), 1)
        return {
            "camera": {"ok": self.cam.ok, "error": self.cam.error},
            "auto": self.auto, "mode": self.mode,
            "modes": {k: {"label": v[3], "width": v[0], "reasoning": v[1]} for k, v in MODES.items()},
            "busy": self.busy,
            "busy_for": round(now - self.started, 1) if self.busy else 0,
            "error": self.error, "last": last, "model": MODEL,
        }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma Vision</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--text:#e8eaf0;--mute:#8d95a6;--acc:#5ec2a5;--warn:#e6b450;--bad:#e5736b}
@media (prefers-color-scheme:light){:root{--bg:#f4f5f7;--card:#fff;--line:#dfe2e8;--text:#1b1e26;--mute:#667085;--acc:#1b8a6b;--warn:#a06a00;--bad:#c2372d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:20px;margin:0;font-weight:650}
.pill{font-size:12px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.pill.ok{color:var(--acc);border-color:var(--acc)}.pill.bad{color:var(--bad);border-color:var(--bad)}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:16px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--mute);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
#video{width:100%;display:block;border-radius:8px;background:#000;aspect-ratio:16/9}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:12px}
button{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 14px;cursor:pointer}
button.primary{background:var(--acc);color:#06241b;border-color:var(--acc);font-weight:600}
button:disabled{opacity:.5;cursor:default}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:8px 12px}
.seg button.on{background:var(--acc);color:#06241b;font-weight:600}
label.tog{display:inline-flex;gap:8px;align-items:center;cursor:pointer}
.status{font-size:13px;color:var(--mute);margin:0 0 10px}
.status b{color:var(--text);font-weight:600}
.bar{height:3px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0 12px;visibility:hidden}
.bar.on{visibility:visible}.bar i{display:block;height:100%;width:35%;background:var(--acc);animation:slide 1.4s ease-in-out infinite}
@keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
ul{list-style:none;margin:0;padding:0}
li.obj{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-top:1px solid var(--line)}
li.obj:first-child{border-top:0}li.obj span:last-child{color:var(--mute);white-space:nowrap}
.empty{color:var(--mute)}
img.snap{width:100%;border-radius:8px;display:block;margin-top:10px}
details{margin-top:12px}summary{cursor:pointer;color:var(--mute);font-size:13px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--mute);margin:8px 0 0;max-height:260px;overflow:auto}
.note{font-size:12px;color:var(--mute);margin:12px 0 0}
.err{color:var(--bad);font-size:13px;margin:8px 0 0}
</style></head><body><main>
<header><h1>Gemma Vision</h1><span class="pill" id="cam">camera</span><span class="pill" id="model"></span></header>
<div class="grid">
  <section class="card"><h2>Live camera</h2>
    <img id="video" src="/stream.mjpg" alt="Live camera view">
    <div class="controls">
      <button class="primary" id="go">Identify now</button>
      <label class="tog"><input type="checkbox" id="auto"> Keep identifying</label>
      <span class="seg" id="modes"></span>
    </div>
    <p class="note" id="modenote"></p>
  </section>
  <section class="card"><h2>What it sees</h2>
    <p class="status" id="status">Waiting for the first identification.</p>
    <div class="bar" id="bar"><i></i></div>
    <ul id="objs"></ul>
    <p class="err" id="err"></p>
    <details id="dsnap" hidden><summary>Frame this answer is based on</summary><img class="snap" id="snap" alt="Analyzed frame"></details>
    <details id="dthink" hidden><summary>Model reasoning</summary><pre id="think"></pre></details>
  </section>
</div>
<p class="note">The video is live. The list refreshes when a pass finishes, which takes tens of seconds on this board, so it describes the frame shown under it, not necessarily what the camera sees right now.</p>
</main><script>
const $=id=>document.getElementById(id);
let built=false,snapStamp=0;
async function post(path,body){await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});poll()}
function build(s){
  const box=$('modes');box.textContent='';
  for(const [k,v] of Object.entries(s.modes)){
    const b=document.createElement('button');b.textContent=v.label;b.dataset.k=k;
    b.onclick=()=>post('/api/settings',{mode:k});box.appendChild(b)}
  $('go').onclick=()=>post('/api/analyze');
  $('auto').onchange=e=>post('/api/settings',{auto:e.target.checked});
  built=true}
function render(s){
  if(!built)build(s);
  $('model').textContent=s.model;
  const c=$('cam');c.textContent=s.camera.ok?'camera live':'camera offline';c.className='pill '+(s.camera.ok?'ok':'bad');
  for(const b of $('modes').children)b.classList.toggle('on',b.dataset.k===s.mode);
  $('auto').checked=s.auto;$('go').disabled=s.busy;
  const m=s.modes[s.mode];
  $('modenote').textContent=m.label+': image sent at '+m.width+' px, reasoning '+(m.reasoning?'on':'off')+'.';
  $('bar').className='bar'+(s.busy?' on':'');
  const st=$('status');
  if(s.busy)st.textContent='Analyzing for '+Math.round(s.busy_for)+' s...';
  else if(s.last)st.textContent=s.last.mode+' pass finished '+Math.round(s.last.age)+' s ago, took '+s.last.secs+' s.';
  else st.textContent=s.camera.ok?'Press Identify now.':'Camera is not available.';
  $('err').textContent=s.error||s.camera.error||'';
  const ul=$('objs');ul.textContent='';
  if(s.last){
    if(!s.last.objects.length){const li=document.createElement('li');li.className='empty';li.textContent=s.last.raw?'Could not read a list. Raw reply: '+s.last.raw:'The model returned nothing.';ul.appendChild(li)}
    for(const o of s.last.objects){const li=document.createElement('li');li.className='obj';
      const a=document.createElement('span');a.textContent=o.name;const b=document.createElement('span');b.textContent=o.where;
      li.append(a,b);ul.appendChild(li)}
    const stamp=Math.round(Date.now()/1000-s.last.age);
    if(Math.abs(stamp-snapStamp)>1){snapStamp=stamp;$('snap').src='/snapshot.jpg?t='+stamp}
    $('dsnap').hidden=false;
    $('dthink').hidden=!s.last.thinking;$('think').textContent=s.last.thinking||''}
}
async function poll(){try{render(await (await fetch('/api/state')).json())}catch(e){$('cam').textContent='server unreachable';$('cam').className='pill bad'}}
poll();setInterval(poll,1000);
</script></body></html>
"""


def make_handler(cam, analyzer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GemmaVision/1"

        def log_message(self, *a):
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ALLOWED_HOSTS

        def _send(self, code, ctype, body, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, "text/plain", b"bad host")
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if path == "/health":
                return self._send(200, "text/plain", b"ok")
            if path == "/api/state":
                return self._send(200, "application/json", json.dumps(analyzer.state()).encode())
            if path == "/snapshot.jpg":
                jpg = analyzer.snapshot_jpeg
                if not jpg:
                    return self._send(404, "text/plain", b"no snapshot yet")
                return self._send(200, "image/jpeg", jpg)
            if path == "/stream.mjpg":
                return self._stream()
            self._send(404, "text/plain", b"not found")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    jpg, seq = cam.stream_jpeg()
                    if jpg is None or seq == seen:
                        time.sleep(0.03)
                        continue
                    seen = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, "text/plain", b"bad host")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, "text/plain", b"json only")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
            except ValueError:
                return self._send(400, "text/plain", b"bad json")
            if self.path == "/api/analyze":
                analyzer.request()
            elif self.path == "/api/settings":
                analyzer.configure(auto=body.get("auto"), mode=body.get("mode"))
            else:
                return self._send(404, "text/plain", b"not found")
            self._send(200, "application/json", b"{}")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="comma-separated bind addresses")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()

    cam = Camera(a.device)
    cam.start()
    analyzer = Analyzer(cam)
    analyzer.start()
    handler = make_handler(cam, analyzer)

    servers = []
    for host in [h.strip() for h in a.host.split(",") if h.strip()]:
        srv = ThreadingHTTPServer((host, a.port), handler)
        srv.daemon_threads = True
        servers.append(srv)
        print("gemma_vision listening on http://%s:%d/" % (host, a.port), flush=True)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
