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
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brain  # safe to import: main() sits behind an __main__ guard
import brain_web_note

HOST = os.environ.get("BRAIN_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("BRAIN_WEB_PORT", "8899"))
FAST_MODEL = os.environ.get("BRAIN_WEB_MODEL", brain.FAST_MODEL)
DEEP_MODEL = os.environ.get("BRAIN_WEB_DEEP_MODEL", brain.DEEP_MODEL)
MAX_Q = int(os.environ.get("BRAIN_WEB_MAX_Q", "500"))            # cap question length
QUEUE_TIMEOUT = float(os.environ.get("BRAIN_WEB_QUEUE_TIMEOUT", "120"))  # deep model can be slow
# GET /note reads from the same folder deploy.sh rsyncs the vault into and the
# same root chunk `source` paths are relative to (BRAIN_VAULT overrides).
VAULT_DIR = brain.SCHOLAR_VAULT_DIR

print("Loading the vault index...", flush=True)
CHUNKS, BM25, META = brain.load_index()
INDEX_MTIME = brain.index_mtime()
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
  .a .srcs .srow{display:flex;align-items:flex-start;gap:8px;margin-top:3px}
  .a .srcs .srow a.src{flex:1;margin-top:0;padding:2px 0}
  .a .srcs .srow a.src:active{opacity:.6}
  .a .srcs .view{
    flex:none;background:none;border:1px solid var(--line);color:var(--muted);border-radius:8px;
    font:inherit;font-size:11px;padding:2px 8px;cursor:pointer;line-height:1.5
  }
  .a .srcs .view:active{border-color:var(--accent2);color:var(--accent2)}
  .meta{font-size:11px;color:var(--muted);margin-top:8px}
  /* note sheet (GET /note viewer) */
  #shade{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:20}
  #note{
    position:fixed;left:0;right:0;bottom:0;top:max(48px,env(safe-area-inset-top));z-index:21;
    background:var(--panel);border:1px solid var(--line);border-bottom:none;border-radius:18px 18px 0 0;
    display:flex;flex-direction:column;box-shadow:0 -10px 40px rgba(0,0,0,.5)
  }
  #note .nh{display:flex;align-items:flex-start;gap:10px;padding:14px 16px 10px;border-bottom:1px solid var(--line)}
  #note .nh .nt{flex:1;min-width:0}
  #note .nh h2{margin:0;font-size:17px;font-weight:600;line-height:1.3;word-wrap:break-word}
  #note .nh .np{color:var(--muted);font-size:11.5px;word-break:break-all;margin-top:3px}
  #note .nh .tags{margin-top:5px;display:flex;flex-wrap:wrap;gap:5px}
  #note .nh .tag{font-size:10.5px;color:var(--accent);border:1px solid #4a3a1a;border-radius:6px;padding:0 6px}
  #note .nh .close{
    flex:none;background:var(--user);border:1px solid var(--line);color:var(--text);border-radius:10px;
    font:inherit;font-size:13px;padding:6px 12px;cursor:pointer
  }
  #note .nb{flex:1;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:12px 16px calc(16px + env(safe-area-inset-bottom));font-size:15px;line-height:1.55}
  #note .nb .obs{display:inline-block;margin-bottom:12px;color:var(--accent2);text-decoration:none;font-size:13px}
  #note .nb .obs:before{content:"↗ "}
  #note .nb h1,#note .nb h2,#note .nb h3,#note .nb h4{margin:16px 0 6px;line-height:1.3}
  #note .nb h1{font-size:20px} #note .nb h2{font-size:17px} #note .nb h3{font-size:15.5px} #note .nb h4{font-size:14.5px}
  #note .nb p{margin:0 0 10px}
  #note .nb ul,#note .nb ol{margin:0 0 10px;padding-left:22px}
  #note .nb li{margin:2px 0}
  #note .nb blockquote{margin:0 0 10px;padding:4px 12px;border-left:3px solid var(--accent);color:var(--muted)}
  #note .nb code{background:var(--user);border-radius:5px;padding:1px 5px;font-size:13px}
  #note .nb pre{background:var(--user);border-radius:9px;padding:10px 12px;overflow-x:auto;font-size:12.5px;line-height:1.45}
  #note .nb pre code{background:none;padding:0}
  #note .nb a{color:var(--accent2)}
  #note .nb .wiki{color:var(--accent2);border-bottom:1px dotted var(--accent2)}
  #note .nb hr{border:none;border-top:1px dashed var(--line);margin:14px 0}
  #note .nb .trunc{color:var(--muted);font-size:12px;margin-top:14px}
  #note .nb .nerr{color:var(--danger)}
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
      <label><input type="checkbox" id="parent"> Parent mode</label>
    </div>
  </form>
  <div id="shade" hidden></div>
  <section id="note" hidden aria-label="Note viewer">
    <div class="nh">
      <div class="nt">
        <h2 id="ntitle"></h2>
        <div class="np" id="npath"></div>
        <div class="tags" id="ntags"></div>
      </div>
      <button class="close" id="nclose" type="button">Close</button>
    </div>
    <div class="nb" id="nbody"></div>
  </section>
<script>
(function(){
  var log=document.getElementById('log'), f=document.getElementById('f'),
      q=document.getElementById('q'), send=document.getElementById('send'),
      sub=document.getElementById('sub'), hint=document.getElementById('hint'),
      web=document.getElementById('web'), deep=document.getElementById('deep'),
      parent=document.getElementById('parent');
  var busy=false;

  function scroll(){ log.scrollTop=log.scrollHeight; }
  function el(cls,txt){ var d=document.createElement('div'); d.className=cls; if(txt!=null) d.textContent=txt; return d; }

  fetch('/health').then(function(r){return r.json();}).then(function(d){
    sub.textContent=(d.chunks||0)+' notes · '+(d.fast||'');
  }).catch(function(){ sub.textContent='offline'; });

  q.addEventListener('input',function(){ q.style.height='auto'; q.style.height=Math.min(q.scrollHeight,120)+'px'; });

  // Vault name for obsidian:// links; brain_web inlines brain.VAULT_NAME here.
  var VAULT=__VAULT_NAME__;
  function obsidianUri(src){
    var file=src.replace(/\.md$/i,'');
    return 'obsidian://open?vault='+encodeURIComponent(VAULT)+'&file='+encodeURIComponent(file);
  }

  // ---- note sheet (GET /note) -------------------------------------------
  var shade=document.getElementById('shade'), note=document.getElementById('note'),
      ntitle=document.getElementById('ntitle'), npath=document.getElementById('npath'),
      ntags=document.getElementById('ntags'), nbody=document.getElementById('nbody'),
      nclose=document.getElementById('nclose');

  function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  // Minimal Obsidian-flavoured markdown -> HTML. Input is escaped FIRST, so
  // nothing in a note can inject markup; the only tags here are ours.
  function inline(s){
    s=s.replace(/`([^`]+)`/g,function(_,c){ return '<code>'+c+'</code>'; });
    s=s.replace(/!\[\[([^\]]+)\]\]/g,function(_,t){ return '<span class="wiki">'+t+'</span>'; });
    s=s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g,'<span class="wiki" title="$1">$2</span>');
    s=s.replace(/\[\[([^\]]+)\]\]/g,'<span class="wiki">$1</span>');
    s=s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
    // Bare URLs. The lookbehind class skips URLs already inside an href="..."
    // or a >text< from the [text](url) pass just above.
    s=s.replace(/(^|[^\w"'=>])(https?:\/\/[^\s<]+[^\s<.,;:)])/g,'$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    s=s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/__([^_]+)__/g,'<b>$1</b>');
    s=s.replace(/(^|[^*\w])\*([^*\n]+)\*/g,'$1<i>$2</i>').replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g,'$1<i>$2</i>');
    s=s.replace(/==([^=]+)==/g,'<mark>$1</mark>');
    return s;
  }
  function md(text){
    var lines=esc(text).split('\n'), out=[], i=0, list=null, para=[];
    function flushPara(){ if(para.length){ out.push('<p>'+inline(para.join(' '))+'</p>'); para=[]; } }
    function closeList(){ if(list){ out.push('</'+list+'>'); list=null; } }
    while(i<lines.length){
      var L=lines[i];
      if(/^```/.test(L)){ flushPara(); closeList(); var buf=[]; i++;
        while(i<lines.length && !/^```/.test(lines[i])){ buf.push(lines[i]); i++; }
        out.push('<pre><code>'+buf.join('\n')+'</code></pre>'); i++; continue; }
      var h=L.match(/^(#{1,6})\s+(.*)$/);
      if(h){ flushPara(); closeList(); var n=Math.min(h[1].length,4); out.push('<h'+n+'>'+inline(h[2])+'</h'+n+'>'); i++; continue; }
      if(/^\s*(-{3,}|\*{3,})\s*$/.test(L)){ flushPara(); closeList(); out.push('<hr>'); i++; continue; }
      var bq=L.match(/^&gt;\s?(.*)$/);
      if(bq){ flushPara(); closeList(); var q=[];
        while(i<lines.length && /^&gt;/.test(lines[i])){ q.push(lines[i].replace(/^&gt;\s?/,'').replace(/^\[!\w+\][-+]?\s*/,'')); i++; }
        out.push('<blockquote>'+inline(q.join(' '))+'</blockquote>'); continue; }
      var ul=L.match(/^\s*[-*+]\s+(.*)$/), ol=L.match(/^\s*\d+[.)]\s+(.*)$/);
      if(ul||ol){ flushPara(); var kind=ul?'ul':'ol';
        if(list!==kind){ closeList(); out.push('<'+kind+'>'); list=kind; }
        var item=(ul||ol)[1].replace(/^\[[ xX]\]\s*/,function(m){ return m.indexOf('x')>=0||m.indexOf('X')>=0?'☑ ':'☐ '; });
        out.push('<li>'+inline(item)+'</li>'); i++; continue; }
      if(!L.trim()){ flushPara(); closeList(); i++; continue; }
      closeList(); para.push(L.trim()); i++;
    }
    flushPara(); closeList();
    return out.join('\n');
  }

  function closeNote(){ note.hidden=true; shade.hidden=true; }
  nclose.addEventListener('click',closeNote);
  shade.addEventListener('click',closeNote);
  document.addEventListener('keydown',function(e){ if(e.key==='Escape' && !note.hidden) closeNote(); });

  function openNote(src){
    ntitle.textContent=src.replace(/^.*\//,'').replace(/\.md$/i,'');
    npath.textContent=src; ntags.innerHTML=''; nbody.innerHTML='<p style="color:var(--muted)">loading...</p>';
    note.hidden=false; shade.hidden=false; nbody.scrollTop=0;
    fetch('/note?path='+encodeURIComponent(src))
    .then(function(r){ return r.json().then(function(j){ return {ok:r.ok,status:r.status,j:j}; }); })
    .then(function(res){
      if(!res.ok){ nbody.innerHTML='<p class="nerr">'+esc((res.j&&res.j.error)||('Error '+res.status))+'</p>'; return; }
      var d=res.j;
      ntitle.textContent=d.title||ntitle.textContent;
      (d.tags||[]).forEach(function(t){ var s=document.createElement('span'); s.className='tag'; s.textContent='#'+t; ntags.appendChild(s); });
      var obs='<a class="obs" href="'+esc(obsidianUri(d.path||src))+'">Open in Obsidian</a>';
      nbody.innerHTML=obs+md(d.body||'')+(d.truncated?'<p class="trunc">Note truncated for the phone. Open it in Obsidian for the full text.</p>':'');
      nbody.scrollTop=0;
    })
    .catch(function(){ nbody.innerHTML='<p class="nerr">Could not load the note. Is Tailscale on?</p>'; });
  }

  function addAnswer(d){
    var a=el('msg a'); a.textContent=d.answer||'';
    if(d.sources && d.sources.length){
      var s=el('srcs'); var b=document.createElement('b'); b.textContent='sources'; s.appendChild(b);
      d.sources.forEach(function(src){
        if(/^https?:/.test(src)){
          var w=document.createElement('a'); w.href=src; w.target='_blank'; w.rel='noopener'; w.textContent=src; s.appendChild(w); return;
        }
        if(!/\.md$/i.test(src)){ var sp=document.createElement('span'); sp.className='src'; sp.textContent=src; s.appendChild(sp); return; }
        // Tap the path: Obsidian opens the note (phone or Mac, vault must be
        // present there). "view": the in-page sheet served from the Jetson.
        var row=el('srow');
        var link=document.createElement('a'); link.className='src'; link.href=obsidianUri(src); link.textContent=src;
        var v=document.createElement('button'); v.type='button'; v.className='view'; v.textContent='view';
        v.addEventListener('click',function(e){ e.preventDefault(); openNote(src); });
        row.appendChild(link); row.appendChild(v); s.appendChild(row);
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
      body:JSON.stringify({question:text, web:web.checked, deep:deep.checked,
      parent:parent.checked})})
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


# PAGE is a raw string full of CSS braces, so no str.format: one placeholder,
# swapped for a JSON string literal so the vault name is JS-safe.
PAGE_HTML = PAGE.replace("__VAULT_NAME__", json.dumps(brain.VAULT_NAME))


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
            return self._send(200, PAGE_HTML, ctype="text/html; charset=utf-8")
        if path == "/health":
            return self._send(200, {"ok": True, "chunks": len(CHUNKS),
                                    "fast": FAST_MODEL, "deep": DEEP_MODEL})
        if path == "/note":
            # Serve one vault note for the in-page viewer. No GPU lock: reading
            # a file must stay responsive while a question is running. The
            # path guard lives in brain_web_note.note_path (traversal, symlink
            # escape, non-.md all rejected).
            qs = urllib.parse.parse_qs(self.path.partition("?")[2])
            rel = (qs.get("path") or [""])[0]
            if not brain_web_note.path_shape_ok(rel):
                return self._send(400, {"error": "bad path"})
            try:
                note = brain_web_note.load_note(VAULT_DIR, rel)
            except Exception as exc:  # unreadable file: report, do not crash the server
                print("[brain_web] note error:", exc, flush=True)
                return self._send(500, {"error": "could not read that note"})
            if note is None:
                return self._send(404, {"error": "note not found"})
            return self._send(200, note)
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
            parent = bool(data.get("parent", False))
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
            # Auto-learn hot-reload (safe here: the GPU lock makes this the
            # only in-flight question, so swapping the globals cannot race a
            # concurrent handle_question).
            global CHUNKS, BM25, META, INDEX_MTIME
            CHUNKS, BM25, META, INDEX_MTIME, _ = brain.maybe_reload_index(
                CHUNKS, BM25, META, INDEX_MTIME)
            answer, sources = brain.handle_question(
                model, question, [], BM25, CHUNKS, META, web, brain._NullRun(),
                persona="parent" if parent else "default")
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
