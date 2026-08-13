# brain - a local RAG chat on the Jetson Orin Nano

## What this is

`brain` is a small command line chat agent that runs entirely on an NVIDIA
Jetson Orin Nano 8GB. You pick any Ollama model you have pulled, chat with it,
and every question is first grounded against your ThesisVault notes using a
local BM25 retrieval index. When the vault does not have the answer, the agent
falls back to a DuckDuckGo web search (via the `ddgs` package) and feeds the
fetched page text back to the model. No cloud API, no keys, no cost.

Pieces:

- `brain.py` - the chat loop (retrieval, web fallback, Ollama streaming).
- `build_index.py` - builds `index/bm25.pkl` from the markdown in `vault/`.
- `vault/` - the ThesisVault markdown, synced from the Mac.
- `brain_skills.py` - the optional bounded skill loop (`BRAIN_SKILLS=1`).
- `brain_ollama_shim.py` - a fake Ollama API on port 11435 so Obsidian's Smart
  Chat plugin can talk to the RAG unpatched.
- `obsidian_rank.py` / `export_obsidian_vectors.py` - the Smart Connections
  vector signal, note level and (optionally) paragraph level.
- `install.sh` - installs Python deps, builds the index, drops the `brain` launcher.
- `jetson-online.sh` / `jetson-online.service` - restore the Jetson internet route.

Install layout on the Jetson:

```
/mnt/nvme/brain/
  brain.py
  build_index.py
  vault/                 rsynced markdown
  index/bm25.pkl         built by build_index.py
~/.local/bin/brain       launcher: exec python3 /mnt/nvme/brain/brain.py "$@"
```

## One-time setup

From the Mac, with the Jetson connected over the USB-C cable:

```bash
./deploy.sh
```

This checks the SSH link, creates the remote layout, rsyncs the vault markdown
(md files only), copies the scripts, and runs `install.sh` on the Jetson, which
installs dependencies and builds the BM25 index.

Config vars you can override:

```bash
JETSON=user@192.168.55.1 \
VAULT_SRC="$HOME/Desktop/Dream/ThesisVault" \
REMOTE=/mnt/nvme/brain \
./deploy.sh
```

### Internet sharing (so the web fallback works)

The Jetson reaches the internet through the Mac over USB-C. Enable it once:

- On the Mac, share WiFi (en0) over the link interface (en10):

  ```bash
  ./jetson-online-mac.sh start
  ```

- On the Jetson, set the default route and install the boot service:

  ```bash
  bash /mnt/nvme/brain/install.sh --net
  ```

## Daily use

```bash
ssh user@192.168.55.1
brain
```

In the chat:

- `/model` - list your Ollama models and switch the active one.
- `/web` - toggle DuckDuckGo web fallback on or off.
- `/exit` - quit.

Type a normal question and the agent retrieves from the vault, optionally
searches the web, and streams the answer.

## Obsidian Smart Chat against the brain (the Ollama shim)

`brain_ollama_shim.py` serves a small subset of the Ollama HTTP API on port
11435 and answers every chat request by running the full RAG pipeline
(`brain.handle_question`) instead of a raw model. Obsidian's Smart Connections
Smart Chat plugin therefore talks to the vault-grounded brain with no plugin
patching at all: as far as it knows it is talking to a normal Ollama server.
Port 11435 and never 11434, because the real Ollama owns 11434 and `brain.py`
generates against it.

Install the service on the Jetson (one time):

```bash
ssh user@192.168.55.1
sudo cp /mnt/nvme/brain/brain-ollama-shim.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now brain-ollama-shim.service
systemctl status brain-ollama-shim.service
```

The unit resolves the tailnet address at startup with `tailscale ip -4` and
binds to it, so Obsidian on the Mac reaches it directly and an IP change does
not need a file edit.

Exact click path in Obsidian:

1. Settings (gear, bottom left).
2. Smart Connections (or Smart Chat) in the left sidebar under Community plugins.
3. Chat model settings, then the **Model Platform** dropdown.
4. Pick **Ollama (Local)**.
5. In the **Ollama host** field type the Jetson URL:
   `http://100.x.x.x:11435` (use `http://127.0.0.1:11435` when Obsidian
   itself runs on the Jetson). If `SHIM_TOKEN` is set, append the token as a
   path segment: `http://100.x.x.x:11435/t/<token>`. The plugin sends no
   headers on the chat request, so a token has to ride the URL.
6. The **Model** dropdown now lists two entries. Pick **jetson-brain** for the
   fast tier (llama3.2:3b) or **jetson-brain-deep** for the deep tier
   (qwen2.5:7b).

If the dropdown says "Failed to fetch models", there are two causes and they
produce identical text.

1. The shim is unreachable. Check `systemctl status brain-ollama-shim.service`
   on the Jetson and that the Mac is on the tailnet. Confirm directly from the
   Mac (add `/t/<token>` before the path when `SHIM_TOKEN` is set):

   ```bash
   curl http://100.x.x.x:11435/api/tags
   curl -X POST http://100.x.x.x:11435/api/show \
        -d '{"model":"jetson-brain:latest"}'
   ```

   The plugin calls `/api/tags` once and then `/api/show` once per listed
   model, so both have to answer.

2. The Mac cannot reach `https://models.dev/api.json`. The plugin fetches that
   index to enrich model metadata and then reads the cached result without a
   null guard, so a failed fetch throws inside the same `try` that wraps the
   Ollama calls and yields the same "Failed to fetch models" string even though
   the shim answered perfectly. Captive portals, DNS filtering, or Obsidian
   starting before Wi-Fi comes up all trigger it. Open the developer console
   (Cmd-Opt-I) and look for `models.dev fetch failed`. One successful model-list
   refresh while the Mac has internet populates the plugin's in-memory cache for
   the rest of the Obsidian session and unblocks the dropdown.

Shim environment variables (all optional, set them in the service unit):

| Variable | Default | What it does |
|---|---|---|
| `SHIM_HOST` | `127.0.0.1` | Bind address (the unit overrides it with the tailnet IP) |
| `SHIM_PORT` | `11435` | Listen port; never use 11434 |
| `SHIM_TOKEN` | empty | Shared secret; empty disables the check |
| `SHIM_WEB` | `0` | `1` allows the web fallback inside Obsidian answers |
| `SHIM_LINKS` | `1` | `0` suppresses the `[[wikilink]]` sources footer |
| `SHIM_MAX_Q` | `500` | Max question characters accepted |
| `SHIM_MAX_BODY` | `200000` | Max request body bytes |
| `SHIM_QUEUE_TIMEOUT` | `120` | Seconds a queued request waits for the single worker |
| `SHIM_MAX_HISTORY_CHARS` | `4000` | Cap on replayed chat history (num_ctx is 6144) |
| `SHIM_FAST_NAME` | `jetson-brain:latest` | Name shown for the fast tier |
| `SHIM_DEEP_NAME` | `jetson-brain-deep:latest` | Name shown for the deep tier |

## Optional features and their environment flags

Every one of these is default OFF and is read by `brain.py` at import time, so
set it in the environment before launching `brain`.

| Variable | Default | What it does |
|---|---|---|
| `BRAIN_OBSIDIAN_RANK` | `0` | Expand context with notes the Smart Connections plugin scores as related, alongside the `[[wikilink]]` expansion |
| `BRAIN_OBSIDIAN_TOP_K` | `5` | Related notes considered per seed note |
| `BRAIN_OBSIDIAN_MAX_EXPAND` | `3` | Related-note chunks added per question |
| `BRAIN_OBSIDIAN_BLOCKS` | `0` | Pick an expanded note's BEST paragraph instead of its first chunk, using the plugin's block-level vectors. Needs a vector export built with `export_obsidian_vectors.py --blocks` (1.3 MB grows to 23 MB) |
| `BRAIN_OBSIDIAN_BLOCK_CANDIDATES` | `5` | Top blocks tried before giving up and keeping the first chunk |
| `BRAIN_SKILLS` | `0` | Run the bounded skill loop when the vault-first pass says INSUFFICIENT (see below) |
| `BRAIN_SKILLS_MAX_STEPS` | `3` | Hard cap on skill steps per question |
| `BRAIN_SKILLS_DEADLINE` | `45` | Seconds the whole loop may take |
| `BRAIN_SKILLS_KEEP_K` | `10` | Chunks the loop's wider vault retry keeps (phase 1 keeps 6) |
| `BRAIN_SKILLS_NOTE_CHARS` | `3000` | Cap on one note pulled in by `get_note` |
| `BRAIN_SKILLS_VERBOSE` | `0` | Log every selection and observation to stderr |
| `BRAIN_TIMING` | `0` | One machine-parseable stage-timing line per question on stderr |

With `BRAIN_TIMING=1` the skill loop reports its own stage as `skills=<seconds>`
between `phase1` and `route`.

## The skill loop

`brain.py` answers from the vault first. When that pass emits INSUFFICIENT, the
old pipeline had exactly one move: a single routing call that chose FETCH or
SEARCH and then answered from the web. With `BRAIN_SKILLS=1` a bounded loop
(`brain_skills.py`) runs first and can take up to three steps, choosing between
`search_vault`, `get_note`, `web_search`, `fetch_url`, and `reason`. It uses the
same single-line prefix protocol as the routing call (`SKILL: <name> | <arg>`),
because a 3B model cannot reliably emit JSON tool calls, and anything
unparseable falls back to the safe default rather than failing the turn. If the
loop produces nothing, the existing web path runs exactly as before.

Worked example, an out-of-vault question with web on:

```
you> what did the Clerc Center's 2024 annual report say about enrollment?

  [ok] BM25 retrieve - 6 chunks - hybrid - top score 8.3 - Clerc Center.md
  [ok] llama3.2:3b - vault insufficient - answered by skill loop

step 1  SKILL: search_vault | Clerc Center enrollment numbers annual report
        -> 10 chunks, still INSUFFICIENT (the vault has the mission notes,
           not the enrollment table)
step 2  SKILL: web_search | Clerc Center 2024 annual report enrollment
        -> 3 excerpts from gallaudet.edu, answered

The 2024 report gives enrollment for KDES and MSSD separately ...

[provenance - not from the vault: answered from live web search results,
treat as unverified]
sources: https://gallaudet.edu/clerc-center/...
```

The provenance footer is the point of the design: an answer that did not come
from the vault says so in its own text, in the same visual convention as the
existing `[grounding check - ...]` footer, so the terminal, the phone web UI,
and the eval harness all see it.

## Reboot persistence

- Jetson side: `jetson-online.service` (a systemd unit installed by
  `install.sh --net`) runs `jetson-online.sh` on boot to restore the default
  route through the Mac. Ollama itself already runs as a systemd service at
  `http://127.0.0.1:11434`, so it comes back on its own after a reboot.
- Mac side: internet sharing is not automatic. After a Mac reboot or after
  unplugging and replugging the cable, run `./jetson-online-mac.sh start` again
  on the Mac to re-enable NAT from en0 to en10.

## Troubleshooting

- No internet on the Jetson (web search returns nothing, downloads fail):
  1. On the Mac: `./jetson-online-mac.sh start`
  2. On the Jetson: `bash /mnt/nvme/brain/jetson-online.sh`
  3. Test on the Jetson: `ping -c1 1.1.1.1` then `ping -c1 duckduckgo.com`.
  The web fallback is optional. If it stays offline, the agent still answers
  from the vault; you can also disable it in-chat with `/web`.

- `ollama` not found or model calls fail: confirm the service is up with
  `systemctl status ollama` and `curl http://127.0.0.1:11434/api/tags`. Pull a
  model if you have none, for example `ollama pull llama3.2:3b`. `brain.py`
  falls back to the REST API automatically if the Python client misbehaves.

- Index missing or stale (`index/bm25.pkl` not found, or vault changed):
  rebuild it on the Jetson:

  ```bash
  cd /mnt/nvme/brain && python3 build_index.py
  ```

  Re-run `./deploy.sh` from the Mac to resync the vault and rebuild in one step.

- Cannot reach the Jetson from the Mac (`deploy.sh` preflight fails): check the
  USB-C cable, confirm the Mac link interface with
  `ifconfig en10 | grep 'inet '` (expect 192.168.55.100), and confirm you can
  `ssh user@192.168.55.1`.
