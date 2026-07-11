# Jetson Orin Nano - Operator Guide

Your headless NVIDIA Jetson Orin Nano Super is a self-contained local AI box. It runs
open models entirely offline, answers questions grounded in your ThesisVault, reaches
the web when it needs to, and is usable from your laptop or your phone.

This guide covers every capability and how to use it. On the device this file lives at
`/mnt/nvme/brain/README.md`; a copy is in `~/README.md`.

---

## At a glance

| | |
|---|---|
| Board | Jetson Orin Nano Super Developer Kit (aarch64) |
| OS / JetPack | Ubuntu 22.04, JetPack R36.4.3 |
| Memory | 8 GB unified (CPU + GPU share it) |
| Storage | 234 GB NVMe at `/mnt/nvme` (about 192 GB free) plus the eMMC root |
| User / host | `user@192.168.55.1` (USB-C) or `snowbeach` on Tailscale |
| Runtime | Ollama systemd service on `127.0.0.1:11434` |
| Power | `nvpmodel` (currently 15W; Super mode available, see below) |

---

## Connecting

### From your Mac (USB-C cable)
```
ssh user@192.168.55.1
```
The USB-C link is a private point-to-point network. The Mac also shares its internet
over this link (see Internet sharing).

### From your iPhone or anywhere (Tailscale + mosh)
With the Tailscale app signed in on the phone and Blink Shell installed:
```
mosh user@jetson.<tailnet>.ts.net      # or: mosh user@100.x.x.x
```
mosh survives network drops and screen locks. Plain `ssh` works too.

### Shared session (laptop and phone see the same chat)
`brain` runs inside a persistent `tmux` session named `brain`. Detach with `Ctrl-b` then
`d`; the session and any running answer keep going. Reattach from any device by running
`brain` again, or `tmux attach -t brain`.

---

## The login menu

When you SSH or mosh in, a menu opens automatically listing everything below. Pick a
number and go. Quit the menu with `q` to reach a normal shell; type `menu` to reopen it.
The menu never appears for scripted (non-interactive) SSH commands, so deploys and copies
are unaffected.

---

## Capability 1: `brain` - your ThesisVault AI

The headline feature. A local retrieval-augmented chat over your 307-note ThesisVault.

Launch:
```
brain                 # opens the model picker, then chats
brain --model qwen3:8b  # skip the picker
```

What it does per question:
1. **Cache check** - a repeat or near-duplicate question returns instantly.
2. **Vault retrieval** - BM25 over the vault, then it follows the top notes'
   `[[wikilinks]]` to pull in linked notes (graph expansion).
3. **Autonomous web decision** - if the vault is not enough, the model itself chooses:
   - **FETCH** a specific page (a link in your notes, or a URL it is confident about) and
     synthesizes it, or
   - **SEARCH** DuckDuckGo (keyless) for open-ended or time-sensitive questions.
   A failed direct fetch falls back to a search, so it never wastes a good web decision.
4. **Cite + cache** - answers cite vault filenames or URLs, and the result is cached.

Commands inside the chat:

| Command | Effect |
|---|---|
| `/model` | Re-pick the Ollama model |
| `/web on` / `/web off` | Allow or forbid all web access (off is fully offline) |
| `/cache` | Show cache stats |
| `/cache clear` | Empty the answer cache |
| `/cache off` / `/cache on` | Toggle caching |
| `/help` | Command list |
| `/exit` | Quit (Ctrl-D and Ctrl-C also quit) |

Files: `/mnt/nvme/brain/brain.py`, `build_index.py`, `answer_cache.py`, the markdown in
`vault/`, the BM25 index at `index/bm25.pkl`, and cached answers in `cache/`.

Rebuild the index after editing notes (then clear the cache so stale answers do not shadow
new content):
```
python3 /mnt/nvme/brain/build_index.py
# inside brain:  /cache clear
```

Redeploy code or re-sync the vault from the Mac: `~/jetson-brain/deploy.sh`.

---

## Capability 2: Ollama models (plain chat, no vault)

Ten models are installed. Pick by task; the 8 GB memory is the limit, so Q4 7B-8B models
are the practical ceiling.

| Model | Size | Good for |
|---|---|---|
| qwen2.5:7b | 4.7 GB | Default. Balanced, tool-friendly, the `brain` default |
| qwen3:8b | 5.2 GB | Strongest reasoning |
| qwen3:4b | 2.5 GB | Fast, capable |
| deepseek-r1:7b | 4.7 GB | Step-by-step reasoning |
| llama3.2:3b | 2.0 GB | Fastest, most memory headroom |
| qwen2.5-coder:3b | 1.9 GB | Coding |
| deepseek-coder:1.3b | 0.8 GB | Lightweight coding |
| gemma4:12b | 7.6 GB | Largest; tight on 8 GB, use alone |
| qwen:latest | 2.3 GB | Older general |
| qwen2.5:0.5b | 0.4 GB | Tiny / experiments |

Run one directly:
```
ollama run qwen3:8b        # type /bye to exit
ollama list                # what is installed
ollama pull <model>        # download a new one (needs internet)
```

---

## Capability 3: NCDD NBA draft lab

A separate RAG plus multi-agent draft war-room in `~/ncdd-rag`.
```
cd ~/ncdd-rag
python3 ncdd_rag.py --help
python3 ncdd_rag.py --list-models
```
Corpus building and stat tooling: `make_corpus.py`, `enrich_corpus.py`, `nba_stat_lab.py`.

---

## Capability 4: Internet sharing (Mac over USB-C)

The Jetson reaches the internet through the Mac's WiFi over the USB-C link (NAT on the
Mac). It is reboot-persistent on both sides.

Check or restore connectivity:
```
/mnt/nvme/brain/jetson-online.sh test    # PASS/FAIL for ping and web
/mnt/nvme/brain/jetson-online.sh         # re-apply route + DNS (needs sudo)
```
On the Mac side, if the Jetson is offline: `~/jetson-brain/jetson-online-mac.sh start`.

Note: web features (search, fetch, `ollama pull`) need the Mac tethered. Vault chat works
fully offline. Away from the laptop, put the Jetson on WiFi for standalone internet.

---

## Capability 5: Performance and power

The Orin Nano ships throttled. For full speed:
```
sudo nvpmodel -m 2       # MAXN Super (unrestricted)
sudo jetson_clocks       # lock clocks high
nvpmodel -q              # confirm the mode
```
Menu option 6 does this for you. Watch load live:
```
tegrastats               # one line per second; Ctrl-C to stop
```

---

## Storage

| Mount | Size | Use |
|---|---|---|
| `/mnt/nvme` | 234 GB (192 free) | Models data, the brain, vault, index, cache |
| `/` (eMMC) | root | OS and home |

---

## Maintenance and troubleshooting

- **No internet**: `jetson-online.sh test`; if FAIL, run `jetson-online-mac.sh start` on
  the Mac, then `jetson-online.sh` on the Jetson.
- **Ollama not answering**: `systemctl status ollama`; restart with
  `sudo systemctl restart ollama`.
- **Out of memory / model will not load**: use a smaller model, or run only one at a time;
  8 GB is shared by CPU and GPU.
- **Index missing or stale**: `python3 /mnt/nvme/brain/build_index.py`, then `/cache clear`.
- **Cached web answer went stale** (e.g. weather): `/cache clear` inside `brain`.
- **Roll back `brain`**: v1/v2 backups sit beside it as `brain.py.v1.bak`, `brain.py.v2.bak`.
- **Reopen the menu**: type `menu`.

---

## File map

```
/mnt/nvme/brain/
  brain.py            the RAG chat agent (launched as: brain)
  build_index.py      builds the BM25 index (v2: wikilinks + urls)
  answer_cache.py     the answer cache
  jetson-menu.sh      this login menu
  jetson-online.sh    route + DNS restore / connectivity test
  README.md           this guide
  vault/              ThesisVault markdown (307 notes)
  index/bm25.pkl      the search index
  cache/              cached answers (short slug files)
~/ncdd-rag/           NCDD NBA draft lab
~/.local/bin/brain    launcher (opens the shared tmux session)
On the Mac: ~/jetson-brain/  source of truth + deploy.sh, jetson-online-mac.sh
```
