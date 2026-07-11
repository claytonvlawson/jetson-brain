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
