#!/usr/bin/env bash
# advisor-online-mac.sh -- bring the "Ask the advisor" endpoint online from the Mac.
#
# The Jetson runs advisor_server.py on 127.0.0.1:8000 (local brain, no internet
# needed with ADVISOR_WEB=0). This script, on the Mac, SSH-forwards that port to
# the Mac and opens a public HTTPS Cloudflare quick tunnel to it, so the site's
# /api/chat proxy on Vercel can reach your brain.
#
#   ./advisor-online-mac.sh start    # forward + tunnel, prints the public URL
#   ./advisor-online-mac.sh status   # show what is running + the current URL
#   ./advisor-online-mac.sh stop     # tear both down
#
# Availability: the advisor is reachable only while this stays running (Mac on,
# Jetson plugged in over USB-C). For real 24/7, put the Jetson on its own wifi
# and run the tunnel there instead. The quick tunnel's URL changes each restart;
# update JETSON_ADVISOR_URL in Vercel when it does (or switch to Tailscale Funnel
# / a named Cloudflare tunnel for a stable URL).
set -uo pipefail

JETSON="${JETSON:-user@192.168.55.1}"
PORT="${PORT:-8000}"
STATE="${HOME}/.advisor-online"
SSH_PIDF="${STATE}/ssh.pid"
CF_PIDF="${STATE}/cloudflared.pid"
CF_LOG="${STATE}/cloudflared.log"
URL_F="${STATE}/url.txt"

mkdir -p "${STATE}"

_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }
_pid()   { [[ -f "$1" ]] && cat "$1" 2>/dev/null || true; }

start() {
  command -v cloudflared >/dev/null 2>&1 || {
    echo "ERROR: cloudflared is not installed. Run: brew install cloudflared" >&2
    exit 1
  }

  # 1) SSH port-forward  Jetson:127.0.0.1:PORT  ->  Mac:localhost:PORT
  if _alive "$(_pid "$SSH_PIDF")"; then
    echo "==> SSH forward already running (pid $(_pid "$SSH_PIDF"))."
  else
    echo "==> Checking SSH to ${JETSON} ..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${JETSON}" 'echo ok' >/dev/null 2>&1; then
      echo "ERROR: cannot SSH to ${JETSON}. Is the USB-C cable connected, the Jetson" >&2
      echo "       booted, and an SSH key set up? (deploy.sh has the same preflight.)" >&2
      exit 1
    fi
    echo "==> Opening SSH forward localhost:${PORT} -> ${JETSON}:127.0.0.1:${PORT} ..."
    ssh -N -L "${PORT}:localhost:${PORT}" "${JETSON}" &
    echo $! > "${SSH_PIDF}"
    sleep 1
  fi

  # 2) Confirm the Jetson advisor is actually answering through the forward
  echo "==> Checking the advisor /health ..."
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "    advisor is answering."
  else
    echo "WARN: nothing answering /health on localhost:${PORT}." >&2
    echo "      Start it on the Jetson:  sudo systemctl start advisor" >&2
    echo "      (or:  ssh ${JETSON} 'cd /mnt/nvme/brain && ADVISOR_TOKEN=... python3 advisor_server.py')" >&2
  fi

  # 3) Cloudflare quick tunnel (no account, no DNS; ephemeral URL)
  if _alive "$(_pid "$CF_PIDF")"; then
    echo "==> Tunnel already running (pid $(_pid "$CF_PIDF"))."
  else
    echo "==> Starting Cloudflare quick tunnel ..."
    : > "${CF_LOG}"
    cloudflared tunnel --url "http://localhost:${PORT}" >"${CF_LOG}" 2>&1 &
    echo $! > "${CF_PIDF}"
  fi

  # 4) Wait for the public URL to appear in the tunnel log
  echo "==> Waiting for the public URL ..."
  url=""
  for _ in $(seq 1 30); do
    url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "${CF_LOG}" 2>/dev/null | head -1 || true)"
    [[ -n "${url}" ]] && break
    sleep 1
  done
  if [[ -z "${url}" ]]; then
    echo "ERROR: the tunnel did not report a URL. See ${CF_LOG}" >&2
    exit 1
  fi
  echo "${url}" > "${URL_F}"

  cat <<EOF

==> Advisor is ONLINE.
    Public URL : ${url}

    In Vercel, set:  JETSON_ADVISOR_URL = ${url}
    then redeploy. Stop with:  $0 stop
EOF
}

status() {
  local s c
  s="$(_pid "$SSH_PIDF")"; c="$(_pid "$CF_PIDF")"
  if _alive "$s"; then echo "ssh forward : running (pid $s)"; else echo "ssh forward : stopped"; fi
  if _alive "$c"; then echo "tunnel      : running (pid $c)"; else echo "tunnel      : stopped"; fi
  [[ -f "$URL_F" ]] && echo "last URL    : $(cat "$URL_F")"
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "advisor     : answering on localhost:${PORT}"
  else
    echo "advisor     : not answering on localhost:${PORT}"
  fi
}

stop() {
  for f in "$CF_PIDF" "$SSH_PIDF"; do
    p="$(_pid "$f")"
    if _alive "$p"; then
      kill "$p" 2>/dev/null && echo "==> stopped pid $p"
    fi
    rm -f "$f"
  done
  echo "==> Advisor offline."
}

case "${1:-}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
