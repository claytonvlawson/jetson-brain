#!/usr/bin/env bash
# advisor-funnel-jetson.sh -- expose the local advisor to the public internet via
# Tailscale Funnel, FROM THE JETSON itself. Run this on the Jetson (not the Mac).
#
# The advisor server binds 127.0.0.1:8000; Funnel proxies https://<node>.<tailnet>.ts.net
# straight to it and gives a STABLE URL that survives reboots and does not depend
# on the Mac. Requires the Jetson to have its own internet (run jetson-wifi.sh)
# so the tunnel stays up when the Mac sleeps.
#
#   ./advisor-funnel-jetson.sh start    # enable Funnel -> localhost:8000, print the URL
#   ./advisor-funnel-jetson.sh status   # tailscale state + funnel config + advisor health
#   ./advisor-funnel-jetson.sh stop     # disable Funnel
#   ./advisor-funnel-jetson.sh url      # print the public https URL only
#
# One-time prerequisites (done once, in a browser, not scriptable from here):
#   curl -fsSL https://tailscale.com/install.sh | sh
#   sudo tailscale up
#   Then at login.tailscale.com: enable HTTPS certificates AND Funnel for the
#   tailnet, and give this node the "funnel" nodeAttr in the ACL policy.
set -uo pipefail

PORT="${PORT:-8000}"
TS="$(command -v tailscale || true)"

_need_ts() {
  if [[ -z "$TS" ]]; then
    cat >&2 <<'EOF'
ERROR: tailscale is not installed on this Jetson. Install + log in first:
  curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up
Then enable Funnel for the tailnet at login.tailscale.com (HTTPS certs on, and
this node needs the "funnel" nodeAttr in the ACL policy).
EOF
    exit 1
  fi
}

_ts_json() { "$TS" status --json 2>/dev/null; }

_ts_state() {
  _ts_json | python3 -c 'import sys,json;print(json.load(sys.stdin).get("BackendState",""))' 2>/dev/null || true
}

_tsnet_url() {
  _ts_json | python3 -c 'import sys,json;d=json.load(sys.stdin);print("https://"+d["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true
}

_advisor_ok() { curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; }

start() {
  _need_ts
  if [[ "$(_ts_state)" != "Running" ]]; then
    echo "ERROR: tailscale is not Running. Run: sudo tailscale up" >&2
    exit 1
  fi
  if ! _advisor_ok; then
    echo "WARN: advisor not answering on localhost:${PORT}. Start it first:" >&2
    echo "        sudo systemctl start advisor" >&2
  fi

  echo "==> Enabling Tailscale Funnel  https://<node>:443 -> http://localhost:${PORT} ..."
  # --bg persists the config; tailscaled restores it across reboots.
  if ! sudo "$TS" funnel --bg "${PORT}"; then
    cat >&2 <<'EOF'

ERROR: could not enable Funnel. Most common cause: Funnel is not enabled for this
tailnet yet, or this node lacks the "funnel" nodeAttr. Fix at login.tailscale.com
(turn Funnel on, enable HTTPS certificates, add the node to the funnel policy),
then re-run. If your tailscale version rejects `funnel --bg <port>`, check the
exact syntax with: tailscale funnel --help
EOF
    exit 1
  fi

  local url; url="$(_tsnet_url)"
  cat <<EOF

==> Advisor is ONLINE via Tailscale Funnel (stable URL, survives reboots + Mac sleep).
    Public URL : ${url}

    In Vercel set:  JETSON_ADVISOR_URL = ${url}
    then redeploy. This URL does NOT change. Disable with: $0 stop
EOF
}

status() {
  _need_ts
  local st; st="$(_ts_state)"
  if [[ "$st" == "Running" ]]; then
    echo "tailscale : Running  ($(_tsnet_url))"
  else
    echo "tailscale : ${st:-not running}  (sudo tailscale up)"
  fi
  echo "--- funnel config ---"
  "$TS" funnel status 2>/dev/null || echo "(none / not supported by this version)"
  if _advisor_ok; then
    echo "advisor   : answering on localhost:${PORT}"
  else
    echo "advisor   : NOT answering on localhost:${PORT}  (sudo systemctl start advisor)"
  fi
}

stop() {
  _need_ts
  echo "==> Disabling Tailscale Funnel ..."
  # serve reset clears all serve/funnel config; on this node that is only the
  # advisor funnel. (Older CLIs also accept `tailscale funnel off`.)
  sudo "$TS" serve reset 2>/dev/null || sudo "$TS" funnel off 2>/dev/null || true
  echo "==> Funnel off."
}

case "${1:-}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  url)    _need_ts; _tsnet_url ;;
  *) echo "usage: $0 {start|stop|status|url}" >&2; exit 2 ;;
esac
