#!/usr/bin/env bash
# brain-web.sh -- run the phone-friendly brain web UI and expose it to YOUR
# tailnet (only your own devices) over HTTPS via Tailscale Serve. Run on the Jetson.
#
# brain_web.py binds 127.0.0.1:8899; `tailscale serve` fronts it with a real
# HTTPS cert at https://<node>.<tailnet>.ts.net/ -- reachable from your iPhone
# (Tailscale on) but NOT the public internet (that would be Funnel). All state is
# tailnet-private; the tailnet ACL is the access control.
#
#   sudo ./brain-web.sh setup     # install+enable the boot service, serve :443 -> :8899, print URL
#   ./brain-web.sh status         # service + serve config + local health
#   ./brain-web.sh url            # print the https URL only
#   sudo ./brain-web.sh restart   # restart the server
#   sudo ./brain-web.sh stop      # stop the service and remove the serve mapping
#
# NOTE: this uses tailnet HTTPS on port 443. Your public advisor Funnel also
# wants 443, so the two cannot both own 443 at once. Idempotent; safe to re-run.
#
# One-time prerequisite (in a browser, not scriptable here): at login.tailscale.com
# enable HTTPS certificates for the tailnet. `tailscale serve` needs it to mint
# the cert for <node>.<tailnet>.ts.net.
set -uo pipefail

REMOTE="/mnt/nvme/brain"
UNIT="brain-web.service"
PORT="${BRAIN_WEB_PORT:-8899}"          # local port brain_web.py binds
TS="$(command -v tailscale || true)"
HERE="$(cd "$(dirname "$0")" && pwd)"

r=$'\033[0m'; b=$'\033[1m'; gr=$'\033[32m'; ye=$'\033[33m'; rd=$'\033[31m'; dm=$'\033[2m'
say(){ printf '%b\n' "$*"; }

_need_ts(){ [ -n "$TS" ] || { say "${rd}ERROR${r} tailscale is not installed on this Jetson."; exit 1; }; }
_url(){ "$TS" status --json 2>/dev/null | python3 -c 'import sys,json;print("https://"+json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null; }
_health(){ curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null; }

setup(){
  _need_ts
  say "${b}==> Installing ${UNIT} (boot service) ...${r}"
  sudo cp "${HERE}/${UNIT}" "/etc/systemd/system/${UNIT}"
  sudo systemctl daemon-reload
  sudo systemctl enable --now "${UNIT}"

  say "${b}==> Waiting for the server on 127.0.0.1:${PORT} ...${r}"
  local up=0 i
  for i in $(seq 1 30); do
    if _health >/dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  if [ "$up" -eq 1 ]; then
    say "  ${gr}OK${r} server is answering locally."
  else
    say "  ${ye}WARN${r} server not up yet. Check: ${dm}journalctl -u ${UNIT} -n 40 --no-pager${r}"
  fi

  say "${b}==> Enabling Tailscale Serve  https://<node>/ (443) -> 127.0.0.1:${PORT} ...${r}"
  if ! sudo "$TS" serve --bg "${PORT}"; then
    cat >&2 <<EOF
  ERROR: could not enable Tailscale Serve. Most common cause: HTTPS certificates
  are not enabled for this tailnet yet. Turn them on at login.tailscale.com
  (Settings -> Keys/Features -> HTTPS Certificates), then re-run. If your CLI
  rejects the syntax, check: tailscale serve --help
EOF
    exit 1
  fi

  local u; u="$(_url)"
  say ""
  say "${b}${gr}==> Brain web UI is LIVE (tailnet-only, survives reboots + Mac sleep).${r}"
  say "    On your iPhone (Tailscale ON), open:  ${b}${u}${r}"
  say "    ${dm}Tip: Share -> Add to Home Screen for an app-like icon.${r}"
}

status(){
  _need_ts
  say "${b}service${r} : $(systemctl is-active "${UNIT}" 2>/dev/null || echo inactive)  ($(systemctl is-enabled "${UNIT}" 2>/dev/null || echo disabled))"
  if _health >/dev/null 2>&1; then say "${b}local${r}   : ${gr}up${r} $(_health)"; else say "${b}local${r}   : ${rd}down${r} on 127.0.0.1:${PORT}"; fi
  say "${b}serve${r}   :"
  "$TS" serve status 2>/dev/null | sed 's/^/  /' || say "  (none)"
  say "${b}url${r}     : $(_url)"
}

url(){ _need_ts; _url; }

restart(){ sudo systemctl restart "${UNIT}" && say "restarted ${UNIT}"; }

stop(){
  _need_ts
  say "==> Stopping ${UNIT} ..."
  sudo systemctl disable --now "${UNIT}" 2>/dev/null || true
  # Remove the serve mapping, but do NOT clobber an advisor Funnel if one exists.
  local cfg; cfg="$("$TS" serve status 2>/dev/null || true)"
  if printf '%s' "$cfg" | grep -q "${PORT}"; then
    if printf '%s' "$cfg" | grep -qiE 'funnel'; then
      say "  ${ye}WARN${r} a Funnel config is also present. Not running 'serve reset' so"
      say "  the advisor stays up. Remove the brain-web mapping by hand if you want:"
      say "    ${dm}sudo tailscale serve reset${r}  (clears ALL serve/funnel config)"
    else
      sudo "$TS" serve reset 2>/dev/null || true
      say "  serve mapping removed."
    fi
  fi
  say "==> Stopped."
}

case "${1:-}" in
  setup|start) setup ;;
  status)      status ;;
  url)         url ;;
  restart)     restart ;;
  stop)        stop ;;
  *) say "usage: sudo $0 {setup|status|url|restart|stop}"; exit 2 ;;
esac
