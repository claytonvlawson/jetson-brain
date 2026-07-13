#!/usr/bin/env bash
# brain-web.sh -- run the phone-friendly brain web UI on the Jetson and reach it
# from your iPhone over Tailscale. Run on the Jetson.
#
# The boot service (brain-web.service) binds brain_web.py to this node's Tailscale
# IP, so on your phone (Tailscale ON) you just open  http://<tailscale-ip>:8899/.
# That needs no MagicDNS and no `tailscale serve` -- handy because iOS Safari
# often will not resolve the .ts.net name. A Tailscale IP is only routable inside
# your tailnet, so the page stays private to your own devices.
#
#   sudo ./brain-web.sh setup      # install + enable the boot service, print the URL
#   ./brain-web.sh status          # service state + health + URL
#   ./brain-web.sh url             # print the http URL only
#   sudo ./brain-web.sh restart    # restart the server
#   sudo ./brain-web.sh stop       # stop + disable the boot service
#   sudo ./brain-web.sh serve-https  # OPTIONAL: also expose the clean
#                                    #   https://<node>.<tailnet>.ts.net/ URL via
#                                    #   `tailscale serve` (needs tailnet HTTPS certs)
#
# Idempotent; safe to re-run.
set -uo pipefail

REMOTE="/mnt/nvme/brain"
UNIT="brain-web.service"
PORT="${BRAIN_WEB_PORT:-8899}"
TS="$(command -v tailscale || true)"
HERE="$(cd "$(dirname "$0")" && pwd)"

r=$'\033[0m'; b=$'\033[1m'; gr=$'\033[32m'; ye=$'\033[33m'; rd=$'\033[31m'; dm=$'\033[2m'
say(){ printf '%b\n' "$*"; }

_need_ts(){ [ -n "$TS" ] || { say "${rd}ERROR${r} tailscale is not installed on this Jetson."; exit 1; }; }
_tsip(){ "$TS" ip -4 2>/dev/null | head -n1; }
# Standalone pipeline (not wrapped in an assignment) so the double quotes in the
# python one-liner stay inside single quotes and do not confuse the parser.
_https_host(){ "$TS" status --json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null; }
_url(){ local ip; ip="$(_tsip)"; [ -n "$ip" ] && printf 'http://%s:%s/' "$ip" "$PORT"; }
_health(){ local ip; ip="$(_tsip)"; [ -n "$ip" ] && curl -fsS "http://${ip}:${PORT}/health" 2>/dev/null; }

setup(){
  _need_ts
  say "${b}==> Installing ${UNIT} (boot service, binds the Tailscale IP) ...${r}"
  sudo cp "${HERE}/${UNIT}" "/etc/systemd/system/${UNIT}"
  sudo systemctl daemon-reload
  sudo systemctl enable --now "${UNIT}"

  say "${b}==> Waiting for the server on the Tailscale IP ...${r}"
  local up=0 i
  for i in $(seq 1 40); do
    if _health >/dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  local u; u="$(_url)"
  if [ "$up" -eq 1 ]; then
    say "  ${gr}OK${r} server is answering on ${u}"
  else
    say "  ${ye}WARN${r} server not up yet. Check: ${dm}journalctl -u ${UNIT} -n 40 --no-pager${r}"
  fi
  say ""
  say "${b}${gr}==> Brain web UI is LIVE (tailnet-only, survives reboots).${r}"
  say "    On your iPhone (Tailscale ON), open:  ${b}${u}${r}"
  say "    ${dm}Type the http:// prefix so Safari navigates instead of searching.${r}"
  say "    ${dm}Tip: Share -> Add to Home Screen for an app-like icon.${r}"
}

status(){
  _need_ts
  say "${b}service${r} : $(systemctl is-active "${UNIT}" 2>/dev/null || echo inactive)  ($(systemctl is-enabled "${UNIT}" 2>/dev/null || echo disabled))"
  if _health >/dev/null 2>&1; then say "${b}health${r}  : ${gr}up${r} $(_health)"; else say "${b}health${r}  : ${rd}down${r} on the Tailscale IP:${PORT}"; fi
  say "${b}url${r}     : $(_url)"
}

url(){ _need_ts; _url; echo; }

restart(){ sudo systemctl restart "${UNIT}" && say "restarted ${UNIT}"; }

stop(){
  say "==> Stopping ${UNIT} ..."
  sudo systemctl disable --now "${UNIT}" 2>/dev/null || true
  say "==> Stopped."
}

# OPTIONAL: also front the server with the clean HTTPS tailnet URL via Tailscale
# Serve. Requires HTTPS certificates enabled for the tailnet at login.tailscale.com.
# Note: `tailscale serve` on 443 collides with a public advisor Funnel (also 443).
serve_https(){
  _need_ts
  say "${b}==> Enabling Tailscale Serve  https://<node>/ (443) -> 127.0.0.1:${PORT} ...${r}"
  if ! sudo "$TS" serve --bg "${PORT}"; then
    cat >&2 <<EOF
  ERROR: could not enable Tailscale Serve. Most common cause: HTTPS certificates
  are not enabled for this tailnet yet. Turn them on at login.tailscale.com
  (DNS -> Enable HTTPS Certificates), then re-run.
EOF
    exit 1
  fi
  local host; host="$(_https_host)"
  say "  ${gr}OK${r} also reachable at:  ${b}https://${host}/${r}"
}

case "${1:-}" in
  setup|start)  setup ;;
  status)       status ;;
  url)          url ;;
  restart)      restart ;;
  stop)         stop ;;
  serve-https)  serve_https ;;
  *) say "usage: sudo $0 {setup|status|url|restart|stop|serve-https}"; exit 2 ;;
esac
