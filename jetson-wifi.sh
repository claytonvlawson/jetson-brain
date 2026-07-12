#!/usr/bin/env bash
#
# jetson-wifi.sh
#
# Connect the Jetson Orin Nano to a WiFi network via NetworkManager so it has an
# internet path that does NOT depend on the Mac / USB-C link. This is what keeps
# the Jetson on Tailscale (and reachable from your iPhone) when the Mac is asleep.
#
# Idempotent and safe to re-run. Reads are unprivileged; the connect/modify steps
# use sudo (you may be prompted for your password once).
#
# Usage:
#   ./jetson-wifi.sh                     Scan, then prompt for SSID + password.
#   ./jetson-wifi.sh "MySSID"            Prompt for the password only (recommended).
#   ./jetson-wifi.sh "MySSID" "MyPass"   Non-interactive (password lands in shell history).
#   ./jetson-wifi.sh --list              Scan and list visible networks, then exit.
#   ./jetson-wifi.sh --status            Show current WiFi + routes + Tailscale, then exit.
#   ./jetson-wifi.sh --forget "MySSID"   Delete a saved WiFi profile.
#
# No em dashes by design (matches the other jetson scripts).
#
set -euo pipefail

r=$'\033[0m'; b=$'\033[1m'; cy=$'\033[36m'; gr=$'\033[32m'; ye=$'\033[33m'; rd=$'\033[31m'; dm=$'\033[2m'
say(){ printf '%b\n' "$*"; }
ok(){  say "  ${gr}OK${r}   $*"; }
warn(){ say "  ${ye}WARN${r} $*"; }
die(){ say "  ${rd}ERROR${r} $*" >&2; exit 1; }

command -v nmcli >/dev/null 2>&1 || die "nmcli (NetworkManager) not found; this script needs it."

# --- Find the WiFi device (e.g. wlP1p1s0 on the Orin Nano) -------------------
wifi_dev(){ nmcli -t -f DEVICE,TYPE dev 2>/dev/null | awk -F: '$2=="wifi"{print $1; exit}'; }
WIFI_DEV="$(wifi_dev)"
[ -n "${WIFI_DEV}" ] || die "no WiFi device found (nmcli sees no 'wifi' type interface)."

# --- Bring the radio up (clear any rfkill soft-block, then NM radio on) -------
radio_up(){
  if command -v rfkill >/dev/null 2>&1; then sudo rfkill unblock wifi 2>/dev/null || true; fi
  sudo nmcli radio wifi on 2>/dev/null || true
  # nmcli marks a device 'unmanaged' sometimes; make sure NM manages the wifi dev.
  sudo nmcli dev set "${WIFI_DEV}" managed yes 2>/dev/null || true
}

show_status(){
  say "${b}${cy}Jetson WiFi status${r}   ${dm}(device ${WIFI_DEV})${r}"
  say "${dm}--- device ---${r}"
  nmcli -t -f DEVICE,TYPE,STATE,CONNECTION dev 2>/dev/null | grep -E "^${WIFI_DEV}:|:wifi:" | sed 's/^/  /'
  local ip gw
  ip="$(nmcli -g IP4.ADDRESS dev show "${WIFI_DEV}" 2>/dev/null | head -n1)"
  gw="$(nmcli -g IP4.GATEWAY dev show "${WIFI_DEV}" 2>/dev/null | head -n1)"
  say "  wifi IP: ${ip:-none}   gateway: ${gw:-none}"
  say "${dm}--- default routes (lowest metric wins) ---${r}"
  ip route show default 2>/dev/null | sed 's/^/  /'
  say "${dm}--- tailscale ---${r}"
  say "  ip: $(tailscale ip -4 2>/dev/null || echo n/a)   name: $(tailscale status --json 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo n/a)"
}

case "${1:-}" in
  --status) show_status; exit 0 ;;
  --list)   radio_up; sudo nmcli dev wifi rescan ifname "${WIFI_DEV}" 2>/dev/null || true; sleep 2
            say "${b}Visible networks${r} ${dm}(SSID / signal / security)${r}"
            nmcli -f SSID,SIGNAL,SECURITY dev wifi list ifname "${WIFI_DEV}" 2>/dev/null | sed 's/^/  /'; exit 0 ;;
  --forget) [ -n "${2:-}" ] || die "usage: $0 --forget \"SSID\""
            sudo nmcli connection delete "${2}" && ok "forgot '${2}'." ; exit 0 ;;
esac

# --- Gather SSID + password --------------------------------------------------
SSID="${1:-}"
PASS="${2:-}"

radio_up

if [ -z "${SSID}" ]; then
  say "${b}Scanning for WiFi...${r}"
  sudo nmcli dev wifi rescan ifname "${WIFI_DEV}" 2>/dev/null || true; sleep 2
  nmcli -f SSID,SIGNAL,SECURITY dev wifi list ifname "${WIFI_DEV}" 2>/dev/null | sed 's/^/  /'
  printf "\n  Enter the SSID to join: "; read -r SSID
  [ -n "${SSID}" ] || die "no SSID entered."
fi

if [ -z "${PASS}" ]; then
  printf "  Password for %s%s%s (leave blank for an open network): " "${b}" "${SSID}" "${r}"
  read -rs PASS; echo
fi

# --- Connect -----------------------------------------------------------------
say "\n${b}Connecting ${WIFI_DEV} to '${SSID}'...${r}"
if [ -n "${PASS}" ]; then
  sudo nmcli dev wifi connect "${SSID}" password "${PASS}" ifname "${WIFI_DEV}" \
    || die "could not connect to '${SSID}' (wrong password, out of range, or hidden SSID)."
else
  sudo nmcli dev wifi connect "${SSID}" ifname "${WIFI_DEV}" \
    || die "could not connect to open network '${SSID}'."
fi

# --- Make it stick: autoconnect on boot, WiFi-primary route + DNS -----------
# The profile NetworkManager created is normally named after the SSID.
PROFILE="$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | awk -F: -v d="${WIFI_DEV}" '$2==d{print $1; exit}')"
PROFILE="${PROFILE:-${SSID}}"
# Tune the profile so WiFi is the primary path AND owns DNS:
#   route-metric 100    -> WiFi default route outranks the USB fallback (metric 700)
#   dns 8.8.8.8/1.1.1.1  -> deterministic public resolvers, not just DHCP's
#   dns-search "~."     -> WiFi is the catch-all DNS routing link, so resolution
#                          follows the primary path and keeps working when the Mac
#                          (USB link) sleeps and its own "~." route disappears.
sudo nmcli connection modify "${PROFILE}" \
  connection.autoconnect yes \
  connection.autoconnect-priority 10 \
  ipv4.route-metric 100 \
  ipv4.dns "8.8.8.8,1.1.1.1" \
  ipv4.dns-search "~." 2>/dev/null || warn "could not tune profile '${PROFILE}' (connected anyway)."
# route-metric and DNS only take effect when the profile is (re)activated. Bounce
# it up now so they apply immediately; otherwise the live route keeps NM's old
# (higher) metric and old DNS until the next reboot.
sudo nmcli connection up "${PROFILE}" ifname "${WIFI_DEV}" >/dev/null 2>&1 || true
ok "profile '${PROFILE}' set to auto-connect on boot (WiFi primary: route metric 100, DNS catch-all)."

# --- Verify the WiFi path itself (independent of the USB/Mac route) ----------
say "\n${b}Verifying internet over WiFi (${WIFI_DEV})...${r}"
sleep 2
if ping -c2 -W3 -I "${WIFI_DEV}" 1.1.1.1 >/dev/null 2>&1; then
  ok "WiFi has its own internet path (ping 1.1.1.1 via ${WIFI_DEV})."
else
  warn "connected to '${SSID}' but could not reach the internet over ${WIFI_DEV} yet."
  warn "check the password / that this network allows internet, then re-run --status."
fi

echo
show_status

# --- Note on failover --------------------------------------------------------
cat <<EOF

${dm}Note: WiFi is now the *primary* internet path. Its default route has metric
100, while the USB-C link to the Mac is a metric-700 fallback (set by
jetson-online.sh). Whenever WiFi is up, traffic and Tailscale egress ride WiFi;
if WiFi drops or is out of range, the Jetson falls back to the Mac link
automatically. Re-run this script with --status to see the live default routes
(lowest metric wins).${r}
EOF
