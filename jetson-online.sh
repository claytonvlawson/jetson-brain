#!/usr/bin/env bash
#
# jetson-online.sh
#
# Brings the Jetson Orin Nano online through the Mac over the USB-C link and
# makes sure DNS resolves. The Mac side is 192.168.55.100 (interface en10),
# the Jetson is 192.168.55.1, and the USB gadget interface on the Jetson is
# normally l4tbr0.
#
# Usage:
#   ./jetson-online.sh          Configure route + DNS (uses sudo), then test.
#   ./jetson-online.sh test     Only run the connectivity checks, change nothing.
#
# Idempotent: safe to run repeatedly. Route uses "ip route replace" so a second
# run is a no-op. DNS writes are only applied when needed.

set -euo pipefail

GATEWAY="192.168.55.100"
FALLBACK_IFACE="l4tbr0"
DNS1="8.8.8.8"
DNS2="1.1.1.1"
CURL_TARGET="https://duckduckgo.com"

# Detect the actual USB gadget interface (l4tbr* or usb*), fall back to l4tbr0.
detect_iface() {
  local found
  found="$(ip -o link show 2>/dev/null | awk -F': ' '{print $2}' \
            | sed 's/@.*//' \
            | grep -E '^(l4tbr|usb|rndis|enx)' \
            | head -n1 || true)"
  if [ -n "${found}" ]; then
    printf '%s' "${found}"
  else
    printf '%s' "${FALLBACK_IFACE}"
  fi
}

run_tests() {
  local ping_ok=1 curl_ok=1

  if ping -c1 -W2 "${DNS1}" >/dev/null 2>&1; then
    echo "PING ${DNS1}: PASS"
  else
    echo "PING ${DNS1}: FAIL"
    ping_ok=0
  fi

  if curl -fsS -m 10 -o /dev/null "${CURL_TARGET}" 2>/dev/null; then
    echo "CURL ${CURL_TARGET}: PASS"
  else
    echo "CURL ${CURL_TARGET}: FAIL"
    curl_ok=0
  fi

  if [ "${ping_ok}" -eq 1 ] && [ "${curl_ok}" -eq 1 ]; then
    return 0
  fi
  return 1
}

configure_route() {
  local iface="$1"
  echo "Ensuring default route via ${GATEWAY} dev ${iface} (sudo may prompt) ..."
  if ! ip link show "${iface}" >/dev/null 2>&1; then
    echo "WARN: interface ${iface} not present yet; trying anyway."
  fi
  sudo ip route replace default via "${GATEWAY}" dev "${iface}"
  echo "Route set: default via ${GATEWAY} dev ${iface}"
}

configure_dns() {
  local iface="$1"
  if systemctl is-active --quiet systemd-resolved 2>/dev/null; then
    echo "systemd-resolved is active; setting DNS via resolvectl (sudo may prompt) ..."
    sudo resolvectl dns "${iface}" "${DNS1}" "${DNS2}" || {
      echo "WARN: resolvectl failed; writing /etc/resolv.conf as a fallback."
      write_resolv_conf
      return
    }
    # Route DNS for the default domain through this link.
    sudo resolvectl domain "${iface}" "~." 2>/dev/null || true
    echo "DNS configured on ${iface}: ${DNS1} ${DNS2}"
  else
    echo "systemd-resolved not active; writing /etc/resolv.conf (sudo may prompt) ..."
    write_resolv_conf
  fi
}

write_resolv_conf() {
  local tmp
  tmp="$(mktemp)"
  {
    echo "# Written by jetson-online.sh"
    echo "nameserver ${DNS1}"
    echo "nameserver ${DNS2}"
  } > "${tmp}"
  if command -v cmp >/dev/null 2>&1 && cmp -s "${tmp}" /etc/resolv.conf 2>/dev/null; then
    echo "/etc/resolv.conf already up to date."
    rm -f "${tmp}"
    return
  fi
  sudo cp "${tmp}" /etc/resolv.conf
  rm -f "${tmp}"
  echo "/etc/resolv.conf set: nameserver ${DNS1}, ${DNS2}"
}

main() {
  local mode="${1:-configure}"

  if [ "${mode}" = "test" ]; then
    echo "Running connectivity checks only (no changes) ..."
    if run_tests; then
      echo "STATUS: ONLINE (all checks passed)"
      exit 0
    else
      echo "STATUS: OFFLINE (one or more checks failed)"
      exit 1
    fi
  fi

  local iface
  iface="$(detect_iface)"
  echo "Using USB link interface: ${iface}"

  configure_route "${iface}"
  configure_dns "${iface}"

  echo "Verifying connectivity ..."
  if run_tests; then
    echo "STATUS: ONLINE (route + DNS configured, all checks passed)"
    exit 0
  else
    echo "STATUS: DEGRADED (route + DNS configured, but checks failed; verify the Mac is sharing internet on en0 to en10)"
    exit 1
  fi
}

main "$@"
