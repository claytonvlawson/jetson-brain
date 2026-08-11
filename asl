#!/usr/bin/env bash
# asl - one command, run from the Mac, that does the whole ASL pipeline:
#   1. starts the ASL inference server on the Jetson over SSH (if it is not already up),
#   2. waits for port 8765 to accept connections,
#   3. launches the Mac webcam client.
#
# Usage:
#   ./asl                 start everything and open the camera window
#   ./asl --no-window     headless smoke test (15 round-trips, prints PASS)
#   ./asl stop            stop the Jetson server
#   ./asl log             tail the Jetson server log
# Any other flags (--camera 1, --width 480, ...) pass through to asl_webcam.py.
set -euo pipefail

# Resolve symlinks by hand so this still finds asl-mac.sh when invoked through a
# symlink in PATH. BSD readlink has no -f, hence the loop.
SRC="$0"
while [ -L "$SRC" ]; do
  SRC_DIR="$(cd -P "$(dirname "$SRC")" && pwd)"
  SRC="$(readlink "$SRC")"
  case "$SRC" in
    /*) ;;
    *) SRC="$SRC_DIR/$SRC" ;;
  esac
done
HERE="$(cd -P "$(dirname "$SRC")" && pwd)"

JETSON_SSH="${JETSON_SSH:-user@192.168.55.1}"
JETSON_IP="${JETSON_IP:-192.168.55.1}"
JETSON_DIR="${JETSON_DIR:-/mnt/nvme/brain}"
PORT="${ASL_PORT:-8765}"
LOG="/tmp/asl_server.log"

SSH_OPTS=(-o ConnectTimeout=8 -o BatchMode=yes)

port_open() { nc -z -G 2 "$JETSON_IP" "$PORT" >/dev/null 2>&1; }

case "${1:-}" in
  stop)
    # The [a] bracket keeps the pattern from matching this very ssh command line,
    # which would otherwise make the remote shell kill itself before pkill runs.
    ssh "${SSH_OPTS[@]}" "$JETSON_SSH" "pkill -f '[a]sl_server\.py'" \
      && echo "Jetson ASL server stopped." \
      || echo "No ASL server was running."
    exit 0
    ;;
  log)
    exec ssh "${SSH_OPTS[@]}" "$JETSON_SSH" "tail -f $LOG"
    ;;
esac

# 1. Is the Jetson reachable at all? Fail with a useful message, not a timeout.
if ! nc -z -G 3 "$JETSON_IP" 22 >/dev/null 2>&1; then
  echo "ERROR: cannot reach the Jetson at $JETSON_IP:22." >&2
  echo "Plug the USB-C cable into the Mac (that link is what carries 192.168.55.x)." >&2
  exit 1
fi

# 2. Start the server only if nothing is listening yet. The ( ... & ) subshell plus the
#    redirected fds is what lets ssh close the channel and return; a plain trailing '&'
#    leaves the remote login shell alive and ssh hangs on it indefinitely.
if port_open; then
  echo "Jetson ASL server is already up on $JETSON_IP:$PORT."
else
  echo "Starting the ASL server on the Jetson..."
  ssh -n "${SSH_OPTS[@]}" "$JETSON_SSH" \
    "cd '$JETSON_DIR' && ( setsid python3 asl_server.py </dev/null >'$LOG' 2>&1 & ) ; exit 0"

  # MediaPipe init takes a few seconds; give it a generous window.
  for _ in $(seq 1 60); do
    port_open && break
    sleep 1
  done

  if ! port_open; then
    echo "ERROR: the server did not start listening on $PORT. Last log lines:" >&2
    ssh "${SSH_OPTS[@]}" "$JETSON_SSH" "tail -25 '$LOG'" >&2 || true
    exit 1
  fi
  echo "Server is listening on $JETSON_IP:$PORT."
fi

# 3. Hand off to the Mac client (which picks the .venv-asl interpreter).
exec "$HERE/asl-mac.sh" "$@"
