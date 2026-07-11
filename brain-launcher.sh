#!/usr/bin/env bash
# brain launcher: run the local RAG agent inside a persistent tmux session
# named "brain" so a laptop and an iPhone attach to the SAME live chat.
# Detach with Ctrl-b then d; the session (and any running answer) keeps going.
# Falls back to a direct run if tmux is not installed yet.
BRAIN_PY="/mnt/nvme/brain/brain.py"
if command -v tmux >/dev/null 2>&1; then
  exec tmux new-session -A -s brain "python3 $BRAIN_PY $*"
else
  exec python3 "$BRAIN_PY" "$@"
fi
