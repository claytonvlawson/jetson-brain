#!/usr/bin/env bash
# research-pull.sh - pull approved research notes from the Jetson into the
# Mac's ThesisVault so Obsidian sees them. One-way (Jetson -> Mac): the
# Jetson's vault/_Research is the authority for agent-found notes, and
# deploy.sh excludes _Research/ so this copy never round-trips back.
# Idempotent; run any time after approving notes with research_review.py.
set -euo pipefail

JETSON="${JETSON:-user@192.168.55.1}"
REMOTE="${REMOTE:-/mnt/nvme/brain}"
VAULT_DST="${VAULT_DST:-$HOME/Desktop/Dream/ThesisVault}"

if [[ ! -d "${VAULT_DST}" ]]; then
  echo "ERROR: Mac vault not found at ${VAULT_DST}" >&2
  exit 1
fi

# Distinguish "nothing to pull yet" from real failures (Jetson off, SSH auth,
# partial transfer): only a verified-missing remote folder is benign; any
# other failure must surface loudly so approved notes are never silently
# missing from Obsidian.
if ! ssh -o ConnectTimeout=10 "${JETSON}" "test -d '${REMOTE}/vault/_Research'"; then
  if ssh -o ConnectTimeout=10 "${JETSON}" "test -d '${REMOTE}'"; then
    echo "    (no _Research folder on the Jetson yet; nothing to pull)"
    exit 0
  fi
  echo "ERROR: cannot reach ${JETSON}:${REMOTE} (Jetson off? SSH down?)" >&2
  exit 1
fi

echo "==> Pulling ${JETSON}:${REMOTE}/vault/_Research/ -> ${VAULT_DST}/_Research/"
rsync -az --delete \
  "${JETSON}:${REMOTE}/vault/_Research/" "${VAULT_DST}/_Research/"

COUNT="$(find "${VAULT_DST}/_Research" -type f -name '*.md' 2>/dev/null | wc -l | tr -d '[:space:]')"
echo "    Done: ${COUNT} research note(s) now in the Mac vault."
