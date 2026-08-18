#!/usr/bin/env bash
# webcache-pull.sh - pull the Jetson's web-fetched notes to the Mac for review.
# One-way (Jetson -> Mac), modeled on research-pull.sh but deliberately
# different in two ways.
#
# 1. DESTINATION IS NOT THE VAULT. Notes land in ~/Desktop/Dream/_WebCacheReview/,
#    a staging folder OUTSIDE ~/Desktop/Dream/ThesisVault. Reason: ~/.claude.json
#    runs mcp_server.py over ~/Desktop/Dream/ThesisVault as an MCP corpus root
#    serving the thesis-brain-vault tools, so pulling machine-fetched scrape into
#    the vault would inject it into a hand-authored scholarship corpus with no
#    known prune on that side. Staging keeps the notes reviewable in Obsidian
#    (just open the folder) while keeping them out of MCP search results.
#    The guard below refuses to run if the destination resolves inside ThesisVault.
#
# 2. NO --delete. research-pull.sh mirrors because its header declares the Jetson
#    the authority for _Research. This script does not: the whole point is to
#    curate these notes locally, and --delete with the Jetson as source would
#    destroy that curation on the next run.
#
# MAC-SIDE ONLY. This script must NEVER be shipped to the Jetson, where it would
# ssh to itself. Do not add it to deploy.sh's preflight list or its copy list.
# Same precedent as jetson-online-mac.sh and bashrc-menu-hook.sh, which are
# deliberately in neither list.
#
# Defaults to a DRY RUN. Pass --go to actually transfer.
set -euo pipefail

# ---- Config (override via environment) ----------------------------------
# JETSON default matches deploy.sh exactly, and honours the same override.
JETSON="${JETSON:-user@192.168.55.1}"
REMOTE="${REMOTE:-/mnt/nvme/brain}"
REVIEW_DST="${REVIEW_DST:-$HOME/Desktop/Dream/_WebCacheReview}"

# Two remote sources, kept in separate subdirectories so a reviewer can see at a
# glance what is pending approval versus what the Jetson already indexes.
# Paths mirror web_paths.py: WEB_QUEUE_DIR and WEB_NOTES_DIR.
QUEUE_SRC="${REMOTE}/queue/_WebCache"
NOTES_SRC="${REMOTE}/vault/_WebCache"
QUEUE_DST="${REVIEW_DST}/queue"
LIVE_DST="${REVIEW_DST}/live"

GO=0
for arg in "$@"; do
  case "${arg}" in
    --go) GO=1 ;;
    -h|--help)
      echo "Usage: $(basename "$0") [--go]"
      echo "  (no args)  dry run: show what would be pulled, transfer nothing"
      echo "  --go       perform the transfer"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

# ---- Guard: never write inside the MCP-served vault ----------------------
# Resolve symlinks and '..' before comparing, so a cute REVIEW_DST like
# "$HOME/Desktop/Dream/ThesisVault/../ThesisVault/_Web" cannot slip past.
VAULT_GUARD="${VAULT_GUARD:-$HOME/Desktop/Dream/ThesisVault}"
resolve_path() {
  # Walk up to the nearest existing ancestor, resolve that with pwd -P, then
  # re-append the tail. Works whether or not the destination exists yet.
  local p="$1" tail=""
  while [[ ! -d "${p}" && "${p}" != "/" && "${p}" != "." ]]; do
    tail="$(basename "${p}")${tail:+/}${tail}"
    p="$(dirname "${p}")"
  done
  local base
  base="$(cd "${p}" 2>/dev/null && pwd -P)" || base="${p}"
  if [[ -n "${tail}" ]]; then
    echo "${base%/}/${tail}"
  else
    echo "${base%/}"
  fi
}

REVIEW_ABS="$(resolve_path "${REVIEW_DST}")"
VAULT_ABS="$(resolve_path "${VAULT_GUARD}")"

if [[ "${REVIEW_ABS}" == "${VAULT_ABS}" || "${REVIEW_ABS}" == "${VAULT_ABS}"/* ]]; then
  cat >&2 <<EOF
ERROR: refusing to pull web scrape into the MCP-served vault.

  destination: ${REVIEW_ABS}
  vault root:  ${VAULT_ABS}

~/.claude.json runs mcp_server.py over that vault as a corpus root for the
thesis-brain-vault tools. Machine-fetched pages do not belong in a hand-authored
scholarship corpus. Point REVIEW_DST somewhere outside the vault and retry.
EOF
  exit 1
fi

# ---- Announce the resolved plan before touching anything ----------------
echo "==> webcache-pull"
echo "    host:  ${JETSON}"
echo "    queue: ${JETSON}:${QUEUE_SRC}/  ->  ${QUEUE_DST}/"
echo "    live:  ${JETSON}:${NOTES_SRC}/  ->  ${LIVE_DST}/"
if [[ "${GO}" -eq 1 ]]; then
  echo "    mode:  TRANSFER (--go)"
else
  echo "    mode:  DRY RUN (pass --go to transfer)"
fi

# ---- Preflight: reach the Jetson ----------------------------------------
# Distinguish "nothing fetched yet" from real failures (Jetson off, SSH auth,
# partial transfer): a verified-missing source folder is benign, anything else
# must surface loudly so pending notes are never silently absent from review.
if ! ssh -o ConnectTimeout=10 "${JETSON}" "test -d '${REMOTE}'"; then
  echo "ERROR: cannot reach ${JETSON}:${REMOTE} (Jetson off? SSH down?)" >&2
  exit 1
fi

pull_one() {
  local label="$1" src="$2" dst="$3"
  if ! ssh -o ConnectTimeout=10 "${JETSON}" "test -d '${src}'"; then
    echo "    (${label}: no ${src} on the Jetson yet; skipping)"
    return 0
  fi
  if [[ "${GO}" -eq 1 ]]; then
    mkdir -p "${dst}"
    # No --delete on purpose: see the header. Local edits and local-only files
    # survive every subsequent run.
    rsync -az "${JETSON}:${src}/" "${dst}/"
    local count
    count="$(find "${dst}" -type f -name '*.md' 2>/dev/null | wc -l | tr -d '[:space:]')"
    echo "    ${label}: ${count} note(s) now in ${dst}"
  else
    echo "    ${label}: would pull (dry run)"
    rsync -azn --itemize-changes "${JETSON}:${src}/" "${dst}/" || true
  fi
}

pull_one "queue" "${QUEUE_SRC}" "${QUEUE_DST}"
pull_one "live" "${NOTES_SRC}" "${LIVE_DST}"

if [[ "${GO}" -eq 1 ]]; then
  echo "    Done. Review in Obsidian by opening ${REVIEW_ABS} as a folder."
  echo "    Approve or reject on the Jetson with web_review.py; this copy is read-only in spirit."
else
  echo "    Dry run only. Nothing was written. Re-run with --go to transfer."
fi
