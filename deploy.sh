#!/usr/bin/env bash
# deploy.sh - push the brain (RAG chat) to the Jetson Orin Nano from the Mac.
# Idempotent: safe to re-run. Assumes brain.py, build_index.py, install.sh,
# requirements.txt, jetson-online.sh, and jetson-online.service sit next to it.
set -euo pipefail

# ---- Config (override via environment) ----------------------------------
JETSON="${JETSON:-snowbeach@192.168.55.1}"
VAULT_SRC="${VAULT_SRC:-$HOME/Desktop/Dream/ThesisVault}"
REMOTE="${REMOTE:-/mnt/nvme/brain}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Deploying brain to ${JETSON}:${REMOTE}"

# ---- Preflight: local files ---------------------------------------------
for f in brain.py brain_ui.py answer_cache.py build_index.py install.sh requirements.txt jetson-online.sh jetson-online.service; do
  if [[ ! -f "${HERE}/${f}" ]]; then
    echo "ERROR: missing required file next to deploy.sh: ${f}" >&2
    exit 1
  fi
done

# ---- Preflight: vault source --------------------------------------------
if [[ ! -d "${VAULT_SRC}" ]]; then
  echo "ERROR: vault source not found: ${VAULT_SRC}" >&2
  echo "       Set VAULT_SRC to the ThesisVault markdown folder and retry." >&2
  exit 1
fi

# ---- Preflight: reach the Jetson ----------------------------------------
echo "==> Checking SSH connectivity to ${JETSON} ..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${JETSON}" 'echo ok' >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: cannot reach the Jetson at ${JETSON} over SSH.

Checklist:
  1. Is the USB-C cable connected between the Mac and the Jetson?
  2. Does the Mac show the link interface (en10, 192.168.55.100)?
       ifconfig en10 | grep 'inet '
  3. Is the Jetson powered on and booted?
  4. Have you accepted the host key / set up an SSH key for ${JETSON}?
       ssh ${JETSON}

Fix the USB-C link, then re-run deploy.sh.
EOF
  exit 1
fi
echo "    SSH ok."

# ---- Remote layout ------------------------------------------------------
echo "==> Creating remote layout under ${REMOTE} ..."
ssh "${JETSON}" "mkdir -p '${REMOTE}/vault' '${REMOTE}/index'"

# ---- Sync the vault (markdown only) -------------------------------------
echo "==> Syncing vault markdown (md only) to ${REMOTE}/vault/ ..."
rsync -az --delete --prune-empty-dirs \
  --include='*/' --include='*.md' --exclude='*' \
  "${VAULT_SRC}/" "${JETSON}:${REMOTE}/vault/"

MD_COUNT="$(ssh "${JETSON}" "find '${REMOTE}/vault' -type f -name '*.md' | wc -l | tr -d '[:space:]'")"
echo "    Vault synced: ${MD_COUNT} markdown files on the Jetson."

# A zero-md sync means VAULT_SRC held no markdown (wrong folder?) -- and with
# --delete it just emptied the remote vault. Fail loudly rather than print a
# green "DEPLOY OK" over a brain that has nothing to index.
if [[ "${MD_COUNT}" -eq 0 ]]; then
  echo "ERROR: 0 markdown files synced from ${VAULT_SRC}." >&2
  echo "       The vault source has no .md files; refusing to report success." >&2
  exit 1
fi

# ---- Ship the code ------------------------------------------------------
echo "==> Copying scripts to ${REMOTE}/ ..."
scp -q \
  "${HERE}/brain.py" \
  "${HERE}/brain_ui.py" \
  "${HERE}/answer_cache.py" \
  "${HERE}/build_index.py" \
  "${HERE}/install.sh" \
  "${HERE}/requirements.txt" \
  "${HERE}/jetson-online.sh" \
  "${HERE}/jetson-online.service" \
  "${JETSON}:${REMOTE}/"

# ---- Make scripts executable and install --------------------------------
echo "==> Marking scripts executable and running install.sh on the Jetson ..."
ssh "${JETSON}" "chmod +x '${REMOTE}/install.sh' '${REMOTE}/jetson-online.sh' && bash '${REMOTE}/install.sh'"

# ---- Done ---------------------------------------------------------------
cat <<EOF

==> Deploy complete.

Next steps:
  1. Open a shell on the Jetson and start chatting:
       ssh ${JETSON}
       brain
     In the chat use /model to switch Ollama model, /web to toggle web search,
     and /exit to quit.

  2. Internet sharing (set up separately, one time):
       - On the Mac:    ./jetson-online-mac.sh start
       - On the Jetson: bash ${REMOTE}/install.sh --net
     This makes the Mac share its WiFi (en0) over the USB-C link (en10) and
     installs the jetson-online.service so the Jetson restores its route on boot.

DEPLOY OK: ${MD_COUNT} md files, code + install pushed to ${JETSON}:${REMOTE}
EOF
