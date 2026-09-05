#!/usr/bin/env bash
#
# install.sh
#
# Installs the local "brain" RAG agent on the Jetson Orin Nano 8GB.
# Runs ON the Jetson as user snowbeach. No sudo is used unless you pass --net,
# which additionally installs the systemd unit that restores route + DNS at boot.
#
# Usage:
#   ./install.sh          Install deps, build index (if vault present), launcher.
#   ./install.sh --net     Same, plus install + enable jetson-online.service (sudo).
#
# Idempotent: safe to re-run.

set -euo pipefail

BRAIN_DIR="/mnt/nvme/brain"
VAULT_DIR="${BRAIN_DIR}/vault"
INDEX_DIR="${BRAIN_DIR}/index"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="${BIN_DIR}/brain"
SERVICE_SRC="${BRAIN_DIR}/jetson-online.service"
SERVICE_DST="/etc/systemd/system/jetson-online.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

INSTALL_NET=0
for arg in "$@"; do
  case "${arg}" in
    --net) INSTALL_NET=1 ;;
    *) echo "Unknown argument: ${arg}" ; echo "Usage: ./install.sh [--net]" ; exit 2 ;;
  esac
done

echo "==> Installing Python dependencies (pip --user, no sudo) ..."
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
  python3 -m pip install --user -r "${SCRIPT_DIR}/requirements.txt"
else
  echo "WARN: requirements.txt not found next to install.sh; installing named deps directly."
  python3 -m pip install --user ollama rank_bm25 ddgs beautifulsoup4 requests
fi

echo "==> Ensuring install directories exist (no sudo; snowbeach owns /mnt/nvme) ..."
mkdir -p "${VAULT_DIR}" "${INDEX_DIR}"

# --embed is REQUIRED here, not optional. brain.py's BRAIN_HYBRID defaults ON,
# but hybrid RRF silently degrades to BM25-only when the index carries no dense
# block -- which is exactly what a plain build produces. That is how this vault
# ran for a full rebuild cycle: hybrid retrieval was enabled and doing nothing.
# Embedding is graceful (build_index warns and continues if Ollama is down or
# nomic-embed-text is missing), so passing it unconditionally is safe.
# It does add a one-time embed pass over every chunk, so the build is slower.
echo "==> Building the BM25 index (with dense vectors for hybrid retrieval) ..."
if [ -n "$(ls -A "${VAULT_DIR}" 2>/dev/null || true)" ]; then
  echo "    (embedding every chunk with nomic-embed-text; this takes a few"
  echo "     minutes on the Orin and is not a hang)"
  python3 "${BRAIN_DIR}/build_index.py" --embed
  echo "Index built at ${INDEX_DIR}/bm25.pkl"
else
  echo "WARN: ${VAULT_DIR} is empty; the ThesisVault has not been synced yet."
  echo "      Skipping index build. rsync the vault, then run:"
  echo "        python3 ${BRAIN_DIR}/build_index.py --embed"
fi

echo "==> Installing launcher at ${LAUNCHER} ..."
mkdir -p "${BIN_DIR}"
cat > "${LAUNCHER}" <<'EOF'
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
EOF
chmod +x "${LAUNCHER}"

echo "==> Ensuring ${BIN_DIR} is on PATH ..."
if ! printf '%s' ":${PATH}:" | grep -q ":${BIN_DIR}:"; then
  if ! grep -qsF 'export PATH="$HOME/.local/bin:$PATH"' "${HOME}/.bashrc" 2>/dev/null; then
    printf '\n# Added by brain install.sh\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "${HOME}/.bashrc"
    echo "Appended PATH export to ~/.bashrc (open a new shell or: source ~/.bashrc)"
  else
    echo "PATH export already present in ~/.bashrc."
  fi
else
  echo "${BIN_DIR} already on PATH."
fi

if [ "${INSTALL_NET}" -eq 1 ]; then
  echo ""
  echo "==> --net requested: installing jetson-online.service."
  echo "    This is the ONLY step that uses sudo and WILL prompt for a password."
  if [ ! -f "${SERVICE_SRC}" ]; then
    echo "ERROR: ${SERVICE_SRC} not found. Place jetson-online.service in ${BRAIN_DIR} first."
    exit 1
  fi
  if [ ! -x "${BRAIN_DIR}/jetson-online.sh" ]; then
    echo "Making ${BRAIN_DIR}/jetson-online.sh executable ..."
    chmod +x "${BRAIN_DIR}/jetson-online.sh" 2>/dev/null || true
  fi
  sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
  sudo systemctl daemon-reload
  sudo systemctl enable jetson-online.service
  echo "jetson-online.service installed and enabled (runs at boot)."
fi

echo ""
echo "===================================================================="
echo "Done. Run: brain"
echo "  (If 'brain' is not found, open a new shell or: source ~/.bashrc)"
if [ "${INSTALL_NET}" -eq 1 ]; then
  echo "Net persistence: ENABLED via jetson-online.service (route + DNS at boot)."
  echo "  Start it now without rebooting: sudo systemctl start jetson-online.service"
else
  echo "Net persistence: NOT enabled. To enable route + DNS at boot, re-run:"
  echo "  ./install.sh --net"
  echo "Or bring the link up once now with: ${BRAIN_DIR}/jetson-online.sh"
fi
echo "===================================================================="
