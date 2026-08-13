#!/usr/bin/env bash
# deploy.sh - push the brain (RAG chat) to the Jetson Orin Nano from the Mac.
# Idempotent: safe to re-run. Assumes brain.py, build_index.py, install.sh,
# requirements.txt, jetson-online.sh, and jetson-online.service sit next to it.
set -euo pipefail

# ---- Config (override via environment) ----------------------------------
JETSON="${JETSON:-user@192.168.55.1}"
VAULT_SRC="${VAULT_SRC:-$HOME/Desktop/Dream/ThesisVault}"
REMOTE="${REMOTE:-/mnt/nvme/brain}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Deploying brain to ${JETSON}:${REMOTE}"

# ---- Preflight: local files ---------------------------------------------
for f in brain.py brain_ui.py answer_cache.py web_cache.py build_index.py update_index.py refresh_web.py prime_cache.py eval_brain.py eval_retrieval.py gold_questions.jsonl heldout_gold.jsonl parent_gold.jsonl meta_gold.jsonl factual_gold.jsonl multihop_gold.jsonl retrieval_gold.jsonl install.sh requirements.txt jetson-online.sh jetson-wifi.sh jetson-online.service asl_server.py asl_sign_model.py asl_webcam.py asl-mac.sh jetson-menu.sh advisor_server.py advisor-funnel-jetson.sh advisor.service advisor.env.example brain_web.py brain-web.sh brain-web.service brain-learn.service brain-learn.timer semantic_cache.py research_sources.py research_fetch.py research_note.py research_agent.py research_review.py research-agent.service research-agent.timer obsidian_rank.py export_obsidian_vectors.py brain_skills.py brain_ollama_shim.py brain-ollama-shim.service escalate.py test_hybrid.py test_is_insufficient.py test_intent.py test_note_card.py test_stream_holdback.py test_obsidian_rank.py test_obsidian_expand.py test_web_semantic.py test_obsidian_blocks.py test_brain_skills.py test_ollama_shim.py test_hyde.py test_ctx_fit.py test_embed_prefix.py; do
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
# _WebCache and _Research are Jetson-authoritative machine-managed folders
# (web fetches and approved research-agent notes); excluding them keeps
# --delete from destroying them and keeps the Mac copy (research-pull.sh)
# from round-tripping back.
rsync -az --delete --prune-empty-dirs \
  --exclude='_WebCache/' \
  --exclude='_Research/' \
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

# ---- Generate the Obsidian Smart Connections vector export ---------------
# Optional signal for BRAIN_OBSIDIAN_RANK (default off -- see brain.py and
# obsidian_rank.py). Non-fatal: a vault with no .smart-env/ (plugin not
# installed, or a different vault) just means this feature has nothing to
# draw on; brain.py already degrades gracefully when the export is missing
# (obsidian_rank.load_note_vectors_from_export returns {}).
echo "==> Generating Obsidian Smart Connections vector export ..."
BRAIN_OBSIDIAN_SMART_ENV_DIR="${VAULT_SRC}/.smart-env/multi" \
  python3 "${HERE}/export_obsidian_vectors.py" "${HERE}/obsidian_vectors.json" \
  || echo "    (skipped: no Obsidian vector data found at ${VAULT_SRC}/.smart-env/multi)"

# ---- Ship the code ------------------------------------------------------
echo "==> Copying scripts to ${REMOTE}/ ..."
scp -q \
  "${HERE}/brain.py" \
  "${HERE}/brain_ui.py" \
  "${HERE}/answer_cache.py" \
  "${HERE}/web_cache.py" \
  "${HERE}/build_index.py" \
  "${HERE}/prime_cache.py" \
  "${HERE}/eval_brain.py" \
  "${HERE}/eval_retrieval.py" \
  "${HERE}/gold_questions.jsonl" \
  "${HERE}/heldout_gold.jsonl" \
  "${HERE}/parent_gold.jsonl" \
  "${HERE}/meta_gold.jsonl" \
  "${HERE}/factual_gold.jsonl" \
  "${HERE}/multihop_gold.jsonl" \
  "${HERE}/retrieval_gold.jsonl" \
  "${HERE}/install.sh" \
  "${HERE}/requirements.txt" \
  "${HERE}/jetson-online.sh" \
  "${HERE}/jetson-wifi.sh" \
  "${HERE}/jetson-online.service" \
  "${HERE}/asl_server.py" \
  "${HERE}/asl_sign_model.py" \
  "${HERE}/asl_webcam.py" \
  "${HERE}/asl-mac.sh" \
  "${HERE}/jetson-menu.sh" \
  "${HERE}/advisor_server.py" \
  "${HERE}/advisor-funnel-jetson.sh" \
  "${HERE}/advisor.service" \
  "${HERE}/advisor.env.example" \
  "${HERE}/brain_web.py" \
  "${HERE}/brain-web.sh" \
  "${HERE}/brain-web.service" \
  "${HERE}/update_index.py" \
  "${HERE}/refresh_web.py" \
  "${HERE}/brain-learn.service" \
  "${HERE}/brain-learn.timer" \
  "${HERE}/semantic_cache.py" \
  "${HERE}/research_sources.py" \
  "${HERE}/research_fetch.py" \
  "${HERE}/research_note.py" \
  "${HERE}/research_agent.py" \
  "${HERE}/research_review.py" \
  "${HERE}/research-agent.service" \
  "${HERE}/research-agent.timer" \
  "${HERE}/obsidian_rank.py" \
  "${HERE}/export_obsidian_vectors.py" \
  "${HERE}/brain_skills.py" \
  "${HERE}/brain_ollama_shim.py" \
  "${HERE}/brain-ollama-shim.service" \
  "${HERE}/escalate.py" \
  "${HERE}/test_hybrid.py" \
  "${HERE}/test_is_insufficient.py" \
  "${HERE}/test_intent.py" \
  "${HERE}/test_note_card.py" \
  "${HERE}/test_stream_holdback.py" \
  "${HERE}/test_obsidian_rank.py" \
  "${HERE}/test_obsidian_expand.py" \
  "${HERE}/test_web_semantic.py" \
  "${HERE}/test_obsidian_blocks.py" \
  "${HERE}/test_brain_skills.py" \
  "${HERE}/test_ollama_shim.py" \
  "${HERE}/test_hyde.py" \
  "${HERE}/test_ctx_fit.py" \
  "${HERE}/test_embed_prefix.py" \
  "${JETSON}:${REMOTE}/"

# obsidian_vectors.json is a generated artifact (see the step above), not
# source code -- shipped separately so a vault with no Smart Connections data
# (missing/empty file, handled non-fatally above) doesn't fail the main copy.
if [[ -f "${HERE}/obsidian_vectors.json" ]]; then
  scp -q "${HERE}/obsidian_vectors.json" "${JETSON}:${REMOTE}/"
  echo "    Shipped obsidian_vectors.json ($(du -h "${HERE}/obsidian_vectors.json" | cut -f1 | tr -d '[:space:]'))."
fi

# ---- Make scripts executable and install --------------------------------
echo "==> Marking scripts executable and running install.sh on the Jetson ..."
ssh "${JETSON}" "chmod +x '${REMOTE}/install.sh' '${REMOTE}/jetson-online.sh' '${REMOTE}/jetson-wifi.sh' '${REMOTE}/asl-mac.sh' '${REMOTE}/jetson-menu.sh' '${REMOTE}/advisor-funnel-jetson.sh' '${REMOTE}/brain-web.sh' '${REMOTE}/brain_web.py' && bash '${REMOTE}/install.sh'"

# ---- Restart the long-running services ----------------------------------
# Deploy overwrites the .py files in place, but a service already running holds
# the OLD code in memory until it is restarted. advisor_server.py and
# brain_web.py both import brain.py, so without this they keep serving the
# pre-deploy retrieval pipeline indefinitely and the deploy looks like a no-op.
# brain-ollama-shim.service is in the same boat (it imports brain.py to answer
# Obsidian Smart Chat), and like the other two it is only restarted when it is
# already installed and running, so a Jetson that never enabled it is untouched.
# systemctl needs root here, so this is best-effort: if sudo is not available
# non-interactively, print the commands rather than failing the deploy.
echo "==> Restarting services that hold brain.py in memory ..."
ssh "${JETSON}" 'if sudo -n true 2>/dev/null; then
  for u in advisor.service brain-web.service brain-ollama-shim.service; do
    if systemctl list-unit-files "$u" >/dev/null 2>&1 && systemctl is-active --quiet "$u"; then
      sudo systemctl restart "$u" && echo "    restarted $u"
    fi
  done
else
  echo "    SKIPPED: passwordless sudo unavailable on the Jetson."
  echo "    Services are still running PRE-DEPLOY code. Run manually:"
  echo "      sudo systemctl restart advisor.service brain-web.service brain-ollama-shim.service"
fi'

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

  3. WiFi (Mac-independent internet; keeps Tailscale up when the Mac sleeps):
       ssh ${JETSON}
       bash ${REMOTE}/jetson-wifi.sh "YourSSID"
     WiFi becomes the primary path (route metric 100); the USB-C link stays a
     metric-700 fallback. Check anytime with: bash ${REMOTE}/jetson-wifi.sh --status

  4. Phone web UI (open the brain in Safari from your iPhone over Tailscale):
       ssh ${JETSON}
       sudo bash ${REMOTE}/brain-web.sh setup
     Installs a boot service (brain_web.py on 127.0.0.1:8899) and exposes it
     tailnet-only over HTTPS via 'tailscale serve'. Prints the https URL to open.
     Status/URL: bash ${REMOTE}/brain-web.sh status

  5. Obsidian Smart Chat against the brain (Ollama-API shim, one time):
       ssh ${JETSON}
       sudo cp ${REMOTE}/brain-ollama-shim.service /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now brain-ollama-shim.service
     Serves a fake Ollama API on the tailnet IP, port 11435 (never 11434, which
     the real Ollama owns). In Obsidian: Settings -> Smart Chat -> model platform
     "Ollama (Local)" -> Ollama host = http://<tailscale-ip>:11435 -> model
     "jetson-brain". See README.md for the full click path.
     Status: systemctl status brain-ollama-shim.service

DEPLOY OK: ${MD_COUNT} md files, code + install pushed to ${JETSON}:${REMOTE}
EOF
