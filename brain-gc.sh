#!/usr/bin/env bash
# brain-gc.sh -- report, and optionally reclaim, disk the brain setup is not using.
# Run on the Jetson.
#
#   ./brain-gc.sh          # DRY RUN: report every candidate and the total. Changes nothing.
#   ./brain-gc.sh --yes    # actually remove them
#
# Written after an audit that found the NVMe 16% full (35G of 234G) with 187G free,
# so this is hygiene, not a rescue. Nearly all of the 35G is ollama blobs: the entire
# brain -- code, vault, index, cache, eval results -- is 85 MB. There are no orphaned
# blobs to garbage-collect either, so the only real reclaim is whole models nothing loads.
#
# THE ONE HAZARD, stated plainly: models are chosen by a KEEP list, not a remove list.
# Anything you pull that is not in KEEP_MODELS below WILL be listed for removal, including
# a model you pulled this morning to experiment with. That is the price of a keep-list --
# it cannot silently miss a stale model, but it also cannot tell new work from junk. This
# is why the dry run is the default and --yes is mandatory. Read the list before you agree
# to it, and add anything you are actively using to KEEP_MODELS.
#
# Nothing here is irreversible in kind: every model re-pulls with `ollama pull <name>`.
# It costs bandwidth and time, not data.

set -uo pipefail

# ---- What the brain actually needs ---------------------------------------
# llama3.2:3b       DEFAULT_MODEL   (brain.py) -- the fast tier
# qwen2.5:7b        DEEP_MODEL      (brain.py) -- /deep
# nomic-embed-text  RERANK_EMBED_MODEL (brain.py) -- embeddings, hybrid retrieval, router
# qwen2.5:0.5b      NOT the brain's: ~/ncdd-rag/ncdd_rag.py depends on it. Keep it.
KEEP_MODELS="${KEEP_MODELS:-llama3.2:3b qwen2.5:7b nomic-embed-text qwen2.5:0.5b}"

BRAIN="${BRAIN:-/mnt/nvme/brain}"

APPLY=0
for a in "$@"; do
  case "$a" in
    --yes|-y) APPLY=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown flag: $a (use --yes to apply, no flag to dry-run)" >&2; exit 2 ;;
  esac
done

if [[ $APPLY -eq 1 ]]; then
  echo "==> brain-gc: APPLYING. Removing the items below."
else
  echo "==> brain-gc: DRY RUN. Nothing will be removed. Re-run with --yes to act."
fi
echo

echo "==> Disk before"
df -h /mnt/nvme / 2>/dev/null | sed 's/^/    /'
echo

# ---- Models --------------------------------------------------------------
# `ollama list` sizes are human strings ("7.6 GB"); normalize to MB for a total.
echo "==> Ollama models not in the keep list"
echo "    keeping: ${KEEP_MODELS}"
DOOMED=""
MODEL_MB=0
while read -r name id size unit rest; do
  [[ "$name" == "NAME" || -z "$name" ]] && continue
  keep=0
  for k in $KEEP_MODELS; do
    # match on the bare name too, so nomic-embed-text covers nomic-embed-text:latest
    [[ "$name" == "$k" || "${name%%:*}" == "${k%%:*}" ]] && keep=1
  done
  if [[ $keep -eq 1 ]]; then continue; fi
  case "$unit" in
    GB) mb=$(awk -v s="$size" 'BEGIN{printf "%d", s*1024}') ;;
    MB) mb=$(awk -v s="$size" 'BEGIN{printf "%d", s}') ;;
    *)  mb=0 ;;
  esac
  MODEL_MB=$((MODEL_MB + mb))
  DOOMED="${DOOMED} ${name}"
  printf "    %-28s %s %s\n" "$name" "$size" "$unit"
done < <(ollama list 2>/dev/null)

if [[ -z "${DOOMED// /}" ]]; then
  echo "    (none -- every installed model is in the keep list)"
else
  printf "    %-28s ~%d MB\n" "TOTAL" "$MODEL_MB"
  if [[ $APPLY -eq 1 ]]; then
    for m in $DOOMED; do
      echo "    removing ${m}"
      ollama rm "$m" >/dev/null 2>&1 || echo "    [could not remove ${m}]"
    done
  fi
fi
echo

# ---- Everything else, as (label, path) pairs -----------------------------
# Deliberately NOT here: index/bm25.V5.pkl. V5 is the shipped generation's rollback,
# and a side-by-side BRAIN_INDEX comparison is the safe way to A/B an index build.
# Deleting the fallback to save 8 MB on a disk with 187 GB free is a bad trade.
# label|path, split on the pipe so a path never has to be guessed out of a
# space-padded string.
CRUFT=(
  "pip wheel cache|${HOME}/.cache/pip"
  "stale index (two gens old)|${BRAIN}/index/bm25.V4.pkl"
  "stale index backup|${BRAIN}/index/bm25.pkl.bak"
)

echo "==> Caches and stale files"
for row in "${CRUFT[@]}"; do
  label="${row%%|*}"; path="${row#*|}"
  [[ -e "$path" ]] || continue
  sz=$(du -sh "$path" 2>/dev/null | cut -f1)
  printf "    %-28s %-8s %s\n" "$label" "$sz" "$path"
  [[ $APPLY -eq 1 ]] && rm -rf "$path"
done

# .bak files and __pycache__ are found rather than listed, since their names carry
# dates and commit refs that change.
while IFS= read -r f; do
  sz=$(du -sh "$f" 2>/dev/null | cut -f1)
  printf "    %-28s %-8s %s\n" "editing backup" "$sz" "$f"
  [[ $APPLY -eq 1 ]] && rm -f "$f"
done < <(find "$BRAIN" -maxdepth 1 -type f \( -name '*.bak' -o -name '*.bak-*' \
             -o -name '*.bak.*' -o -name '*.ARM0-*' \) 2>/dev/null)

while IFS= read -r d; do
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "    %-28s %-8s %s\n" "__pycache__" "$sz" "$d"
  [[ $APPLY -eq 1 ]] && rm -rf "$d"
done < <(find "$BRAIN" -maxdepth 2 -type d -name '__pycache__' 2>/dev/null)
echo

# ---- Root-filesystem steps that need sudo --------------------------------
# This Jetson does NOT have passwordless sudo, so these normally skip. `sudo -n`
# is what keeps the script from blocking on a password prompt in a non-interactive
# run, which would hang a deploy or a timer instead of failing cleanly.
echo "==> System caches (need sudo)"
if sudo -n true 2>/dev/null; then
  apt_sz=$(du -sh /var/cache/apt 2>/dev/null | cut -f1)
  jrl_sz=$(journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[MG]' | tail -1)
  printf "    %-28s %s\n" "apt cache" "${apt_sz:-?}"
  printf "    %-28s %s\n" "systemd journal" "${jrl_sz:-?}"
  if [[ $APPLY -eq 1 ]]; then
    sudo apt-get clean
    sudo journalctl --vacuum-size=64M >/dev/null 2>&1
  fi
else
  echo "    skipped: passwordless sudo is not configured on this device."
  echo "    To do these by hand:  sudo apt-get clean && sudo journalctl --vacuum-size=64M"
fi
echo

if [[ $APPLY -eq 1 ]]; then
  echo "==> Disk after"
  df -h /mnt/nvme / 2>/dev/null | sed 's/^/    /'
else
  echo "==> Dry run complete. Re-run with --yes to reclaim the above."
fi
