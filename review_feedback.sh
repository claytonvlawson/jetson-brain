#!/usr/bin/env bash
# review_feedback.sh -- review feedback_research.py drafts on the Mac and push
# approvals to the live parent advisor.
#
#   ./review_feedback.sh            # interactive review, then mirror + restart
#   ./review_feedback.sh --list     # just list the queue
#   ./review_feedback.sh --no-push  # review, but leave the push for later
#
# Approved notes land in ThesisVault/LanguageDeprivationAgent/Feedback Research
# (a normal folder: the cloud retriever and mirror_vault.sh both skip every
# underscore folder, so _Research would never reach the droplet).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BRAIN_ROOT="${BRAIN_ROOT:-$HOME/.advisor-feedback}"
export BRAIN_VAULT="${BRAIN_VAULT:-$HOME/Desktop/Dream/ThesisVault}"
export RESEARCH_NOTES_DIR="${RESEARCH_NOTES_DIR:-$BRAIN_VAULT/LanguageDeprivationAgent/Feedback Research}"
export RESEARCH_PUSH_CMD="${RESEARCH_PUSH_CMD:-ADVISOR_VM=advisor-vm $HOME/vault-advisor-cloud/mirror_vault.sh}"

PUSH="--push"
ARGS=()
for a in "$@"; do
  case "$a" in
    --no-push) PUSH="" ;;
    *) ARGS+=("$a") ;;
  esac
done

exec "$HERE/.venv/bin/python" "$HERE/research_review.py" ${PUSH} "${ARGS[@]+"${ARGS[@]}"}"
