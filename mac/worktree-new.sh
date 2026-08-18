#!/usr/bin/env bash
# worktree-new.sh - give one Claude Code session its own checkout of this repo.
#
# WHY THIS EXISTS
# On 2026-08-18 three Claude Code sessions ran against this single working
# directory at the same time. They shared one .git, one index, and one HEAD, so:
#
#   * one session's `git commit` captured another session's STAGED files and
#     shipped 18 files under a subject describing 1 of them;
#   * a `git checkout` and a `git reset --mixed HEAD~1` on the shared HEAD moved
#     the other sessions' files underneath them, with no error shown to anyone;
#   * two of the embedding session's hunks are still stranded inside the web
#     queue commit (567bf72), so a test in one commit is proved by code in
#     another.
#
# `git add` writes to $GIT_DIR/index. In a LINKED worktree that index lives at
# .git/worktrees/<name>/index, a physically separate file, and HEAD is per
# worktree too. So a second session staging a file cannot have it swept into
# this session's commit, and a branch switch over there cannot rewrite files
# over here. git also refuses to check out one branch in two worktrees, which
# turns "two agents silently on one branch" into a loud error at creation time.
#
# This script is Mac-side only and is deliberately in NEITHER deploy.sh list,
# matching jetson-online-mac.sh, bashrc-menu-hook.sh and webcache-pull.sh. It
# has no business on the Jetson.
#
# USAGE
#   mac/worktree-new.sh <name> [base-branch]
#   mac/worktree-new.sh embed-experiment          # branches from main
#   mac/worktree-new.sh hotfix main
#
# Creates ../jetson-brain-<name> on a new branch <name>.
set -uo pipefail

PROG="$(basename "$0")"

usage() {
  cat <<'USAGE'
usage: mac/worktree-new.sh <name> [base-branch]

Creates a linked git worktree at ../jetson-brain-<name> on a new branch <name>,
so a second Claude Code session can work without sharing this session's git
index. Base branch defaults to main.

  list        mac/worktree-new.sh --list
  remove      git worktree remove ../jetson-brain-<name>
              (add --force if it still has uncommitted changes)
USAGE
}

case "${1:-}" in
  -h|--help|"") usage; [[ -z "${1:-}" ]] && exit 1 || exit 0 ;;
  --list)       git worktree list; exit $? ;;
esac

NAME="$1"
BASE="${2:-main}"

# Reject anything that would not make a clean directory or ref name.
if [[ ! "${NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "${PROG}: name must be [A-Za-z0-9._-] only, got: ${NAME}" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${HERE}" || exit 1

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "${PROG}: ${HERE} is not a git repo." >&2
  exit 1
fi

DEST="$(cd .. && pwd)/jetson-brain-${NAME}"

if [[ -e "${DEST}" ]]; then
  echo "${PROG}: ${DEST} already exists. Pick another name, or remove it with:" >&2
  echo "  git worktree remove ${DEST}" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/${NAME}"; then
  echo "${PROG}: branch '${NAME}' already exists." >&2
  echo "  Attach a worktree to it instead: git worktree add ${DEST} ${NAME}" >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${BASE}"; then
  echo "${PROG}: base branch '${BASE}' does not exist." >&2
  exit 1
fi

echo "==> Creating worktree ${DEST}"
echo "    branch ${NAME} from ${BASE}"
if ! git worktree add -b "${NAME}" "${DEST}" "${BASE}"; then
  echo "${PROG}: git worktree add failed." >&2
  exit 1
fi

# ---- Carry over local state git does not track ---------------------------
# A linked worktree starts with tracked files only. Everything in .gitignore
# stays behind, so anything the repo needs at RUNTIME has to come across
# explicitly. None of this is fatal if it is missing; deploy.sh regenerates
# obsidian_vectors.json itself and guards on its absence.

# advisor.env.local holds local advisor config and is gitignored (secrets).
# Copied, not symlinked: a worktree may want to diverge on it.
if [[ -f "${HERE}/advisor.env.local" ]]; then
  cp "${HERE}/advisor.env.local" "${DEST}/advisor.env.local" \
    && echo "    Copied advisor.env.local"
fi

# index-mac/ is a 6.6M generated Mac-side index. Symlinked rather than copied:
# it is a build artifact, and two copies drift.
if [[ -d "${HERE}/index-mac" ]]; then
  ln -s "${HERE}/index-mac" "${DEST}/index-mac" \
    && echo "    Symlinked index-mac -> ${HERE}/index-mac"
fi

# ---- Prove deploy.sh would still work from over there --------------------
# deploy.sh preflights ~75 sibling files and hard-exits on the first missing
# one, BEFORE the vault rsync and the service restarts. Every name on that list
# is tracked, so a fresh worktree should pass, but assert it rather than assume:
# a future untracked addition to that list would otherwise only surface as a
# broken deploy from a worktree.
PREFLIGHT_LINE="$(grep -m1 '^for f in brain.py' "${DEST}/deploy.sh" 2>/dev/null)"
if [[ -n "${PREFLIGHT_LINE}" ]]; then
  LIST="${PREFLIGHT_LINE#for f in }"
  LIST="${LIST%; do}"
  MISSING=0
  for f in ${LIST}; do
    [[ -f "${DEST}/${f}" ]] || { echo "    MISSING: ${f}" >&2; MISSING=$((MISSING + 1)); }
  done
  if [[ "${MISSING}" -eq 0 ]]; then
    echo "    deploy.sh preflight: OK, all required files present"
  else
    echo "    deploy.sh preflight: ${MISSING} file(s) missing, deploying from here WILL fail" >&2
  fi
else
  echo "    deploy.sh preflight: could not parse the file list, check by hand" >&2
fi

cat <<NEXT

Worktree ready.

  cd ${DEST}

Point the other Claude Code session at that directory. It now has its own index
and its own HEAD, so its \`git add\` and \`git commit\` cannot touch this checkout.

Still shared, because there is one object store and one set of refs:
  * commits, branches and tags are visible from both sides immediately
  * a \`git push\` from either side pushes the same remote
  * both sides deploy to the SAME Jetson, and deploy.sh has NO locking, so two
    concurrent deploys can rsync --delete over each other and restart services
    mid-scp. Coordinate deploys by hand.

Two habits that matter even with worktrees:
  * stage explicit paths, never \`git add -A\`
  * \`git diff --cached --stat\` before every commit, and read it

NEXT
