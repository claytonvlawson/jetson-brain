#!/usr/bin/env bash
# Run this on your Mac AFTER starting option 9 (ASL sign detection) on the Jetson.
set -euo pipefail

# Resolve the directory this script lives in (so asl_webcam.py is found next to it).
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Prefer a local venv if one is already provisioned with the deps. Homebrew's python3 is
#    PEP 668 externally-managed, so 'pip install --user' is refused there; the venv sidesteps it.
VENV_PY="$HERE/.venv-asl/bin/python"
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import cv2, numpy" >/dev/null 2>&1; then
  exec "$VENV_PY" "$HERE/asl_webcam.py" "$@"
fi

# 2. Otherwise fall back to the plain python3 path. Make sure python3 exists.
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found on this Mac. Install it (e.g. 'brew install python') and try again." >&2
  exit 1
fi

# 3. Make sure cv2 (OpenCV) and numpy import. Install them for the user if missing.
#    We do NOT install mediapipe on the Mac; the Jetson does the hand landmark work.
if ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
  echo "Setting up webcam dependencies (opencv-python, numpy) for your user..."
  echo "Running: pip install --user opencv-python numpy"
  if ! python3 -m pip install --user opencv-python numpy; then
    # PEP 668 (externally-managed) or no wheel for this interpreter. Build a venv instead.
    echo "pip --user failed; creating a local venv at $HERE/.venv-asl instead..."
    python3 -m venv "$HERE/.venv-asl"
    "$VENV_PY" -m pip install --quiet --upgrade pip
    "$VENV_PY" -m pip install --quiet opencv-python numpy
  fi
  # Re-check so we fail loudly if neither install took.
  if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "Dependencies ready (venv)."
    exec "$VENV_PY" "$HERE/asl_webcam.py" "$@"
  fi
  if ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "ERROR: cv2 or numpy still will not import after install. Check your pip / python3." >&2
    exit 1
  fi
  echo "Dependencies ready."
fi

# 4. Launch the webcam client. Extra flags pass straight through.
exec python3 "$HERE/asl_webcam.py" "$@"
