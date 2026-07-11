#!/usr/bin/env bash
# Run this on your Mac AFTER starting option 9 (ASL sign detection) on the Jetson.
set -euo pipefail

# Resolve the directory this script lives in (so asl_webcam.py is found next to it).
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Make sure python3 exists.
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found on this Mac. Install it (e.g. 'brew install python') and try again." >&2
  exit 1
fi

# 2. Make sure cv2 (OpenCV) and numpy import. Install them for the user if missing.
#    We do NOT install mediapipe on the Mac; the Jetson does the hand landmark work.
if ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
  echo "Setting up webcam dependencies (opencv-python, numpy) for your user..."
  echo "Running: pip install --user opencv-python numpy"
  python3 -m pip install --user opencv-python numpy
  # Re-check so we fail loudly if the install did not take.
  if ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "ERROR: cv2 or numpy still will not import after install. Check your pip / python3." >&2
    exit 1
  fi
  echo "Dependencies ready."
fi

# 3. Launch the webcam client. Extra flags pass straight through.
exec python3 "$HERE/asl_webcam.py" "$@"
