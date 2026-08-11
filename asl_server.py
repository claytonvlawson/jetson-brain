#!/usr/bin/env python3
"""asl_server.py - Jetson-side ASL sign-detection TCP server.

Runs on the Jetson Orin Nano. A Mac client streams JPEG camera frames over the
USB-C network link; this server detects hands with MediaPipe, classifies the
sign with a small KNN model (asl_sign_model.SignClassifier), and streams back
one JSON reply per frame.

PROTOCOL
========
Transport is a raw TCP stream over the USB-C link.
  * Jetson is the SERVER at 192.168.55.1, listening on 0.0.0.0:8765.
  * Mac is the CLIENT, connecting to 192.168.55.1:8765.

Framing is length-prefixed in BOTH directions. Every message is a 4-byte
big-endian UNSIGNED length header followed by exactly that many payload bytes.

  Client -> Server (per frame):
    [4-byte big-endian uint32 N][N bytes of JPEG-encoded BGR frame]

  Server -> Client (per frame):
    [4-byte big-endian uint32 M][M bytes of UTF-8 JSON]

Server JSON reply shape (exact keys):
    {
      "sign":       <string or null>,
      "confidence": <float 0..1>,
      "num_hands":  <int>,
      "handedness": [<"Left"|"Right"> ...],
      "fps":        <float>
    }

The client sends frames continuously and receives exactly one JSON reply per
frame. On Ctrl-C the client closes the socket cleanly; the server logs the
disconnect and keeps listening for the next client. No pickle is used anywhere:
JSON plus raw JPEG bytes only.
"""

import argparse
import itertools
import json
import os
import socket
import struct
import sys
import time
import traceback
import urllib.request
from collections import deque, namedtuple

import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from asl_sign_model import SignClassifier


# ----------------------------------------------------------------------------
# Paths (Jetson runtime layout)
# ----------------------------------------------------------------------------
BRAIN_DIR = "/mnt/nvme/brain"
MODEL_PRIMARY = os.path.join(BRAIN_DIR, "models", "hand_landmarker.task")
MODEL_FALLBACK = "/tmp/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATASET_PATH = os.path.join(BRAIN_DIR, "asl_dataset.json")

# Length header: 4-byte big-endian unsigned int.
HEADER = struct.Struct(">I")

# Lightweight landmark container used for synthetic self-test landmarks. Real
# frames pass MediaPipe NormalizedLandmark objects straight through; both expose
# .x / .y / .z attributes.
LM = namedtuple("LM", ["x", "y", "z"])


# ----------------------------------------------------------------------------
# Model resolution
# ----------------------------------------------------------------------------
def resolve_model_path():
    """Return a usable hand_landmarker.task path, downloading it if needed."""
    for candidate in (MODEL_PRIMARY, MODEL_FALLBACK):
        if os.path.exists(candidate):
            return candidate

    dest = MODEL_PRIMARY
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)
    print("Hand landmark model not found locally. Downloading to %s ..." % dest)
    urllib.request.urlretrieve(MODEL_URL, dest)
    print("Download complete.")
    return dest


def build_landmarker(
    model_path,
    running_mode=vision.RunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=None,
    min_hand_presence_confidence=None,
):
    """Build a MediaPipe HandLandmarker.

    Defaults to IMAGE running mode, which is what the server uses in production:
    every frame gets a full palm detection, so no state crosses frames and a
    fingerspelling handshape change is never fitted inside the previous frame's
    region of interest. IMAGE mode uses the plain detect() call.

    RunningMode.VIDEO is available (opt-in via --video-mode) and is faster,
    because it reuses the previous frame's hand ROI and skips palm detection
    while the hand-presence score holds. It requires
    detect_for_video(image, timestamp_ms) with strictly increasing timestamps,
    and it carries per-hand tracking state, so a VIDEO landmarker must not be
    shared across client connections. It has NOT been A/B measured on real
    fingerspelling on this hardware; treat it as experimental.

    Threshold semantics differ by mode and are NOT interchangeable:
      * min_hand_detection_confidence gates palm detection in both modes.
      * min_tracking_confidence / min_hand_presence_confidence are inert in
        IMAGE mode (there is no tracking path to gate) and live in VIDEO mode,
        where they decide how long a lost hand keeps being tracked. They default
        to MediaPipe's 0.5 in IMAGE mode (where the value is irrelevant) and to
        a stricter 0.7 in VIDEO mode so tracking releases a lost hand quickly.
    """
    if running_mode == vision.RunningMode.VIDEO:
        tracking_default = 0.7
    else:
        tracking_default = 0.5
    if min_tracking_confidence is None:
        min_tracking_confidence = tracking_default
    if min_hand_presence_confidence is None:
        min_hand_presence_confidence = tracking_default

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


# ----------------------------------------------------------------------------
# Length-prefixed socket helpers
# ----------------------------------------------------------------------------
def recv_exactly(sock, n):
    """Read exactly n bytes from sock. Return None if the peer closed early."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            # 0-length read means the socket was closed.
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock):
    """Read one length-prefixed message. Return the payload bytes or None."""
    header = recv_exactly(sock, HEADER.size)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length == 0:
        return b""
    return recv_exactly(sock, length)


def send_message(sock, payload):
    """Send one length-prefixed message (bytes payload)."""
    sock.sendall(HEADER.pack(len(payload)) + payload)


def send_json(sock, obj):
    """Encode obj as compact UTF-8 JSON and send it length-prefixed."""
    payload = json.dumps(obj).encode("utf-8")
    send_message(sock, payload)


# ----------------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------------
def detect_hands(landmarker, bgr_frame, timestamp_ms=None):
    """Run the landmarker on a BGR frame.

    Pass timestamp_ms=None for an IMAGE-mode landmarker (the production default);
    the plain detect() call is used and each frame is independent.

    Pass an int timestamp_ms for a VIDEO-mode landmarker; detect_for_video() is
    used and the timestamp must be strictly increasing for the lifetime of that
    landmarker object, because MediaPipe raises if a timestamp repeats or goes
    backwards. The landmarker must belong to a single client connection: VIDEO
    mode carries tracking state and exposes no reset.

    Return (hand_landmarks_list, handedness_names, handedness_lists) where
    hand_landmarks_list is a list of per-hand landmark lists, handedness_names is
    a parallel list of "Left"/"Right" strings, and handedness_lists is the raw
    per-hand category list (kept for its scores, used by pick_best_hand).
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        result = landmarker.detect(mp_image)
    else:
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

    hand_landmarks_list = list(result.hand_landmarks or [])
    handedness_lists = list(result.handedness or [])
    handedness_names = []
    for cat_list in handedness_lists:
        if cat_list:
            handedness_names.append(cat_list[0].category_name)
        else:
            handedness_names.append("Right")
    return hand_landmarks_list, handedness_names, handedness_lists


def pick_best_hand(landmarker, result_handedness):
    """Return the index of the most-confident hand, or -1 if none."""
    best_idx = -1
    best_score = -1.0
    for idx, cat_list in enumerate(result_handedness):
        if cat_list:
            score = cat_list[0].score
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def extract_sign(result):
    """Normalize a SignClassifier.classify() result into (sign, confidence).

    Accepts a (sign, confidence) tuple, a {"sign", "confidence"} dict, or a bare
    label string, so the server is robust to the classifier's exact return type.
    """
    if isinstance(result, tuple) or isinstance(result, list):
        sign = result[0] if len(result) > 0 else None
        conf = result[1] if len(result) > 1 else 1.0
    elif isinstance(result, dict):
        sign = result.get("sign")
        conf = result.get("confidence", 1.0)
    else:
        sign = result
        conf = 1.0
    if sign is None:
        return None, 0.0
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return sign, conf


# ----------------------------------------------------------------------------
# Per-client frame loop
# ----------------------------------------------------------------------------
def handle_client(conn, addr, make_landmarker, classifier, collect_label, video_mode=False):
    """Serve one connected client until it disconnects.

    The landmarker is built here, per connection, and closed on disconnect, so
    nothing survives a client going away. That matters for VIDEO mode, which
    carries per-hand tracking ROIs and has no reset API: a shared landmarker
    would fit the first frames of a new connection inside the previous client's
    stale ROI, reporting hands (and, in --collect mode, recording samples) for a
    region that no longer holds one. The frame counter is created here for the
    same reason, so timestamps restart at 1 together with the tracking state.
    """
    print("Client connected from %s" % (addr,))
    frame_times = deque(maxlen=30)
    landmarker = None
    # VIDEO mode needs one strictly increasing millisecond per frame (wall clock
    # can repeat at high fps and can step backwards). IMAGE mode needs no clock.
    frame_counter = itertools.count(1) if video_mode else None

    try:
        landmarker = make_landmarker()
        while True:
            jpeg_bytes = recv_message(conn)
            if jpeg_bytes is None:
                break  # client closed the socket

            now = time.perf_counter()
            frame_times.append(now)
            if len(frame_times) >= 2:
                span = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / span if span > 0 else 0.0
            else:
                fps = 0.0

            # Decode the JPEG into a BGR frame.
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                send_json(
                    conn,
                    {
                        "sign": None,
                        "confidence": 0.0,
                        "num_hands": 0,
                        "handedness": [],
                        "fps": round(fps, 2),
                    },
                )
                continue

            timestamp_ms = next(frame_counter) if frame_counter is not None else None
            hand_landmarks_list, handedness_names, handedness_lists = detect_hands(
                landmarker, frame, timestamp_ms
            )
            num_hands = len(hand_landmarks_list)

            if num_hands == 0:
                send_json(
                    conn,
                    {
                        "sign": None,
                        "confidence": 0.0,
                        "num_hands": 0,
                        "handedness": [],
                        "fps": round(fps, 2),
                    },
                )
                continue

            best_idx = pick_best_hand(landmarker, handedness_lists)
            if best_idx < 0:
                best_idx = 0
            hand_landmarks = hand_landmarks_list[best_idx]
            if best_idx < len(handedness_names):
                hand_handedness = handedness_names[best_idx]
            else:
                hand_handedness = "Right"

            if collect_label is not None:
                # Data-collection mode: teach the classifier this sample.
                classifier.add_sample(collect_label, hand_landmarks)
                count = _sample_count(classifier, collect_label)
                reply_sign = "collecting:%s (%d)" % (collect_label, count)
                send_json(
                    conn,
                    {
                        "sign": reply_sign,
                        "confidence": 1.0,
                        "num_hands": num_hands,
                        "handedness": handedness_names,
                        "fps": round(fps, 2),
                    },
                )
                continue

            # Normal classification mode.
            classify_result = classifier.classify(hand_landmarks, hand_handedness)
            sign, confidence = extract_sign(classify_result)
            send_json(
                conn,
                {
                    "sign": sign,
                    "confidence": round(confidence, 4),
                    "num_hands": num_hands,
                    "handedness": handedness_names,
                    "fps": round(fps, 2),
                },
            )
    except (BrokenPipeError, ConnectionResetError) as exc:
        print("Client %s dropped: %s" % (addr, exc))
    except OSError as exc:
        print("Socket error with client %s: %s" % (addr, exc))
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as exc:  # noqa: BLE001 - close must never kill the server
                print("Warning: failed to close landmarker: %s" % exc)
        try:
            conn.close()
        except OSError:
            pass
        if collect_label is not None:
            # Persist collected samples when the collector disconnects.
            try:
                classifier.save()
                print("Saved collected samples for label '%s'." % collect_label)
            except Exception as exc:  # noqa: BLE001 - never crash the server on save
                print("Warning: failed to save dataset: %s" % exc)
        print("Client %s disconnected." % (addr,))


def _sample_count(classifier, label):
    """Best-effort count of stored samples for a label (for the reply string)."""
    getter = getattr(classifier, "sample_count", None)
    if callable(getter):
        try:
            return int(getter(label))
        except Exception:  # noqa: BLE001
            return 0
    samples = getattr(classifier, "samples", None)
    if isinstance(samples, dict) and label in samples:
        try:
            return len(samples[label])
        except TypeError:
            return 0
    return 0


# ----------------------------------------------------------------------------
# Server loop
# ----------------------------------------------------------------------------
def serve(host, port, make_landmarker, classifier, collect_label, video_mode=False):
    """Accept clients one at a time; each gets its own landmarker."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print_banner(host, port, collect_label, video_mode)

    try:
        while True:
            try:
                conn, addr = server_sock.accept()
            except KeyboardInterrupt:
                print("\nServer shutting down (Ctrl-C).")
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                handle_client(
                    conn, addr, make_landmarker, classifier, collect_label, video_mode
                )
            except Exception as exc:  # noqa: BLE001 - one bad client must not kill the server
                # Print the traceback: without it a detection path that fails on
                # every frame looks like a healthy server that just drops clients.
                print("Unhandled error serving %s: %s" % (addr, exc))
                traceback.print_exc()
    finally:
        server_sock.close()


def print_banner(host, port, collect_label, video_mode=False):
    mode = "DATA COLLECTION (label=%s)" % collect_label if collect_label else "detection"
    print("=" * 60)
    print(" ASL sign-detection server ready [%s mode]" % mode)
    print(" Landmarker running mode: %s" % ("VIDEO (experimental)" if video_mode else "IMAGE"))
    print(" Listening on %s:%d (Jetson USB-C link is 192.168.55.1)" % (host, port))
    print(" On the Mac, run:  ./asl-mac.sh")
    print("=" * 60)


# ----------------------------------------------------------------------------
# Self-test (no socket, no camera)
# ----------------------------------------------------------------------------
def synthetic_open_hand():
    """Build a synthetic open-hand set of 21 landmarks (objects with x/y/z)."""
    # Rough open-palm layout: wrist at bottom center, fingers splayed upward.
    pts = [
        (0.50, 0.90, 0.0),  # 0 wrist
        (0.40, 0.80, 0.0),  # 1 thumb cmc
        (0.34, 0.72, 0.0),  # 2 thumb mcp
        (0.30, 0.64, 0.0),  # 3 thumb ip
        (0.26, 0.58, 0.0),  # 4 thumb tip
        (0.44, 0.60, 0.0),  # 5 index mcp
        (0.43, 0.46, 0.0),  # 6 index pip
        (0.42, 0.36, 0.0),  # 7 index dip
        (0.41, 0.28, 0.0),  # 8 index tip
        (0.52, 0.58, 0.0),  # 9 middle mcp
        (0.52, 0.42, 0.0),  # 10 middle pip
        (0.52, 0.32, 0.0),  # 11 middle dip
        (0.52, 0.24, 0.0),  # 12 middle tip
        (0.60, 0.60, 0.0),  # 13 ring mcp
        (0.61, 0.46, 0.0),  # 14 ring pip
        (0.62, 0.36, 0.0),  # 15 ring dip
        (0.63, 0.28, 0.0),  # 16 ring tip
        (0.68, 0.64, 0.0),  # 17 pinky mcp
        (0.70, 0.52, 0.0),  # 18 pinky pip
        (0.72, 0.44, 0.0),  # 19 pinky dip
        (0.74, 0.38, 0.0),  # 20 pinky tip
    ]
    return [LM(x, y, z) for (x, y, z) in pts]


def _assert_detect_shape(hands, names, cats, where):
    """Assert detect_hands() returned three parallel per-hand lists."""
    assert len(hands) == len(names) == len(cats), (
        "%s: detect_hands returned mismatched lists (%d hands, %d names, %d cats)"
        % (where, len(hands), len(names), len(cats))
    )


def run_selftest():
    """Exercise both detection paths and the classifier: no socket, no camera.

    Both legs go through detect_hands(), the same function the serve loop calls,
    so a broken detect path fails here instead of failing silently once per
    client frame after deploy.
    """
    open_hand_label = "open_hand"
    landmarkers = []
    try:
        model_path = resolve_model_path()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1) Production path: IMAGE-mode landmarker, plain detect(), blank frame
        #    (0 hands is the expected and fine answer).
        image_lm = build_landmarker(model_path)
        landmarkers.append(image_lm)
        hands, names, cats = detect_hands(image_lm, blank)
        _assert_detect_shape(hands, names, cats, "IMAGE mode")

        # 2) Opt-in path (--video-mode): VIDEO-mode landmarker driven through
        #    detect_for_video with the same +1ms counter serve() feeds it. Two
        #    successive calls prove the VIDEO graph builds and accepts the
        #    counter values; the third proves a repeated timestamp is rejected,
        #    which is the contract handle_client's per-connection counter relies on.
        video_lm = build_landmarker(model_path, running_mode=vision.RunningMode.VIDEO)
        landmarkers.append(video_lm)
        for ts in (1, 2):
            hands, names, cats = detect_hands(video_lm, blank, ts)
            _assert_detect_shape(hands, names, cats, "VIDEO mode ts=%d" % ts)
        try:
            detect_hands(video_lm, blank, 2)
        except Exception:
            pass
        else:
            raise AssertionError(
                "VIDEO mode accepted a repeated timestamp; timestamps are "
                "expected to be strictly increasing"
            )

        # 3) Classifier round-trips a synthetic open-hand set to its label.
        classifier = SignClassifier(dataset_path=DATASET_PATH)
        lms = synthetic_open_hand()
        classifier.add_sample(open_hand_label, lms)
        sign, _conf = extract_sign(classifier.classify(lms, "Right"))
        assert sign == open_hand_label, (
            "expected classify() to return %r, got %r" % (open_hand_label, sign)
        )
    except Exception as exc:  # noqa: BLE001 - selftest reports failure via exit code
        print("FAIL asl_server selftest: %s" % exc)
        traceback.print_exc()
        return 1
    finally:
        for lm in landmarkers:
            try:
                lm.close()
            except Exception:  # noqa: BLE001
                pass

    print("PASS asl_server selftest")
    return 0


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Jetson-side ASL sign-detection TCP server."
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port to listen on.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind to.")
    parser.add_argument(
        "--collect",
        metavar="LABEL",
        default=None,
        help="Data-collection mode: store every detected hand under LABEL.",
    )
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help=(
            "EXPERIMENTAL: run the landmarker in VIDEO mode (tracked ROIs, "
            "faster) instead of the default per-frame IMAGE detection. Not "
            "measured on fingerspelling; can hold a stale handshape for several "
            "frames during fast transitions."
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run an offline self-test (no socket, no camera) and exit.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.selftest:
        return run_selftest()

    model_path = resolve_model_path()
    running_mode = (
        vision.RunningMode.VIDEO if args.video_mode else vision.RunningMode.IMAGE
    )

    def make_landmarker():
        return build_landmarker(model_path, running_mode=running_mode)

    # Prove the configured detection path actually runs BEFORE binding the port.
    # Otherwise a landmarker that cannot build or detect leaves a server that
    # listens, accepts, and fails on every client's first frame while looking
    # healthy to the ./asl launcher's port check.
    try:
        probe = make_landmarker()
        try:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            detect_hands(probe, blank, 1 if args.video_mode else None)
        finally:
            probe.close()
    except Exception as exc:  # noqa: BLE001 - fail loudly instead of serving broken
        print("FATAL: hand landmarker startup check failed: %s" % exc)
        traceback.print_exc()
        return 1

    classifier = SignClassifier(dataset_path=DATASET_PATH)

    serve(
        args.host, args.port, make_landmarker, classifier, args.collect, args.video_mode
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
