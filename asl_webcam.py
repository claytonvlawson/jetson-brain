#!/usr/bin/env python3
"""
asl_webcam.py - Mac-side webcam client for the ASL -> Jetson pipeline.

The Mac captures webcam frames, JPEG-encodes them, and streams them over the
USB-C link to the Jetson, which runs the MediaPipe + classifier inference and
streams back one JSON reply per frame. This client imports only cv2 and numpy;
it does NOT import mediapipe (the Jetson does all inference).

PROTOCOL (Mac client <-> Jetson server over USB-C)
--------------------------------------------------
Roles:
  * Jetson is the SERVER at 192.168.55.1, listening on TCP port 8765 on 0.0.0.0.
  * Mac is the CLIENT, connects to 192.168.55.1:8765.

Framing is length-prefixed in both directions:
  * Client -> Server, per frame:
        4-byte big-endian UNSIGNED length N, then N bytes of JPEG-encoded BGR frame.
  * Server -> Client, per frame:
        4-byte big-endian UNSIGNED length M, then M bytes of UTF-8 JSON.

Server JSON reply shape (exact keys):
    {"sign": <string or null>, "confidence": <float 0..1>, "num_hands": <int>,
     "handedness": [<"Left"|"Right">...], "fps": <float>}

The client sends frames continuously; the server returns exactly one JSON reply
per frame. On Ctrl-C the client closes the socket cleanly and the server keeps
listening for the next client. No pickle is ever used: JSON + raw JPEG bytes only.
"""

import argparse
import json
import socket
import struct
import sys
import time

import cv2
import numpy as np


# 4-byte big-endian unsigned length prefix.
_LEN = struct.Struct(">I")


def recv_exactly(sock, num_bytes):
    """Read exactly num_bytes from sock, looping until satisfied.

    Returns a bytes object of length num_bytes, or None if the socket is
    closed before that many bytes arrive (a 0-length read means closed).
    """
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            # 0-length read: peer closed the connection.
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock, jpeg_bytes):
    """Send a length-prefixed JPEG frame to the server."""
    sock.sendall(_LEN.pack(len(jpeg_bytes)) + jpeg_bytes)


def recv_json(sock):
    """Receive one length-prefixed UTF-8 JSON reply. Returns dict or None."""
    header = recv_exactly(sock, _LEN.size)
    if header is None:
        return None
    (length,) = _LEN.unpack(header)
    payload = recv_exactly(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def resize_keep_aspect(frame, target_width):
    """Resize frame to target_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w == target_width or w == 0:
        return frame
    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_overlay(frame, reply):
    """Draw the recognized sign, confidence, fps, and num_hands on the frame."""
    sign = reply.get("sign")
    confidence = reply.get("confidence", 0.0) or 0.0
    num_hands = reply.get("num_hands", 0) or 0
    fps = reply.get("fps", 0.0) or 0.0

    # Bright green when we have a sign, muted grey when we do not.
    if sign:
        label = str(sign)
        color = (60, 255, 60)
    else:
        label = "..."
        color = (150, 150, 150)

    # Big, legible sign label near the top-left.
    cv2.putText(frame, label, (24, 72), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, label, (24, 72), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, color, 4, cv2.LINE_AA)

    # Secondary line: confidence, hands, fps.
    info = "conf {:.2f}   hands {}   fps {:.1f}".format(confidence, num_hands, fps)
    cv2.putText(frame, info, (24, 118), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, info, (24, 118), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Mac webcam client that streams frames to the Jetson ASL server.")
    parser.add_argument("--host", default="192.168.55.1",
                        help="Jetson server IP (default: 192.168.55.1)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Jetson server TCP port (default: 8765)")
    parser.add_argument("--camera", type=int, default=0,
                        help="cv2 VideoCapture index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Resize width, keeps aspect ratio (default: 640)")
    parser.add_argument("--jpeg-quality", type=int, default=70,
                        help="JPEG encode quality 0..100 (default: 70)")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless smoke test: no imshow, PASS after 15 round-trips")
    return parser.parse_args(argv)


def connect(host, port):
    """Connect to the Jetson server. Returns a socket or exits(1) on failure."""
    try:
        sock = socket.create_connection((host, port), timeout=10)
        sock.settimeout(None)
        return sock
    except OSError as exc:
        print("Could not connect to the Jetson ASL server at {}:{} ({}).".format(
            host, port, exc))
        print("On the Jetson, start option 9 (the ASL inference server) and try again.")
        sys.exit(1)


def main(argv=None):
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera index {}.".format(args.camera))
        sys.exit(1)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    sock = connect(args.host, args.port)

    window_title = "ASL -> Jetson (press q to quit)"
    roundtrips = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            frame = resize_keep_aspect(frame, args.width)

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                print("JPEG encode failed; skipping frame.")
                continue
            jpeg_bytes = buf.tobytes()

            try:
                send_frame(sock, jpeg_bytes)
                reply = recv_json(sock)
            except OSError as exc:
                print("Connection to the Jetson dropped ({}).".format(exc))
                print("On the Jetson, restart option 9 (the ASL inference server).")
                sys.exit(1)

            if reply is None:
                print("The Jetson closed the connection.")
                print("On the Jetson, restart option 9 (the ASL inference server).")
                sys.exit(1)

            roundtrips += 1

            if args.no_window:
                if roundtrips >= 15:
                    print("PASS asl_webcam roundtrip")
                    return 0
                continue

            draw_overlay(frame, reply)
            cv2.imshow(window_title, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        # Clean shutdown on Ctrl-C; the server keeps listening for the next client.
        print("\nStopping (Ctrl-C).")
    finally:
        try:
            sock.close()
        except OSError:
            pass
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
