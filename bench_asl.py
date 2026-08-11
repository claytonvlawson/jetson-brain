#!/usr/bin/env python3
"""Measure ASL pipeline throughput. Reuses asl_webcam's protocol helpers.

Two modes:
  --source live          stream the webcam (default; not repeatable frame-for-frame)
  --source VIDEO.mp4     replay a recorded clip, so an IMAGE-mode run and a
                         VIDEO-mode run see byte-identical input. This is the
                         only fair way to compare, because MediaPipe VIDEO mode
                         only pays off when a hand is actually present to track.

Also: --record N writes N webcam frames to a clip and exits, so you can capture
a signing clip once and replay it for every future benchmark.
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, "/Users/you.lawson/jetson-brain")

import cv2  # noqa: E402
from asl_webcam import connect, recv_json, resize_keep_aspect, send_frame  # noqa: E402

WARMUP = 15


def record(path, frames, width):
    """Capture `frames` webcam frames to `path` so benchmarks are repeatable."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera failed")
        return 1
    ok, first = cap.read()
    if not ok:
        print("camera read failed")
        return 1
    first = resize_keep_aspect(first, width)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    written = 0
    print("Recording %d frames. Sign at the camera now..." % frames)
    while written < frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(resize_keep_aspect(frame, width))
        written += 1
    writer.release()
    cap.release()
    print("wrote %d frames to %s" % (written, path))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="live", help="'live' or a path to a video clip")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--label", default="", help="tag printed with the results")
    ap.add_argument("--record", metavar="PATH", help="record a clip and exit")
    ap.add_argument("--dump", metavar="PATH", help="write per-frame replies as JSON")
    args = ap.parse_args()

    if args.record:
        return record(args.record, args.frames, args.width)

    live_source = args.source == "live"
    cap = cv2.VideoCapture(0 if live_source else args.source)
    if not cap.isOpened():
        print("could not open source: %s" % args.source)
        return 1

    sock = connect(args.host, 8765)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

    server_fps, rtt_ms, per_frame = [], [], []
    hands_seen = sent = 0
    t_start = None

    try:
        for i in range(WARMUP + args.frames):
            ok, frame = cap.read()
            if not ok:
                if live_source:
                    break
                # Loop the clip so a short recording still fills the frame budget.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            frame = resize_keep_aspect(frame, args.width)
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            t0 = time.time()
            send_frame(sock, buf.tobytes())
            reply = recv_json(sock)
            t1 = time.time()
            if reply is None:
                print("server closed the connection")
                return 1
            if i < WARMUP:
                continue
            if t_start is None:
                t_start = time.time()
            sent += 1
            rtt_ms.append((t1 - t0) * 1000.0)
            server_fps.append(reply.get("fps") or 0.0)
            if (reply.get("num_hands") or 0) > 0:
                hands_seen += 1
            per_frame.append({
                "i": sent,
                "sign": reply.get("sign"),
                "confidence": reply.get("confidence"),
                "num_hands": reply.get("num_hands"),
                "handedness": reply.get("handedness"),
            })
        elapsed = time.time() - t_start if t_start else 0.0
    finally:
        sock.close()
        cap.release()

    if not sent:
        print("no frames measured")
        return 1

    tag = (" [%s]" % args.label) if args.label else ""
    print("=== bench%s  source=%s ===" % (tag, args.source))
    print("frames measured  : %d" % sent)
    print("round-trip rate  : %.2f fps (end-to-end)" % (sent / elapsed))
    print("per-frame latency: %.1f ms mean / %.1f ms median" % (
        statistics.mean(rtt_ms), statistics.median(rtt_ms)))
    print("server fps       : %.2f mean / %.2f median" % (
        statistics.mean(server_fps), statistics.median(server_fps)))
    print("frames with hands: %d/%d  (%.0f%%)" % (
        hands_seen, sent, 100.0 * hands_seen / sent))

    # Label behavior matters more than fps here: a tracked-ROI regression shows up
    # as extra label churn and as confident labels on frames that have no real hand.
    labels = [p["sign"] for p in per_frame]
    named = [s for s in labels if s]
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    print("frames with a sign: %d/%d" % (len(named), sent))
    print("label switches   : %d (lower = steadier)" % switches)
    counts = {}
    for s in named:
        counts[s] = counts.get(s, 0) + 1
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print("label histogram  : %s" % ", ".join("%s=%d" % kv for kv in top[:8]))

    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump({"label": args.label, "source": args.source,
                       "frames": per_frame}, fh, indent=1)
        print("per-frame dump   : %s" % args.dump)
    if hands_seen == 0:
        print("WARNING: no hands detected - this run does NOT exercise VIDEO-mode tracking,")
        print("         so it cannot show whether the change helps. Record a signing clip:")
        print("           bench_asl.py --record hands.mp4 --frames 150")
    return 0


if __name__ == "__main__":
    sys.exit(main())
