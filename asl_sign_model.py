"""ASL sign classifier module.

Pure python + numpy. No mediapipe, no cv2 imported here so the module stays
lightweight and testable. The server is expected to run MediaPipe Hands and
pass the resulting 21 landmarks straight into this module.

A "hand" is a list of 21 landmark points. Each point may be either:
  - an object with attributes .x, .y, .z (e.g. a MediaPipe NormalizedLandmark)
  - a dict with keys "x", "y", "z"
Coordinates are MediaPipe normalized image coords in the 0..1 range.

Public API:
  normalize_landmarks(hand) -> np.ndarray of shape (21, 3)
  class SignClassifier:
      __init__(dataset_path=None)
      classify(hand, handedness="Right") -> (label|None, confidence float)
      add_sample(label, hand) -> None
      save() -> None
      num_samples() -> int

Run standalone to execute a self-test:
  python3 asl_sign_model.py
"""

import json
import os

import numpy as np

# Number of landmarks MediaPipe Hands returns per hand.
NUM_LANDMARKS = 21

# KNN configuration.
KNN_K = 3
KNN_MIN_CONFIDENCE = 0.5

# Confidence assigned to an exact geometry-rule match.
GEOMETRY_CONFIDENCE = 0.8

# Finger tip / pip landmark indices used by the geometry rules.
# Each entry: name -> (tip_index, pip_index).
FINGER_INDICES = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

# Thumb landmark indices.
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2

# Wrist landmark index (origin for normalization and distance rules).
WRIST = 0

# Mapping from a 5-tuple finger state (thumb, index, middle, ring, pinky)
# of 0/1 values to a human-readable sign label.
GEOMETRY_MAP = {
    (1, 1, 1, 1, 1): "5 / open hand",
    (0, 0, 0, 0, 0): "S / fist",
    (0, 1, 0, 0, 0): "1 / point",
    (0, 1, 1, 0, 0): "V / two",
    (1, 1, 0, 0, 0): "L",
    (0, 1, 1, 1, 0): "W / three",
    (0, 0, 0, 0, 1): "I / pinky",
    (1, 1, 0, 0, 1): "ILY (I love you)",
    (1, 0, 0, 0, 0): "thumbs up",
}


def _get_coord(point, axis):
    """Read a single coordinate from a landmark point.

    Supports both attribute-style access (point.x) and dict-style access
    (point["x"]). Returns a float.
    """
    # Dict-style access first (dicts do not carry an .x attribute).
    if isinstance(point, dict):
        return float(point[axis])
    # Attribute-style access (MediaPipe NormalizedLandmark, namedtuple, etc).
    return float(getattr(point, axis))


def _to_array(hand):
    """Convert a hand (list of 21 points) into an (N, 3) numpy float array.

    Accepts both attribute-style and dict-style points. Returns None when the
    hand is missing or malformed so callers can bail out gracefully.
    """
    if hand is None:
        return None
    try:
        rows = []
        for point in hand:
            rows.append(
                [
                    _get_coord(point, "x"),
                    _get_coord(point, "y"),
                    _get_coord(point, "z"),
                ]
            )
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def normalize_landmarks(hand):
    """Return a (21, 3) array normalized to be translation and scale invariant.

    Steps:
      1. Translate so the wrist (landmark 0) sits at the origin.
      2. Divide by the hand's overall size, defined as the maximum distance of
         any landmark from the wrist, so absolute hand size does not matter.

    Raises ValueError if the hand does not contain exactly 21 valid points.
    """
    coords = _to_array(hand)
    if coords is None or coords.shape[0] != NUM_LANDMARKS:
        raise ValueError("normalize_landmarks requires exactly 21 landmarks")

    # 1) Translate so the wrist is the origin.
    wrist = coords[WRIST]
    centered = coords - wrist

    # 2) Scale by the max distance from the wrist (overall hand size).
    distances = np.linalg.norm(centered, axis=1)
    scale = float(distances.max())
    if scale <= 1e-9:
        # Degenerate hand (all points coincide); avoid divide by zero.
        scale = 1.0
    return centered / scale


def _finger_extended(coords, tip_index, pip_index):
    """A finger is extended if its tip is farther from the wrist than its pip.

    Distances are Euclidean in the (already normalized-or-raw) coordinate space.
    """
    wrist = coords[WRIST]
    tip_dist = np.linalg.norm(coords[tip_index] - wrist)
    pip_dist = np.linalg.norm(coords[pip_index] - wrist)
    return 1 if tip_dist > pip_dist else 0


def _thumb_extended(coords):
    """Decide whether the thumb is extended.

    Kept deliberately simple and robust: the thumb points sideways rather than
    away from the wrist, so distance-from-wrist is unreliable. Instead compare
    how far the thumb tip sits from the hand center along the x-axis versus the
    thumb ip joint, scaled by the hand size so the test is scale invariant.
    """
    center = coords.mean(axis=0)
    # Hand size proxy: max distance from wrist, matching normalize_landmarks.
    wrist = coords[WRIST]
    scale = float(np.linalg.norm(coords - wrist, axis=1).max())
    if scale <= 1e-9:
        scale = 1.0

    tip_offset = abs(coords[THUMB_TIP][0] - center[0]) / scale
    ip_offset = abs(coords[THUMB_IP][0] - center[0]) / scale
    return 1 if tip_offset > ip_offset else 0


def _finger_state(coords):
    """Return the 5-tuple (thumb, index, middle, ring, pinky) of 0/1 values."""
    thumb = _thumb_extended(coords)
    index = _finger_extended(coords, *FINGER_INDICES["index"])
    middle = _finger_extended(coords, *FINGER_INDICES["middle"])
    ring = _finger_extended(coords, *FINGER_INDICES["ring"])
    pinky = _finger_extended(coords, *FINGER_INDICES["pinky"])
    return (thumb, index, middle, ring, pinky)


class SignClassifier:
    """KNN-with-geometry-fallback ASL sign classifier."""

    def __init__(self, dataset_path=None):
        """Load a KNN dataset json from dataset_path if it exists.

        The dataset format is:
          {"labels": [str, ...], "samples": [[63 floats], ...]}
        where each sample is a flattened normalized 21x3 hand.
        """
        self.dataset_path = dataset_path
        self.labels = []
        self.samples = []  # list of flat lists of 63 floats
        if dataset_path and os.path.exists(dataset_path):
            self._load(dataset_path)

    def _load(self, path):
        """Load labels and samples from a dataset json file."""
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            labels = data.get("labels", [])
            samples = data.get("samples", [])
            # Keep only well-formed 63-float samples paired with a label.
            for label, sample in zip(labels, samples):
                if len(sample) == NUM_LANDMARKS * 3:
                    self.labels.append(str(label))
                    self.samples.append([float(v) for v in sample])
        except (OSError, ValueError, KeyError, TypeError):
            # Corrupt or unreadable dataset: start empty rather than crash.
            self.labels = []
            self.samples = []

    def num_samples(self):
        """Return the number of stored training samples."""
        return len(self.samples)

    def add_sample(self, label, hand):
        """Normalize a hand and append it to the in-memory dataset."""
        normalized = normalize_landmarks(hand)  # raises on bad input
        self.labels.append(str(label))
        self.samples.append(normalized.flatten().tolist())

    def save(self):
        """Persist the in-memory dataset to dataset_path as json."""
        if not self.dataset_path:
            raise ValueError("SignClassifier.save requires a dataset_path")
        directory = os.path.dirname(self.dataset_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.dataset_path, "w") as fh:
            json.dump({"labels": self.labels, "samples": self.samples}, fh)

    def _knn_classify(self, normalized):
        """KNN over normalized landmarks. Returns (label, confidence) or None.

        Uses k=3 majority vote with confidence = votes_for_winner / k.
        Returns None (so the caller can fall back to geometry) when there is no
        training data or the winning confidence is below the threshold.
        """
        if not self.samples:
            return None

        query = normalized.flatten()
        data = np.asarray(self.samples, dtype=np.float64)
        # Euclidean distance from the query to every stored sample.
        dists = np.linalg.norm(data - query, axis=1)

        k = min(KNN_K, len(self.samples))
        nearest = np.argsort(dists)[:k]

        # Majority vote among the k nearest neighbours.
        votes = {}
        for idx in nearest:
            lbl = self.labels[idx]
            votes[lbl] = votes.get(lbl, 0) + 1
        winner = max(votes, key=votes.get)

        # Confidence is normalized against the actual number of neighbours
        # examined (k, which is min(KNN_K, len(samples))), matching the contract
        # that confidence = votes_for_winner / k. Dividing by the fixed KNN_K
        # would make small datasets (<3 samples) unable to ever clear the
        # confidence threshold, disabling KNN until 3+ samples are collected.
        confidence = votes[winner] / k
        if confidence >= KNN_MIN_CONFIDENCE:
            return winner, confidence
        return None

    def _geometry_classify(self, coords):
        """Built-in geometry rules. Returns (label, confidence) or (None, 0.0).

        Works with no training data at all.
        """
        state = _finger_state(coords)
        label = GEOMETRY_MAP.get(state)
        if label is not None:
            return label, GEOMETRY_CONFIDENCE
        return None, 0.0

    def classify(self, hand, handedness="Right"):
        """Classify a hand into an ASL sign.

        1) If a trained KNN dataset is loaded and produces a confident vote,
           return that (label, confidence).
        2) Otherwise fall back to the built-in geometry rules.
        Returns (None, 0.0) if nothing matches or the hand is malformed.

        handedness is accepted for API compatibility; the current geometry
        rules are handedness agnostic thanks to the symmetric thumb test.
        """
        coords = _to_array(hand)
        if coords is None or coords.shape[0] != NUM_LANDMARKS:
            # Robust to hands with fewer than 21 points.
            return None, 0.0

        normalized = normalize_landmarks(hand)

        # 1) Try KNN first when we have training data.
        knn_result = self._knn_classify(normalized)
        if knn_result is not None:
            return knn_result

        # 2) Fall back to geometry rules (also the no-training-data path).
        return self._geometry_classify(coords)


def _synthetic_open_hand():
    """Build a synthetic open-hand ("5") landmark set for the self-test.

    The layout is a rough splayed hand: wrist at the bottom, five fingers
    fanning upward (decreasing y toward the tips), with the thumb offset to the
    side. Values are plausible MediaPipe-style normalized coords. Each finger's
    tip is placed farther from the wrist than its pip so all fingers register
    as extended, and the thumb tip sits farther out in x than the thumb ip.
    """
    # Points are (x, y, z). y decreases upward toward fingertips.
    pts = [
        (0.50, 0.90, 0.0),  # 0 wrist
        # Thumb: splayed out to the left, tip farthest in x from center.
        (0.42, 0.82, 0.0),  # 1 thumb cmc
        (0.34, 0.76, 0.0),  # 2 thumb mcp
        (0.27, 0.70, 0.0),  # 3 thumb ip
        (0.18, 0.66, 0.0),  # 4 thumb tip
        # Index finger.
        (0.46, 0.62, 0.0),  # 5 index mcp
        (0.45, 0.48, 0.0),  # 6 index pip
        (0.44, 0.36, 0.0),  # 7 index dip
        (0.44, 0.24, 0.0),  # 8 index tip
        # Middle finger.
        (0.52, 0.60, 0.0),  # 9 middle mcp
        (0.52, 0.44, 0.0),  # 10 middle pip
        (0.52, 0.30, 0.0),  # 11 middle dip
        (0.52, 0.18, 0.0),  # 12 middle tip
        # Ring finger.
        (0.58, 0.62, 0.0),  # 13 ring mcp
        (0.59, 0.48, 0.0),  # 14 ring pip
        (0.60, 0.34, 0.0),  # 15 ring dip
        (0.60, 0.22, 0.0),  # 16 ring tip
        # Pinky finger.
        (0.64, 0.66, 0.0),  # 17 pinky mcp
        (0.66, 0.54, 0.0),  # 18 pinky pip
        (0.68, 0.44, 0.0),  # 19 pinky dip
        (0.70, 0.34, 0.0),  # 20 pinky tip
    ]
    # Return as dict-style points to also exercise dict access.
    return [{"x": x, "y": y, "z": z} for (x, y, z) in pts]


if __name__ == "__main__":
    # Self-test: a synthetic open hand must classify as "5 / open hand".
    clf = SignClassifier()  # no dataset -> geometry rules only
    hand = _synthetic_open_hand()

    label, conf = clf.classify(hand, handedness="Right")
    assert label == "5 / open hand", (
        "expected '5 / open hand', got %r (conf=%s)" % (label, conf)
    )

    # A hand with fewer than 21 points must return (None, 0.0).
    short_label, short_conf = clf.classify(hand[:10])
    assert (short_label, short_conf) == (None, 0.0), "short-hand guard failed"

    print("PASS asl_sign_model")
