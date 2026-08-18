"""Persistent Q&A transcript for the brain REPL.

The REPL's in-memory `history` list is prompt context, not a record: it is
trimmed to HISTORY_TURNS exchanges and dies with the process. The readline file
(.brain_history) keeps only the lines you TYPED. So once the terminal scrollback
rolls over, or you quit, every answer the model gave is gone.

This module is the record. One JSON object per answered turn, appended to
`transcript.jsonl` in CACHE_DIR, read back by the REPL's /history command.

Two deliberate properties, both copied from brain_input.py:

- It never raises. A read-only cache dir, a full disk or a half-written line is
  a no-op, never a traceback in the middle of an answer.
- It is OFF until enable() is called, and only the REPL's main() calls it. That
  is what keeps brain_web.py (:8899) and advisor_server.py (:8000) out of the
  file even though all three share handle_question/_finish.
"""

import json
import os
import time

# Rotate at 5 MB, keeping one generation. A long answer is a few KB, so this is
# thousands of turns; the point is only that an unattended REPL cannot fill the
# NVMe.
MAX_BYTES = 5 * 1024 * 1024

_path = None
_model = ""
_mode = "fast"


def enable(path):
    """Arm transcript writing at `path`. Call once, from the REPL only.

    Returns True when the file is usable. Every failure leaves the module
    disabled rather than raising, so a broken cache dir costs you /history and
    nothing else.
    """
    global _path
    try:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Touch with owner-only permissions, matching .brain_history. Doing it
        # here rather than on first append means the mode is right even if the
        # first thing that happens is a read.
        if not os.path.exists(path):
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        _path = path
        return True
    except Exception:
        _path = None
        return False


def enabled():
    return _path is not None


def set_turn(model, mode="fast"):
    """Record which model and mode the NEXT append belongs to.

    _finish() is shared by every answer path and has no idea which /deep,
    /parent or /scholar flag the REPL set for this question, and threading two
    more parameters through a dozen call sites to tell it would be churn for
    metadata. The REPL knows; it says so here, once per question.
    """
    global _model, _mode
    _model = str(model or "")
    _mode = str(mode or "fast")


def _rotate_if_big():
    try:
        if os.path.getsize(_path) < MAX_BYTES:
            return
        os.replace(_path, _path + ".1")
    except Exception:
        pass


def append(question, answer, sources=None, model=None, mode=None):
    """Append one answered turn. Silent no-op when disabled."""
    if _path is None:
        return False
    src = []
    for s in (sources or []):
        try:
            src.append(str(s))
        except Exception:
            pass
    rec = {
        "ts": time.time(),
        "model": str(model) if model else _model,
        "mode": str(mode) if mode else _mode,
        "q": str(question or ""),
        "a": str(answer or ""),
        "sources": src,
    }
    try:
        _rotate_if_big()
        with open(_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def read(limit=None, search=None):
    """Return saved turns oldest-first.

    `limit` keeps only the last N (applied AFTER `search`, so `/history 5 foo`
    means the five most recent matches, not matches among the five most recent).
    A line that does not parse is skipped: a transcript truncated by a full disk
    should still show you everything before the truncation.
    """
    if _path is None:
        return []
    out = []
    needle = (search or "").strip().lower()
    try:
        with open(_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if needle and needle not in (
                        rec.get("q", "") + "\n" + rec.get("a", "")).lower():
                    continue
                out.append(rec)
    except Exception:
        return out
    if limit is not None and limit > 0 and len(out) > limit:
        out = out[len(out) - limit:]
    return out


def count():
    """Number of saved turns, cheap enough for a status line."""
    return len(read())


def clear():
    """Truncate the transcript. Returns how many turns were removed."""
    if _path is None:
        return 0
    n = count()
    try:
        with open(_path, "w", encoding="utf-8"):
            pass
    except Exception:
        return 0
    return n
