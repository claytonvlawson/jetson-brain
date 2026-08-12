"""Automatic local -> Mac Studio escalation for jetson-brain.

Standalone by design: no import from brain.py, so it can't be broken by
concurrent edits there and can be wired in later behind a flag. Always
tries the local Ollama model first; the complexity heuristic only shortens
how long that attempt is allowed to run before falling back to the bigger
model on the Mac Studio. Never raises -- callers get an EscalateResult with
an "[error] ..." text on total failure, matching brain.py's chat_once
convention (callers can check .text for that prefix same as elsewhere).

CLI:
    python3 escalate.py "some question"
    python3 escalate.py --score "some question"   (heuristic only, no I/O)
    python3 escalate.py --ping                     (check the Mac Studio)
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# ---- Config ---------------------------------------------------------------

LOCAL_HOST = os.environ.get("ESCALATE_LOCAL_HOST", "http://127.0.0.1:11434")
LOCAL_MODEL = os.environ.get("ESCALATE_LOCAL_MODEL", "llama3.2:3b")
LOCAL_TIMEOUT = float(os.environ.get("ESCALATE_LOCAL_TIMEOUT", "12"))
LOCAL_TIMEOUT_HARD = float(os.environ.get("ESCALATE_LOCAL_TIMEOUT_HARD", "5"))
LOCAL_KEEP_ALIVE = os.environ.get("ESCALATE_LOCAL_KEEP_ALIVE", "-1")

# Jetson is 192.168.55.1 from the Mac's side of the USB-C link (see
# deploy.sh's JETSON=user@192.168.55.1); the Mac itself sits at
# 192.168.55.100 on that same link. Override to a Tailscale ts.net name
# if the two aren't cabled together.
REMOTE_HOST = os.environ.get("ESCALATE_REMOTE_HOST", "http://192.168.55.100:11434")
REMOTE_MODEL = os.environ.get("ESCALATE_REMOTE_MODEL", "llama3.3:70b")
REMOTE_TIMEOUT = float(os.environ.get("ESCALATE_REMOTE_TIMEOUT", "120"))
REMOTE_TOKEN = os.environ.get("ESCALATE_REMOTE_TOKEN", "")
REMOTE_KEEP_ALIVE = os.environ.get("ESCALATE_REMOTE_KEEP_ALIVE", "10m")

# requests' single `timeout=` covers connect+read together, so an unreachable
# host (Mac asleep/unplugged) would otherwise eat the full REMOTE_TIMEOUT
# budget just to fail. Split it: fail fast on connect, stay generous on read.
CONNECT_TIMEOUT = float(os.environ.get("ESCALATE_CONNECT_TIMEOUT", "5"))


def _flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "false", "False", "no")


ESCALATE_ENABLED = _flag("ESCALATE_ENABLED", "1")
ESCALATE_VERBOSE = _flag("ESCALATE_VERBOSE", "0")

LOCAL_GEN_OPTIONS = {
    "num_predict": 640,
    "num_ctx": 6144,
    "temperature": 0.2,
    "repeat_penalty": 1.1,
    "num_gpu": 999,
}
REMOTE_GEN_OPTIONS = {
    "num_predict": 1024,
    "num_ctx": 8192,
    "temperature": 0.2,
    "repeat_penalty": 1.1,
}

# ---- Complexity heuristic ---------------------------------------------------

REASONING_WORDS = (
    "why", "explain", "analyze", "analyse", "compare", "contrast",
    "trade-off", "tradeoff", "pros and cons", "evaluate", "justify",
    "step by step", "step-by-step", "prove", "derive",
)
CODE_WORDS = (
    "code", "function", "script", "debug", "traceback", "stack trace",
    "regex", "algorithm", "refactor", "class ", "def ", "implement",
    "compile", "syntax error", "exception",
)
MATH_WORDS = (
    "calculate", "compute", "equation", "integral", "derivative",
    "probability", "matrix", "solve for", "sum of", "average of",
    "percentage",
)

LONG_CHARS = int(os.environ.get("ESCALATE_LONG_CHARS", "320"))
LONG_WORDS = int(os.environ.get("ESCALATE_LONG_WORDS", "55"))
MULTI_QMARKS = int(os.environ.get("ESCALATE_MULTI_QMARKS", "2"))
MANY_SENTENCES = int(os.environ.get("ESCALATE_MANY_SENTENCES", "3"))
HARD_SCORE = int(os.environ.get("ESCALATE_HARD_SCORE", "3"))


def score_complexity(question):
    """Cheap, tunable difficulty signal. Structural signals (length/shape)
    are weak evidence on their own (1 pt); content triggers (reasoning/code/
    math vocabulary) are strong evidence the 3B model will struggle (2 pts)."""
    q = question.strip()
    ql = q.lower()
    signals = {
        "long_chars": len(q) >= LONG_CHARS,
        "long_words": len(q.split()) >= LONG_WORDS,
        "multi_part": q.count("?") >= MULTI_QMARKS,
        "many_sentences": len(re.findall(r"[.!?]+", q)) >= MANY_SENTENCES,
        "reasoning_words": any(w in ql for w in REASONING_WORDS),
        "code_words": any(w in ql for w in CODE_WORDS),
        "math_words": any(w in ql for w in MATH_WORDS),
    }
    score = (
        signals["long_chars"]
        + signals["long_words"]
        + signals["multi_part"]
        + signals["many_sentences"]
        + 2 * signals["reasoning_words"]
        + 2 * signals["code_words"]
        + 2 * signals["math_words"]
    )
    return score, signals


# ---- Result type -------------------------------------------------------------

@dataclass
class EscalateResult:
    text: str
    tier: str  # "local" | "remote" | "none"
    escalated: bool
    reason: str
    latency_s: float
    score: int = 0
    signals: dict = field(default_factory=dict)
    local_error: Optional[str] = None
    remote_error: Optional[str] = None


def _log(msg):
    if ESCALATE_VERBOSE:
        print("[escalate] " + msg, flush=True)


# ---- Model calls ---------------------------------------------------------

def _chat(host, model, messages, options, keep_alive, timeout, token=None):
    headers = {"Authorization": "Bearer " + token} if token else {}
    resp = requests.post(
        host + "/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        },
        headers=headers,
        timeout=(min(CONNECT_TIMEOUT, timeout), timeout),
    )
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content", "") or ""


def call_local(messages, timeout=None):
    t0 = time.time()
    timeout = timeout if timeout is not None else LOCAL_TIMEOUT
    try:
        text = _chat(LOCAL_HOST, LOCAL_MODEL, messages, LOCAL_GEN_OPTIONS,
                     LOCAL_KEEP_ALIVE, timeout)
        return EscalateResult(text=text, tier="local", escalated=False,
                               reason="local_ok", latency_s=time.time() - t0)
    except requests.exceptions.Timeout as exc:
        return EscalateResult(text="", tier="local", escalated=False,
                               reason="local_timeout", latency_s=time.time() - t0,
                               local_error=str(exc))
    except Exception as exc:
        return EscalateResult(text="", tier="local", escalated=False,
                               reason="local_error", latency_s=time.time() - t0,
                               local_error=str(exc))


def call_remote(messages, timeout=None):
    t0 = time.time()
    timeout = timeout if timeout is not None else REMOTE_TIMEOUT
    try:
        text = _chat(REMOTE_HOST, REMOTE_MODEL, messages, REMOTE_GEN_OPTIONS,
                     REMOTE_KEEP_ALIVE, timeout, token=REMOTE_TOKEN)
        return EscalateResult(text=text, tier="remote", escalated=True,
                               reason="remote_ok", latency_s=time.time() - t0)
    except Exception as exc:
        return EscalateResult(text="", tier="remote", escalated=True,
                               reason="remote_error", latency_s=time.time() - t0,
                               remote_error=str(exc))


def ping_remote(timeout=5.0):
    try:
        resp = requests.get(REMOTE_HOST + "/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False


# ---- Orchestration ---------------------------------------------------------

def _build_messages(question, history=None):
    messages = list(history) if history else []
    messages.append({"role": "user", "content": question})
    return messages


def answer(question, messages=None, history=None):
    score, signals = score_complexity(question)
    msgs = messages if messages is not None else _build_messages(question, history)
    timeout = LOCAL_TIMEOUT_HARD if score >= HARD_SCORE else LOCAL_TIMEOUT

    local = call_local(msgs, timeout=timeout)
    local.score, local.signals = score, signals
    _log("score=%d local=%s(%.2fs)" % (score, local.reason, local.latency_s))

    if local.reason == "local_ok":
        return local

    if not ESCALATE_ENABLED:
        local.tier = "none"
        local.reason = "remote_disabled"
        local.text = "[error] local failed and remote escalation is disabled: " + (local.local_error or "timeout")
        return local

    remote = call_remote(msgs)
    remote.score, remote.signals = score, signals
    remote.local_error = local.local_error
    _log("score=%d remote=%s(%.2fs)" % (score, remote.reason, remote.latency_s))

    if remote.reason == "remote_ok":
        remote.reason = local.reason  # preserve *why* we escalated
        return remote

    return EscalateResult(
        text="[error] could not reach either model: local=%s remote=%s"
        % (local.local_error, remote.remote_error),
        tier="none", escalated=True, reason="both_failed",
        latency_s=local.latency_s + remote.latency_s,
        score=score, signals=signals,
        local_error=local.local_error, remote_error=remote.remote_error,
    )


# ---- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]

    if argv and argv[0] == "--ping":
        print("remote reachable:", ping_remote())
        raise SystemExit(0)

    if argv and argv[0] == "--score":
        q = " ".join(argv[1:])
        if not q:
            print('usage: python3 escalate.py --score "<question>"')
            raise SystemExit(1)
        score, signals = score_complexity(q)
        hard = score >= HARD_SCORE
        print("score=%d (HARD_SCORE=%d) -> local timeout will be %.1fs"
              % (score, HARD_SCORE, LOCAL_TIMEOUT_HARD if hard else LOCAL_TIMEOUT))
        for k, v in signals.items():
            print("  %-15s %s" % (k, v))
        raise SystemExit(0)

    if not argv:
        print('usage: python3 escalate.py "<question>"')
        print('       python3 escalate.py --score "<question>"   (heuristic only, no model calls)')
        print('       python3 escalate.py --ping                 (check the Mac Studio is reachable)')
        raise SystemExit(1)

    question = " ".join(argv)
    t0 = time.time()
    result = answer(question)
    dt = time.time() - t0

    print("-" * 60)
    print("tier=%s escalated=%s reason=%s score=%d total=%.2fs"
          % (result.tier, result.escalated, result.reason, result.score, dt))
    if result.local_error:
        print("local error:", result.local_error)
    if result.remote_error:
        print("remote error:", result.remote_error)
    print("-" * 60)
    print(result.text)
