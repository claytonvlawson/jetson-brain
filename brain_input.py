"""Terminal input hygiene: line editing, history, scroll quieting, cursor.

Why this module exists at all.

CPython only routes input() through libreadline when the `readline` module has
been IMPORTED -- importing it is what patches PyOS_ReadlineFunctionPointer. For
a script (as opposed to the interactive interpreter) `site` never imports it, so
until this module ran, brain.py's REPL read every line in plain canonical mode:
the kernel line discipline gives backspace and Ctrl-U and nothing else. An arrow
key's bytes (ESC [ A) landed in the line buffer and the tty echoed them as the
literal junk "^[[A". A mouse wheel tick reached the same place by a longer road,
because a terminal with ALTERNATE SCROLL on translates the wheel into those very
arrow sequences.

So the two reported symptoms -- "arrow keys print garbage" and "scrolling prints
garbage" -- were one defect with two entry points, and the fix is one import.

This lives in its own module rather than in brain.py because review_qa.py has
the same defect and cannot import brain.py (circularity, and brain.py costs
seconds to import). Everything here is best-effort: a terminal helper must never
be the reason the program fails to answer a question.
"""

import atexit
import os
import sys

try:
    import readline
except ImportError:      # pragma: no cover - readline is stdlib on Linux/macOS
    readline = None

# libedit (macOS system Python) and GNU readline (the Jetson) disagree on
# parse_and_bind syntax. We bind nothing today, so this is recorded for callers
# rather than acted on -- but it must be checked BEFORE any parse_and_bind is
# ever added, or the binding silently does nothing on one of the two machines.
USING_LIBEDIT = bool(readline) and "libedit" in (readline.__doc__ or "")

# ---------------------------------------------------------------------------
# Mouse / scroll quieting.
#
# ESC [ ? 1000 l  disable X10/normal mouse click reporting
# ESC [ ? 1002 l  disable button-event (drag) tracking
# ESC [ ? 1003 l  disable any-event (motion) tracking
# ESC [ ? 1006 l  disable SGR extended mouse mode
# ESC [ ? 1015 l  disable urxvt extended mouse mode
# ESC [ ? 1007 l  disable ALTERNATE SCROLL -- the one that turns wheel into arrows
# ---------------------------------------------------------------------------
_DISABLE_MOUSE = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?1007l"

SHOW_CURSOR = "\x1b[?25h"
HIDE_CURSOR = "\x1b[?25l"

# readline's "this span occupies zero columns" brackets. Without them readline
# counts the bytes of a color escape as printing characters and every redraw
# lands one prompt-width off.
RL_START_IGNORE = "\001"
RL_END_IGNORE = "\002"

_ESC = "\x1b"

HISTORY_LEN = 1000
_history_path = None


def _write(s):
    """Write to the real stdout, flushed, swallowing every failure."""
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except Exception:
        pass


def _isatty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def quiet_scroll():
    """Stop the terminal turning mouse/trackpad scrolls into keystrokes.

    Safe to call repeatedly, and it SHOULD be called repeatedly: the modes are
    terminal state, not our state, so a subprocess ($EDITOR, less), an SSH
    remote or a tmux re-attach can turn mouse reporting back on behind our back
    and reintroduce the symptom mid-session. The original one-shot call at
    startup was the reason scrolling still leaked characters after a while.

    No-op when stdout is not a TTY (piped or logged). Never raises.
    """
    if not _isatty():
        return
    _write(_DISABLE_MOUSE)


def restore_cursor():
    """Make the cursor visible again. Idempotent; ?25h is a mode, not a count."""
    if not _isatty():
        return
    _write(SHOW_CURSOR)


def _restore_terminal():
    """atexit: leave the terminal at least as usable as we found it.

    Registered once, at import. The animator hides the cursor per question and
    previously relied on a matching show() on paths that a KeyboardInterrupt in
    the cache fast path could skip entirely, which dumped the user back to a
    shell with no cursor at all.

    The isatty() guard is NOT optional and must match quiet_scroll() and
    restore_cursor(). brain.py imports this module at module scope, and brain.py
    is in turn imported by eval_brain.py, eval_retrieval.py, brain_web.py,
    advisor_server.py and brain_ollama_shim.py -- so without the guard every
    redirected run and every journald-logged service appends seven raw escape
    sequences after its last line of output. `eval_brain.py --json` goes to
    lengths to keep stdout pure JSON, and a trailing escape run breaks
    json.loads with "Extra data: line 2 column 1".
    """
    if not _isatty():
        return
    _write(SHOW_CURSOR + _DISABLE_MOUSE)


def ansi_prompt(text):
    """Bracket every escape run in `text` so readline counts it as zero-width.

    readline derives the cursor column from the prompt string it was handed. A
    colored caret is ~25 bytes of SGR escape for 2 printing columns, so an
    unbracketed prompt puts every redraw, wrap and history recall about 23
    columns off. Each ESC..final-byte run is wrapped in \\001..\\002.

    Returns text unchanged when readline is unavailable (the brackets would then
    be printed literally, which is worse than doing nothing).
    """
    if not text or readline is None:
        return text or ""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != _ESC:
            out.append(ch)
            i += 1
            continue
        # Consume one escape sequence: ESC, then bytes up to and including the
        # final byte. CSI (ESC [) ends at @-~; OSC (ESC ]) ends at BEL or ST.
        j = i + 1
        if j < n and text[j] == "]":
            while j < n and text[j] != "\a":
                if text[j] == _ESC and j + 1 < n and text[j + 1] == "\\":
                    j += 1
                    break
                j += 1
        else:
            if j < n and text[j] == "[":
                j += 1
            while j < n and not ("@" <= text[j] <= "~"):
                j += 1
        j = min(j + 1, n)
        out.append(RL_START_IGNORE + text[i:j] + RL_END_IGNORE)
        i = j
    return "".join(out)


def setup(history_path=None):
    """Install line editing and history. Call once, early, before any prompt.

    Importing readline is the whole fix for the arrow-key junk; everything else
    here is comfort. Returns True when line editing is actually available.
    """
    global _history_path
    quiet_scroll()
    if readline is None:
        return False
    try:
        readline.set_history_length(HISTORY_LEN)
    except Exception:
        pass
    if history_path:
        _history_path = os.path.expanduser(history_path)
        try:
            os.makedirs(os.path.dirname(_history_path), exist_ok=True)
        except Exception:
            pass
        try:
            readline.read_history_file(_history_path)
        except Exception:
            # Missing or unreadable history is normal on a first run.
            pass
        atexit.register(save_history)
    return True


def save_history():
    """Persist input history. Never raises; a read-only home is not an error."""
    if readline is None or not _history_path:
        return
    try:
        readline.write_history_file(_history_path)
    except Exception:
        pass


def read_line(prompt=""):
    """input() with the mouse modes re-armed and the cursor guaranteed visible.

    `prompt` is passed INTO input() rather than pre-printed, which is what lets
    readline compute the cursor column correctly. Wrap any colored prompt in
    ansi_prompt() first.

    Propagates EOFError and KeyboardInterrupt exactly like input() does, so
    callers keep their existing control flow.
    """
    quiet_scroll()
    restore_cursor()
    return input(prompt)


# Registered at import so it covers every exit path that runs atexit at all,
# including the ones that previously leaked a hidden cursor.
atexit.register(_restore_terminal)
