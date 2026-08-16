#!/usr/bin/env python3
"""brain_ui.py - a Claude Code "ultracode" style terminal UI for the brain REPL.

Pure standard library, ANSI escape codes only (no rich / textual), so it runs
on the Jetson without extra installs. It renders:

  * a rounded welcome banner (model / vault / cache),
  * a bordered prompt row,
  * an animated phase -> agent -> result progress tree that maps the single
    RAG pipeline (cache check, BM25 retrieve, wikilink expand, model generate,
    web fallback) onto Claude Code's ultracode multi-agent layout,
  * a streamed answer block with a `sources:` branch.

Everything degrades to plain ASCII with no cursor control when stdout is not a
UTF-8 TTY (pipes, systemd, dumb terminals), so brain still works headless.

The animation runs on a background thread that redraws the tree in place every
~120 ms. Stage state is mutated by the main thread under a lock; when the model
starts producing the final answer the tree is frozen and the answer streams
below it as plain appended text.
"""

import os
import re
import sys
import time
import shutil
import threading
import urllib.parse


# ---------------------------------------------------------------------------
# Color palette. Truecolor when the terminal advertises it, else 256-color.
# ---------------------------------------------------------------------------
class Palette:
    """Claude Code terminal theme: deep orange on near-black. Primary accent is
    the Claude Code orange (#e5670a), used for borders, highlights, and the
    animated spinner. Truecolor when the terminal advertises it, else 256-color
    approximations of the same hues."""

    # Orange gradient stops (r,g,b): deep orange -> amber -> gold.
    SUNSET = [(229, 103, 10), (255, 140, 0), (255, 185, 50)]

    def __init__(self, truecolor):
        self.truecolor = truecolor
        if truecolor:
            self.ACCENT = "\x1b[38;2;229;103;10m"        # Claude Code orange
            self.ACCENT_DIM = "\x1b[38;2;180;80;8m"      # darker orange
            self.ACCENT_BRIGHT = "\x1b[38;2;255;140;0m"  # bright amber
            self.MUTED = "\x1b[38;2;148;163;184m"        # slate
            self.MUTED_DARK = "\x1b[38;2;100;116;139m"
            self.SUCCESS = "\x1b[38;2;52;211;153m"       # green
            self.ERROR = "\x1b[38;2;255;99;99m"
            self.BORDER = "\x1b[38;2;229;103;10m"        # Claude Code orange
            self.GOLD = "\x1b[38;2;255;185;50m"          # amber-gold
            self.CORAL = "\x1b[38;2;255;140;0m"          # orange-amber
            self.PINK = "\x1b[38;2;229;103;10m"          # base orange
        else:
            self.ACCENT = "\x1b[38;5;208m"               # orange
            self.ACCENT_DIM = "\x1b[38;5;202m"           # dark orange
            self.ACCENT_BRIGHT = "\x1b[38;5;214m"        # yellow-orange
            self.MUTED = "\x1b[38;5;245m"
            self.MUTED_DARK = "\x1b[38;5;240m"
            self.SUCCESS = "\x1b[38;5;79m"
            self.ERROR = "\x1b[38;5;203m"
            self.BORDER = "\x1b[38;5;208m"               # orange
            self.GOLD = "\x1b[38;5;214m"                 # yellow-orange
            self.CORAL = "\x1b[38;5;208m"                # orange
            self.PINK = "\x1b[38;5;208m"                 # orange
        self.RESET = "\x1b[0m"
        self.BOLD = "\x1b[1m"
        self.ITALIC = "\x1b[3m"
        # Per-frame spinner tint: shimmer through the orange-to-amber range.
        self.SPIN_TINT = [self.PINK, self.ACCENT, self.CORAL,
                          self.GOLD, self.ACCENT_BRIGHT, self.ACCENT_DIM]


# Spinner frames: Claude Code's rotating asterisk-star set.
SPIN_FRAMES = ["✻", "✢", "✳", "✶", "✺", "✵"]
SPIN_INTERVAL = 0.12  # seconds per frame

# Box drawing (UTF-8) with an ASCII fallback set.
BOX_U = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
         "h": "─", "v": "│"}
BOX_A = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"}

GLYPH_U = {"bullet": "⏺", "branch": "⎿", "ok": "✔",
           "fail": "✗", "skip": "⊘", "dot": "·", "caret": ">"}
GLYPH_A = {"bullet": "*", "branch": "+-", "ok": "[ok]", "fail": "[x]",
           "skip": "[skip]", "dot": "-", "caret": ">"}


def _supports_truecolor():
    ct = os.environ.get("COLORTERM", "")
    return "truecolor" in ct or "24bit" in ct


def _supports_unicode():
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


# Stage states.
QUEUED, RUNNING, DONE, SKIP, ERROR = "queued", "running", "done", "skip", "error"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _truncate_visible(s, width):
    """Truncate a string that may contain ANSI escape codes to `width` visible
    columns. Escape codes are kept but do not count toward the width, and a
    RESET is appended if the text was cut so color never bleeds. This keeps each
    animated status line from wrapping on narrow terminals, which would make the
    spinner drift downward frame by frame."""
    if width <= 0:
        return s
    out = []
    vis = 0
    i = 0
    n = len(s)
    cut = False
    while i < n:
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        if vis >= width:
            cut = True
            break
        out.append(s[i])
        vis += 1
        i += 1
    res = "".join(out)
    if cut:
        res += "\x1b[0m"
    return res


class Stage:
    __slots__ = ("key", "label", "verb", "phase", "state", "detail")

    def __init__(self, key, label, verb, phase):
        self.key = key
        self.label = label
        self.verb = verb
        self.phase = phase
        self.state = QUEUED
        self.detail = ""


class UI:
    """Owns terminal capabilities and renders every screen the brain shows."""

    def __init__(self, stream=None):
        self.out = stream or sys.stdout
        self.tty = False
        try:
            self.tty = self.out.isatty()
        except Exception:
            self.tty = False
        if os.environ.get("TERM", "") == "dumb":
            self.tty = False
        self.unicode = _supports_unicode()
        self.plain = not self.tty  # plain = no color, no cursor control
        self.truecolor = _supports_truecolor()
        self.p = Palette(self.truecolor)
        self.box = BOX_U if self.unicode else BOX_A
        self.g = GLYPH_U if self.unicode else GLYPH_A

    # -- low level --------------------------------------------------------
    def _w(self, s):
        try:
            self.out.write(s)
            self.out.flush()
        except Exception:
            pass

    def c(self, text, *codes):
        """Wrap text in color codes, then reset. No-op in plain mode."""
        if self.plain or not codes:
            return text
        return "".join(codes) + text + self.p.RESET

    def _osc8(self, url, text):
        """Wrap text in an OSC 8 clickable hyperlink. No-op in plain mode."""
        if self.plain:
            return text
        return "\x1b]8;;" + url + "\x1b\\" + text + "\x1b]8;;\x1b\\"

    def gradient(self, text, stops=None, bold=False):
        """Spread a truecolor gradient across the characters of `text` (a Nagai
        sunset by default). Falls back to a flat accent color when truecolor is
        unavailable, and to bare text in plain mode. `text` must be plain (no
        embedded escape codes)."""
        if self.plain:
            return text
        stops = stops or self.p.SUNSET
        if not self.truecolor or len(text) == 0:
            return self.c(text, self.p.ACCENT, *( (self.p.BOLD,) if bold else () ))
        chars = list(text)
        n = len(chars)
        segs = max(1, len(stops) - 1)
        b = self.p.BOLD if bold else ""
        out = []
        for i, ch in enumerate(chars):
            t = 0.0 if n == 1 else i / (n - 1)
            pos = t * segs
            k = min(int(pos), segs - 1)
            f = pos - k
            r = int(stops[k][0] + (stops[k + 1][0] - stops[k][0]) * f)
            g = int(stops[k][1] + (stops[k + 1][1] - stops[k][1]) * f)
            bl = int(stops[k][2] + (stops[k + 1][2] - stops[k][2]) * f)
            out.append("\x1b[38;2;%d;%d;%dm%s%s" % (r, g, bl, b, ch))
        return "".join(out) + self.p.RESET

    def width(self):
        try:
            return min(shutil.get_terminal_size((80, 24)).columns, 100)
        except Exception:
            return 80

    # -- welcome banner ---------------------------------------------------
    def welcome(self, model, chunk_count, cache_entries, vault_name, cache_dir):
        p, box = self.p, self.box
        # Two-column width: left info box + right tips box, matching Claude Code.
        left_inner = 36
        right_inner = 34
        total_inner = left_inner + right_inner + 3  # 3 for " | " separator

        # Render a pixel-art robot head in the Claude Code orange style.
        robot = [
            r"  ██████  ",
            r" ██ ██ ██ ",
            r"  ██████  ",
            r" ██    ██ ",
        ]

        def lpad(s, width):
            """Pad on the right to `width` VISIBLE chars: ANSI escape codes do
            not count, so colored and plain rows land on the same column."""
            pad = width - len(_ANSI_RE.sub("", s))
            return s + (" " * max(0, pad))

        # Left column content.
        left_lines = [lpad("", left_inner)]
        left_lines.append(lpad(
            "  " + self.c("Welcome back, Clayton!", p.ACCENT, p.BOLD),
            left_inner))
        left_lines.append(lpad("", left_inner))
        for rline in robot:
            left_lines.append(lpad(self.c(rline, p.ACCENT), left_inner))
        left_lines.append(lpad("", left_inner))

        def info_row(label, value, tail=""):
            colored = ("  " + self.c("%-7s" % (label + ":"), p.MUTED)
                       + " " + self.c(value, p.ACCENT))
            if tail:
                colored += " " + self.c(tail, p.MUTED_DARK)
            left_lines.append(lpad(colored, left_inner))

        info_row("model", _short_model(model), "(Ollama)")
        info_row("vault", vault_name, "%s chunks" % _commas(chunk_count))
        info_row("cache", "%d entries" % cache_entries)
        info_row("host", "localhost:11434")

        left_lines.append(lpad("", left_inner))

        # Right column: tips, matching Claude Code's "Tips for getting started" panel.
        right_lines = [lpad(
            " " + self.c("Tips for getting started", p.ACCENT, p.BOLD),
            right_inner)]
        right_lines.append(lpad("", right_inner))

        tips = [
            "Ask a question; brain retrieves",
            "  from the vault, then answers.",
            "",
            "/web on|off  toggles DuckDuckGo",
            "  web fallback.",
            "",
            "/model  picks the Ollama model.",
            "/cache  inspects the answer bank.",
            "",
            "Ctrl-C  interrupts a run.",
            "Ctrl-D or /exit  quits.",
        ]
        for tip in tips:
            right_lines.append(lpad(" " + self.c(tip, p.MUTED), right_inner))

        # Right column: "What's new" section.
        right_lines.append(lpad("", right_inner))
        right_lines.append(lpad(
            " " + self.c("What's new", p.ACCENT, p.BOLD), right_inner))
        right_lines.append(lpad("", right_inner))
        whatsnew = [
            "Two-tier models: /deep <q> uses",
            "  qwen2.5:7b on hard questions.",
            "",
            "Wikilink graph expansion pulls",
            "  linked notes into context.",
            "",
            "Semantic answer cache for instant",
            "  repeat answers.",
        ]
        for line in whatsnew:
            right_lines.append(lpad(" " + self.c(line, p.MUTED), right_inner))

        # Pad columns to the same height.
        while len(left_lines) < len(right_lines):
            left_lines.append(lpad("", left_inner))
        while len(right_lines) < len(left_lines):
            right_lines.append(lpad("", right_inner))

        # Orange dashed outer border (Claude Code style).
        h_dash = (box["h"] * total_inner) if self.unicode else ("-" * total_inner)
        top = " " + self.c(box["tl"] + h_dash + box["tr"], p.ACCENT)
        bot = " " + self.c(box["bl"] + h_dash + box["br"], p.ACCENT)
        # Inner vertical divider between left and right panels.
        div = self.c(box["v"], p.ACCENT_DIM)

        lines = [top]
        for lc, rc in zip(left_lines, right_lines):
            lines.append(" " + div + lc + " " + div + " " + rc + div)
        lines.append(bot)
        self._w("\n".join(lines) + "\n\n")

    # -- prompt row -------------------------------------------------------
    def read_prompt(self):
        """Render the bordered prompt box and read one line. Returns the text,
        or raises EOFError / KeyboardInterrupt like input() does."""
        p, box = self.p, self.box
        w = self.width()
        inner = max(20, w - 3)
        if not self.plain:
            top = " " + self.c(box["tl"] + box["h"] * inner + box["tr"], p.BORDER)
            bot = " " + self.c(box["bl"] + box["h"] * inner + box["br"], p.BORDER)
            chips = "   " + self.c("/web  /model  /cache  /clear  /exit", p.MUTED_DARK)
            hint = self.c("⏎ send", p.MUTED_DARK) if self.unicode else self.c("enter=send", p.MUTED_DARK)
            self._w(top + "\n")
            # Middle row: caret, then the terminal echoes typed input after it.
            self._w(" " + self.c(box["v"], p.BORDER) + " "
                    + self.c(self.g["caret"], p.ACCENT, p.BOLD) + " ")
            # We cannot both draw a closing border AND use input() echo on the
            # same line cleanly, so read here; the bottom border follows.
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self._w("\n")
                raise
            self._w(bot + "\n")
            self._w(chips + "    " + hint + "\n")
            return line
        # Plain mode: simple prompt.
        return input("\nyou> ")

    # -- run / progress tree ----------------------------------------------
    def start_run(self, model):
        """Begin a pipeline run and return a Run handle. Stages are preset so
        the tree shows queued/skipped items just like an ultracode run."""
        stages = [
            Stage("cache", "Cache check", "Checking cache", "Retrieval"),
            Stage("bm25", "BM25 retrieve", "Retrieving", "Retrieval"),
            Stage("expand", "Wikilink expand", "Expanding links", "Retrieval"),
            Stage("model", model, "Consulting " + _short_model(model), "Generation"),
            # The scholar persona's second pass is a whole extra generation
            # (~30-45s) with streaming forced off, so without a registered
            # stage the user watches a frozen spinner for the duration. Note
            # that Run.by_key makes an UNregistered key a silent no-op: that is
            # why run.done("obsidian") and run.done("skills") already print
            # nothing today. A stage must be listed here to exist at all.
            Stage("depth", "Depth pass", "Re-retrieving missed facets", "Generation"),
            Stage("web", "Web fallback", "Fetching web", "Generation"),
        ]
        return Run(self, stages)

    # -- simple messages --------------------------------------------------
    def note(self, text):
        self._w(" " + self.c(text, self.p.MUTED) + "\n")


class Run:
    """One question's lifecycle: an animated progress tree, then a streamed
    answer. Thread-safe stage mutation; the animator only reads."""

    def __init__(self, ui, stages):
        self.ui = ui
        self.stages = stages
        self.by_key = {s.key: s for s in stages}
        self.lock = threading.Lock()
        self.start = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        self._last_lines = 0
        self._frame = 0
        self._streaming = False
        self._detail_live = ""  # live token shown next to the running spinner
        if not self.ui.plain:
            self.ui._w("\x1b[?25l")  # hide cursor during animation
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    # -- public stage API -------------------------------------------------
    def running(self, key, live=""):
        with self.lock:
            s = self.by_key.get(key)
            if s:
                s.state = RUNNING
            self._detail_live = live
        if self.ui.plain:
            self.ui._w(" ... %s\n" % (s.verb if s else key))

    def done(self, key, detail=""):
        with self.lock:
            s = self.by_key.get(key)
            if s:
                s.state = DONE
                s.detail = detail
            self._detail_live = ""
        if self.ui.plain and s:
            self.ui._w("  [ok] %s%s\n" % (s.label, (" - " + detail) if detail else ""))

    def skip(self, key, detail=""):
        with self.lock:
            s = self.by_key.get(key)
            if s:
                s.state = SKIP
                s.detail = detail
        if self.ui.plain and s and detail:
            self.ui._w("  [skip] %s - %s\n" % (s.label, detail))

    def fail(self, key, detail=""):
        with self.lock:
            s = self.by_key.get(key)
            if s:
                s.state = ERROR
                s.detail = detail
        if self.ui.plain and s:
            self.ui._w("  [x] %s%s\n" % (s.label, (" - " + detail) if detail else ""))

    # -- streaming answer -------------------------------------------------
    def begin_answer(self):
        """Freeze the tree, print it once statically, and open the answer
        block with the accent bullet. Returns a token sink callable."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if not self.ui.plain:
            self._redraw()
            self.ui._w("\x1b[?25h")  # restore cursor
        self._streaming = True
        self._col = 0
        self._width = self.ui.width()
        g = self.ui.g
        # Blank line, then the answer bullet -- a gold Nagai sun.
        self.ui._w("\n " + self.ui.c(g["bullet"], self.ui.p.GOLD, self.ui.p.BOLD) + " ")
        self._col = 3
        return self._emit

    def _emit(self, piece):
        """Append streamed text under the answer bullet with soft wrapping."""
        if not piece:
            return
        for ch in piece:
            if ch == "\n":
                self.ui._w("\n   ")
                self._col = 3
                continue
            self.ui._w(ch)
            self._col += 1
            if self._col >= self._width - 2 and ch == " ":
                self.ui._w("\n   ")
                self._col = 3

    def sources(self, srcs, vault_name="ThesisVault"):
        """Print the `sources:` branch under the answer. srcs is a list of
        strings (vault filenames or urls). Vault sources are rendered as
        clickable OSC 8 hyperlinks that open the note in Obsidian."""
        p, g = self.ui.p, self.ui.g
        self.ui._w("\n")
        if not srcs:
            self.ui._w("\n")
            return
        branch = self.ui.c(g["branch"], p.MUTED_DARK)
        label = self.ui.c("sources:", p.MUTED, p.BOLD)
        self.ui._w("   " + branch + " " + label + "\n")
        encoded_vault = urllib.parse.quote(vault_name, safe='')
        for s in srcs:
            is_web = str(s).startswith("http")
            dot = self.ui.c(g["dot"], p.MUTED_DARK)
            if is_web:
                url = str(s)
                tag = self.ui.c("web:", p.ACCENT_DIM)
                display = self.ui.c(str(s), p.MUTED)
                self.ui._w("       " + dot + " " + tag + " "
                           + self.ui._osc8(url, display) + "\n")
            else:
                encoded_file = urllib.parse.quote(str(s), safe='')
                url = "obsidian://open?vault=" + encoded_vault + "&file=" + encoded_file
                display = self.ui.c(str(s), p.MUTED)
                self.ui._w("       " + dot + " "
                           + self.ui._osc8(url, display) + "\n")
        self.ui._w("\n")

    def finish(self):
        """Ensure the animator is stopped and the cursor restored."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if not self.ui.plain:
            self.ui._w("\x1b[?25h")

    # -- rendering --------------------------------------------------------
    def _animate(self):
        while not self._stop.is_set():
            with self.lock:
                self._frame = (self._frame + 1) % len(SPIN_FRAMES)
                self._redraw()
            time.sleep(SPIN_INTERVAL)

    def _redraw(self):
        """Compose and print the tree, rewinding over the previous draw."""
        lines = self._compose()
        # Truncate to the terminal width so no status line wraps. A wrapped line
        # occupies 2 physical rows but is counted once, so the cursor-up rewind
        # below would under-shoot and the tree would creep downward each frame.
        w = self.ui.width() - 1
        lines = [_truncate_visible(ln, w) for ln in lines]
        buf = []
        if self._last_lines:
            buf.append("\x1b[%dA" % self._last_lines)  # cursor up
        buf.append("\r")
        for ln in lines:
            buf.append("\x1b[2K" + ln + "\n")  # clear line, print
        buf.append("\x1b[0J")  # clear anything below
        self.ui._w("".join(buf))
        self._last_lines = len(lines)

    def _compose(self):
        p, g = self.ui.p, self.ui.g
        elapsed = int(time.monotonic() - self.start)
        spin = SPIN_FRAMES[self._frame] if self.ui.unicode else "*"
        lines = []
        last_phase = None
        # Fixed column for the right-aligned status marker.
        markcol = 62
        for s in self.stages:
            if s.phase != last_phase:
                lines.append("")
                lines.append(" " + self.ui.gradient(s.phase, p.SUNSET, bold=True))
                last_phase = s.phase
            if s.state == RUNNING:
                # Shimmer the working glyph through the Nagai sunset per frame.
                glyph = self.ui.c(spin, p.SPIN_TINT[self._frame % len(p.SPIN_TINT)])
                label = self.ui.c(s.verb + "…" if self.ui.unicode else s.verb + "...",
                                  p.ACCENT, p.ITALIC)
                tail = " " + self.ui.c("(%ds%s)" % (
                    elapsed, (" %s %s" % (g["dot"], self._detail_live)) if self._detail_live else ""),
                    p.MUTED)
                lines.append("   " + glyph + " " + label + tail)
            elif s.state == DONE:
                glyph = self.ui.c(g["bullet"], p.SUCCESS)
                text = s.label
                pad = max(1, markcol - len(text) - 4)
                lines.append("   " + glyph + " " + text + (" " * pad)
                             + self.ui.c(g["ok"], p.SUCCESS))
                if s.detail:
                    lines.append("     " + self.ui.c(g["branch"], p.MUTED_DARK)
                                 + " " + self.ui.c(s.detail, p.MUTED))
            elif s.state == SKIP:
                glyph = self.ui.c(g["bullet"], p.MUTED_DARK)
                text = self.ui.c(s.label, p.MUTED_DARK)
                pad = max(1, markcol - len(s.label) - 4)
                lines.append("   " + glyph + " " + text + (" " * pad)
                             + self.ui.c(g["skip"], p.MUTED))
                if s.detail:
                    lines.append("     " + self.ui.c(g["branch"], p.MUTED_DARK)
                                 + " " + self.ui.c(s.detail, p.MUTED_DARK))
            elif s.state == ERROR:
                glyph = self.ui.c(g["bullet"], p.ERROR)
                text = s.label
                pad = max(1, markcol - len(text) - 4)
                lines.append("   " + glyph + " " + text + (" " * pad)
                             + self.ui.c(g["fail"], p.ERROR))
                if s.detail:
                    lines.append("     " + self.ui.c(g["branch"], p.MUTED_DARK)
                                 + " " + self.ui.c(s.detail, p.ERROR))
            else:  # QUEUED
                glyph = self.ui.c(g["bullet"], p.MUTED_DARK)
                lines.append("   " + glyph + " "
                             + self.ui.c(s.label + "  (queued)", p.MUTED_DARK))
        return lines


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _commas(n):
    try:
        return "{:,}".format(int(n))
    except Exception:
        return str(n)


def _short_model(model):
    try:
        return str(model).split(":")[0]
    except Exception:
        return str(model)
