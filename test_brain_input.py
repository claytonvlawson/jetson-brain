#!/usr/bin/env python3
"""Tests for brain_input and the readline-driven prompt.

These exist because the terminal UI had no test at all, and the bug they guard
against is invisible to every other test in the repo: input() only does line
editing when the readline module has been imported, so a refactor that drops
`import readline` silently restores the "arrow keys print ^[[A" behaviour with
nothing failing.

Run: python3 test_brain_input.py
"""

import builtins
import re
import sys
import time

import brain_input
import brain_ui


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def visible(s):
    """The part of a prompt readline counts as occupying columns."""
    return re.sub("\001[^\002]*\002", "", s)


def test_readline_is_imported():
    # The whole fix in one assertion.
    check("readline module is available",
          brain_input.readline is not None,
          "(without it input() cannot do line editing)")
    check("input() is routed through readline",
          sys.modules.get("readline") is not None)


def test_ansi_prompt_brackets_escapes():
    caret = "\x1b[38;2;229;103;10m\x1b[1m> \x1b[0m"
    out = brain_input.ansi_prompt(caret)
    check("every escape run is bracketed",
          "\x1b" not in visible(out),
          "leftover: %r" % visible(out))
    check("visible width is the printing chars only",
          visible(out) == "> ",
          repr(visible(out)))
    check("brackets are balanced",
          out.count("\001") == out.count("\002"),
          "%d vs %d" % (out.count("\001"), out.count("\002")))
    check("plain text is untouched",
          brain_input.ansi_prompt("> ") == "> ")
    check("empty input is safe",
          brain_input.ansi_prompt("") == "" and brain_input.ansi_prompt(None) == "")


def test_ansi_prompt_handles_osc8():
    # Run.sources emits OSC 8 hyperlinks; they terminate on BEL, not on a CSI
    # final byte, so a CSI-only scanner would swallow the rest of the string.
    link = "\x1b]8;;http://x\x07text\x1b]8;;\x07"
    out = brain_input.ansi_prompt(link)
    check("OSC 8 does not eat the label",
          visible(out) == "text", repr(visible(out)))


def test_ansi_prompt_degrades_without_readline():
    real = brain_input.readline
    try:
        brain_input.readline = None
        out = brain_input.ansi_prompt("\x1b[1m> \x1b[0m")
        check("no \\001 markers emitted when readline is absent",
              "\001" not in out,
              "they would print literally")
    finally:
        brain_input.readline = real


def test_read_prompt_passes_the_prompt_into_input():
    """The regression that matters: a pre-printed caret leaves readline
    computing the cursor column from an empty prompt, so every redraw lands on
    top of the border."""
    ui = brain_ui.UI()
    ui.plain = False          # force the decorated path
    ui.out = sys.stdout       # _can_edit requires the real stdout
    seen = {}

    def fake_input(prompt=""):
        seen["prompt"] = prompt
        return "hello"

    real_input = builtins.input
    try:
        builtins.input = fake_input
        line = ui.read_prompt()
    finally:
        builtins.input = real_input

    check("read_prompt returns the typed line", line == "hello", repr(line))
    check("input() received a non-empty prompt",
          bool(seen.get("prompt")),
          "bare input() is the bug")
    check("the prompt carries zero-width markers",
          "\001" in seen.get("prompt", "") and "\002" in seen.get("prompt", ""))
    check("the caret is visible in the prompt",
          "❯" in seen.get("prompt", "") or ">" in visible(seen.get("prompt", "")))


def test_read_prompt_propagates_eof():
    ui = brain_ui.UI()
    ui.plain = False
    ui.out = sys.stdout

    def boom(prompt=""):
        raise EOFError()

    real_input = builtins.input
    raised = None
    try:
        builtins.input = boom
        ui.read_prompt()
    except EOFError as exc:
        raised = exc
    except Exception as exc:       # pragma: no cover
        raised = exc
    finally:
        builtins.input = real_input
    check("EOFError still propagates (Ctrl-D must exit)",
          isinstance(raised, EOFError), repr(raised))


def test_read_prompt_propagates_interrupt():
    ui = brain_ui.UI()
    ui.plain = False
    ui.out = sys.stdout

    def boom(prompt=""):
        raise KeyboardInterrupt()

    real_input = builtins.input
    raised = None
    try:
        builtins.input = boom
        ui.read_prompt()
    except KeyboardInterrupt as exc:
        raised = exc
    finally:
        builtins.input = real_input
    check("KeyboardInterrupt still propagates",
          isinstance(raised, KeyboardInterrupt), repr(raised))


def test_depth_stage_is_opt_in():
    ui = brain_ui.UI()
    plain = ui.start_run("llama3.2:3b")
    keys = [s.key for s in plain.stages]
    plain.finish()
    check("no depth stage by default",
          "depth" not in keys,
          "it rendered 'Depth pass (queued)' forever: %s" % keys)

    deep = ui.start_run("llama3.2:3b", depth=True)
    dkeys = [s.key for s in deep.stages]
    deep.finish()
    check("depth stage present when asked", "depth" in dkeys, str(dkeys))
    check("depth sits before the web fallback",
          dkeys.index("depth") < dkeys.index("web"), str(dkeys))


def test_unregistered_stage_is_a_noop():
    ui = brain_ui.UI()
    run = ui.start_run("llama3.2:3b")   # no depth stage registered
    try:
        run.running("depth")
        run.done("depth", "x")
        ok = True
    except Exception as exc:            # pragma: no cover
        ok = False
        print("     %r" % exc)
    finally:
        run.finish()
    check("driving an unregistered stage does not raise", ok)


def test_finish_is_idempotent():
    ui = brain_ui.UI()
    run = ui.start_run("llama3.2:3b")
    try:
        run.finish()
        run.finish()
        ok = True
    except Exception as exc:            # pragma: no cover
        ok = False
        print("     %r" % exc)
    check("finish() twice is safe (begin_answer already calls it)", ok)


def test_halt_animator_cannot_deadlock():
    """The muter must never wait on the lock the stuck animator is holding.

    _animate holds self.lock for the whole of _redraw, including the write. A
    first version of this fix set the mute flag under that same lock, so halting
    a thread wedged in write() blocked forever and hung the REPL -- strictly
    worse than the output corruption it was meant to fix. Regression guard.
    """
    import io
    import threading

    class BlockingTTY(io.StringIO):
        def __init__(self):
            io.StringIO.__init__(self)
            self.entered = threading.Event()
            self.release = threading.Event()

        def isatty(self):
            return True

        def write(self, s):
            self.entered.set()
            self.release.wait(10)     # simulate a wedged tty / SSH backpressure
            return len(s)

        def flush(self):
            pass

    buf = BlockingTTY()
    ui = brain_ui.UI()
    ui.out = buf
    ui.plain = False
    ui.tty = True
    ui.width = lambda: 80
    ui.height = lambda: 24

    run = brain_ui.Run(ui, [brain_ui.Stage("a", "A", "doing", "P")])
    if not buf.entered.wait(5):       # pragma: no cover
        check("animator reached the blocking write", False)
        return
    check("animator holds the lock while blocked", run.lock.locked())

    done = threading.Event()

    def halt():
        run._halt_animator()
        done.set()

    threading.Thread(target=halt, daemon=True).start()
    returned = done.wait(4)
    check("_halt_animator returns while the animator is stuck",
          returned, "DEADLOCK: it waited on the stuck thread's lock")
    check("the animator is muted", run._muted.is_set())

    buf.release.set()
    time.sleep(0.3)
    before = len(buf.getvalue())
    run._redraw()
    check("a muted frame writes nothing", len(buf.getvalue()) == before)
    try:
        run._halt_animator()
        ok = True
    except Exception as exc:          # pragma: no cover
        ok = False
        print("     %r" % exc)
    check("halting twice is safe", ok)


def test_import_is_silent_on_a_non_tty():
    """Importing this module must not put escape bytes into redirected output.

    brain.py imports brain_input at module scope, and brain.py is imported by
    eval_brain.py, eval_retrieval.py, brain_web.py, advisor_server.py and
    brain_ollama_shim.py. The atexit handler is armed at import, so without an
    isatty() guard every redirected run and every journald-logged service gained
    a trailing run of seven escape sequences -- and `eval_brain.py --json`, which
    goes to lengths to keep stdout pure JSON, stopped parsing.
    """
    import subprocess

    for mod in ("brain_input", "brain"):
        r = subprocess.run([sys.executable, "-c", "import " + mod],
                           capture_output=True)
        check("importing %s writes nothing to a pipe" % mod,
              r.stdout == b"",
              "emitted %r" % r.stdout[:60])

    # The realistic consumer: a machine-read JSON report on stdout.
    prog = "import brain, json; print(json.dumps({'ok': 1}))"
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True)
    try:
        import json
        json.loads(r.stdout.decode())
        ok = True
        why = ""
    except Exception as exc:
        ok = False
        why = repr(exc)
    check("stdout stays parseable as pure JSON", ok, why)


def test_depth_stage_never_stays_queued():
    """Every /scholar exit must resolve the depth stage.

    It is registered on the caller's condition (scholar_once and SCHOLAR_PASS2)
    but only driven inside the vault_ok branch, so a phase-1 INSUFFICIENT turn
    leaving via the skill loop or the web fallback used to freeze with
    "Depth pass (queued)" -- the very symptom this change removes. brain.py
    defaults it to SKIP before phase 1; done() overrides that when it runs.
    """
    import io

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    # The web-fallback exit: depth registered, second pass never taken.
    buf = FakeTTY()
    ui = brain_ui.UI()
    ui.out = buf
    ui.plain = False
    ui.tty = True
    ui.width = lambda: 100
    ui.height = lambda: 40
    run = ui.start_run("qwen2.5:7b", depth=True)
    run.skip("cache", "bypassed for /scholar")
    run.done("bm25", "12 chunks")
    run.done("expand", "")
    run.skip("depth", "vault answer needed no second pass")   # brain.py default
    run.done("model", "vault insufficient")
    run.done("web", "search")
    run.begin_answer()("web answer")
    run.finish()
    states = {s.key: s.state for s in run.stages}
    check("depth is resolved, not queued",
          states["depth"] != brain_ui.QUEUED,
          "state=%r" % states["depth"])
    check("no stage is left queued",
          all(v != brain_ui.QUEUED for v in states.values()),
          str(states))

    # And when the pass does run, done() must win over the earlier default skip.
    ui.out = FakeTTY()
    r2 = ui.start_run("qwen2.5:7b", depth=True)
    r2.skip("depth", "vault answer needed no second pass")
    r2.running("depth")
    r2.done("depth", "")
    r2.finish()
    check("done() overrides the default skip",
          r2.by_key["depth"].state == brain_ui.DONE,
          "state=%r" % r2.by_key["depth"].state)


def main():
    print("brain_input / prompt tests")
    for fn in (test_readline_is_imported,
               test_ansi_prompt_brackets_escapes,
               test_ansi_prompt_handles_osc8,
               test_ansi_prompt_degrades_without_readline,
               test_read_prompt_passes_the_prompt_into_input,
               test_read_prompt_propagates_eof,
               test_read_prompt_propagates_interrupt,
               test_depth_stage_is_opt_in,
               test_unregistered_stage_is_a_noop,
               test_finish_is_idempotent,
               test_halt_animator_cannot_deadlock,
               test_import_is_silent_on_a_non_tty,
               test_depth_stage_never_stays_queued):
        print("\n%s:" % fn.__name__)
        fn()
    print("")
    if FAILURES:
        print("FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
