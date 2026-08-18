#!/usr/bin/env python3
"""Tests for brain_transcript and the /history command.

The bug this guards against is the one the feature exists to fix: before it,
nothing anywhere persisted an answer, so a restart erased every conversation.
The subtler failure it also guards is over-persistence -- brain_web.py (:8899)
and advisor_server.py (:8000) both reach _finish through handle_question, and
they must NEVER write into the REPL's personal history. That isolation is not a
policy comment, it is the fact that only main() calls enable(); this file pins
it down.

Run: python3 test_brain_transcript.py
"""

import json
import os
import shutil
import sys
import tempfile

import brain_transcript


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


class Temp(object):
    """A fresh, armed transcript per test; disarmed and removed after."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="brain_transcript_")
        self.path = os.path.join(self.dir, "transcript.jsonl")
        brain_transcript.enable(self.path)
        return self

    def __exit__(self, *exc):
        brain_transcript._path = None
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def test_disabled_is_a_silent_noop():
    brain_transcript._path = None
    check("append() is False while disabled",
          brain_transcript.append("q", "a") is False)
    check("read() is empty while disabled", brain_transcript.read() == [])
    check("clear() is 0 while disabled", brain_transcript.clear() == 0)
    check("enabled() is False", brain_transcript.enabled() is False)


def test_a_turn_survives_a_restart():
    # The requirement, stated as a test: write, drop all module state as a
    # process exit would, re-enable the same path, read it back.
    with Temp() as t:
        brain_transcript.set_turn("qwen2.5:3b", "deep")
        brain_transcript.append("what is language deprivation?",
                                "the answer", ["/v/H2016.md"])
        brain_transcript._path = None
        brain_transcript._model = ""
        brain_transcript._mode = "fast"
        brain_transcript.enable(t.path)
        recs = brain_transcript.read()
        check("one turn read back", len(recs) == 1, repr(recs))
        if recs:
            check("question preserved", recs[0]["q"] == "what is language deprivation?")
            check("answer preserved", recs[0]["a"] == "the answer")
            check("sources preserved", recs[0]["sources"] == ["/v/H2016.md"])
            check("model preserved", recs[0]["model"] == "qwen2.5:3b")
            check("mode preserved", recs[0]["mode"] == "deep")


def test_file_is_owner_only():
    with Temp() as t:
        brain_transcript.append("q", "a")
        mode = os.stat(t.path).st_mode & 0o777
        check("transcript is mode 0600", mode == 0o600, oct(mode))


def test_a_corrupt_line_is_skipped_not_fatal():
    # A transcript truncated by a full disk must still show everything before
    # the truncation rather than blowing up /history.
    with Temp() as t:
        brain_transcript.append("q1", "a1")
        with open(t.path, "a") as fh:
            fh.write("{not json at all\n\n")
        brain_transcript.append("q2", "a2")
        recs = brain_transcript.read()
        check("good lines survive a corrupt one",
              [r["q"] for r in recs] == ["q1", "q2"], repr(recs))


def test_limit_and_search():
    with Temp():
        brain_transcript.append("alpha question", "about DEPRIVATION")
        brain_transcript.append("beta question", "unrelated")
        brain_transcript.append("gamma question", "unrelated")
        check("limit keeps the newest",
              [r["q"] for r in brain_transcript.read(limit=1)] == ["gamma question"])
        check("search matches the answer body, case-insensitively",
              [r["q"] for r in brain_transcript.read(search="deprivation")]
              == ["alpha question"])
        check("search matches the question too",
              len(brain_transcript.read(search="beta")) == 1)
        check("a search with no match is empty",
              brain_transcript.read(search="zzzz") == [])


def test_clear_reports_and_empties():
    with Temp():
        brain_transcript.append("q1", "a1")
        brain_transcript.append("q2", "a2")
        check("clear() returns the removed count", brain_transcript.clear() == 2)
        check("nothing left after clear", brain_transcript.read() == [])


def test_rotation_bounds_the_file():
    with Temp() as t:
        saved = brain_transcript.MAX_BYTES
        brain_transcript.MAX_BYTES = 200
        try:
            for i in range(20):
                brain_transcript.append("q%d" % i, "x" * 100)
            check("live file stayed under the cap",
                  os.path.getsize(t.path) < 2 * brain_transcript.MAX_BYTES,
                  str(os.path.getsize(t.path)))
            check("one rotated generation exists",
                  os.path.exists(t.path + ".1"))
        finally:
            brain_transcript.MAX_BYTES = saved


def test_repl_only_isolation():
    # brain.py's _log_turn is the ONLY writer, and it is inert until main()
    # arms the module. brain_web / advisor_server import brain and never do.
    import brain
    brain_transcript._path = None
    brain._log_turn("web question", "web answer", [])
    check("_log_turn writes nothing while disabled",
          brain_transcript.read() == [])
    with Temp():
        brain_transcript.set_turn("m", "fast")
        brain._log_turn("repl question", "repl answer", [])
        check("_log_turn writes once armed",
              [r["q"] for r in brain_transcript.read()] == ["repl question"])
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "brain_web.py")).read()
    check("brain_web.py never arms the transcript",
          "brain_transcript" not in src)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "advisor_server.py")).read()
    check("advisor_server.py never arms the transcript",
          "brain_transcript" not in src)


def test_history_command_renders():
    import brain
    with Temp():
        brain_transcript.set_turn("qwen2.5:3b", "scholar")
        brain._log_turn("q one", "a one", ["/v/H2016 Deprivation.md"])
        text = brain._history_text(brain_transcript.read(), None)
        check("render includes the question", "q one" in text)
        check("render includes the answer", "a one" in text)
        check("render includes the sources", "H2016 Deprivation" in text)
        check("render includes the mode", "scholar" in text)
        check("render is uncolored without a UI", "\x1b[" not in text)


def main():
    for fn in (test_disabled_is_a_silent_noop,
               test_a_turn_survives_a_restart,
               test_file_is_owner_only,
               test_a_corrupt_line_is_skipped_not_fatal,
               test_limit_and_search,
               test_clear_reports_and_empties,
               test_rotation_bounds_the_file,
               test_repl_only_isolation,
               test_history_command_renders):
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
