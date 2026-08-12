#!/usr/bin/env python3
"""Smoke test for brain.chat_stream_holdback.

Guards the phase-1 streaming path: the assembled text a holdback stream
returns must be identical to what the blocking chat_once path would have
produced from the same generation, a refusal must abort the stream with
nothing emitted (and stop consuming the generator), and transport errors must
fall back cleanly. Uses a fake in-process ollama module; no server needed.

Run:  python3 test_stream_holdback.py
Exits non-zero on any failure.
"""

import sys
import types


class FakeOllama(object):
    """Stands in for the ollama module. Serves one scripted generation per
    test and records how many stream parts were actually consumed."""

    def __init__(self, pieces, raise_at=None, raise_on_call=False):
        self.pieces = pieces
        self.raise_at = raise_at
        self.raise_on_call = raise_on_call
        self.consumed = 0

    def chat(self, model, messages, stream, options, keep_alive):
        if self.raise_on_call:
            raise RuntimeError("connection refused")

        def _gen():
            for n, piece in enumerate(self.pieces):
                if self.raise_at is not None and n == self.raise_at:
                    raise RuntimeError("mid-stream failure")
                self.consumed += 1
                yield {"message": {"content": piece}}
        return _gen()


def _install(fake):
    sys.modules["ollama"] = fake_module = types.ModuleType("ollama")
    fake_module.chat = fake.chat
    return fake_module


def _split(text, size=23):
    return [text[i:i + size] for i in range(0, len(text), size)]


def main():
    import brain

    failures = []

    def check(name, cond, detail=""):
        status = "ok  " if cond else "FAIL"
        print("%s %s%s" % (status, name, (" - " + detail) if detail else ""))
        if not cond:
            failures.append(name)

    long_answer = (
        "ASL is a complete natural language with its own grammar and syntax. "
        "Deaf children exposed to it from birth acquire language on a typical "
        "developmental timeline, and the research on the critical period shows "
        "that early accessible input is the deciding factor for later fluency, "
        "literacy, and wellbeing across the lifespan of the child.")
    long_refusal = (
        "INSUFFICIENT - the provided vault snippets do not contain any "
        "information about this topic. The notes cover language deprivation "
        "and sign language policy, but nothing in them addresses the question "
        "that was asked, so an answer cannot be grounded in the vault sources "
        "and the web would be needed to answer it responsibly.")

    # --- 1. long sufficient answer streams and reassembles exactly ---------
    fake = FakeOllama(_split(long_answer))
    _install(fake)
    sink_parts = []
    begins = []

    def begin():
        begins.append(1)
        return sink_parts.append

    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("long answer: streamed", streamed is True)
    check("long answer: text identical to blocking assembly",
          text == long_answer)
    check("long answer: sink received the full text once",
          "".join(sink_parts) == long_answer and len(begins) == 1)
    check("long answer: whole generation consumed",
          fake.consumed == len(fake.pieces))

    # --- 2. long refusal aborts with nothing emitted ------------------------
    fake = FakeOllama(_split(long_refusal))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("long refusal: not streamed", streamed is False)
    check("long refusal: nothing emitted", not sink_parts and not begins)
    check("long refusal: re-check on returned text still refuses",
          brain._is_insufficient(text))
    check("long refusal: generation aborted early",
          fake.consumed < len(fake.pieces),
          "consumed %d/%d" % (fake.consumed, len(fake.pieces)))

    # --- 3. short sufficient answer (ends under the holdback) ---------------
    short_answer = "ASL is a full natural language."
    fake = FakeOllama(_split(short_answer))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("short answer: streamed", streamed is True)
    check("short answer: text and sink identical",
          text == short_answer and "".join(sink_parts) == short_answer)

    # --- 4. short refusal (ends under the holdback) --------------------------
    fake = FakeOllama(_split("I can't answer this question."))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("short refusal: not streamed, nothing emitted",
          streamed is False and not sink_parts and not begins)

    # --- 5. transport error before anything emitted -> (None, False) --------
    fake = FakeOllama([], raise_on_call=True)
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("error before emission: falls back",
          text is None and streamed is False and not begins)

    # --- 6. mid-stream error after the flush keeps the partial --------------
    pieces = _split(long_answer)
    fake = FakeOllama(pieces, raise_at=len(pieces) - 1)
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    expected_partial = "".join(pieces[:-1])
    check("mid-stream error: partial returned as streamed",
          streamed is True and text == expected_partial)
    check("mid-stream error: interruption note emitted",
          "".join(sink_parts).startswith(expected_partial)
          and "interrupted" in "".join(sink_parts))

    # --- 7. whitespace-inflated refusal must NOT stream ----------------------
    # Regression for the raw-char holdback bug: 288 raw chars of padding
    # normalize to only ~180, so a raw-char counter would have decided
    # "sufficient" before the refusal phrase arrived and streamed a refusal.
    # The normalized-length counter keeps buffering until the phrase lands.
    padded_refusal = ("note    " * 36) + "I can't answer this question from the vault snippets."
    assert brain._normalized_len(padded_refusal[:288]) < 200 < len(padded_refusal[:288])
    check("padded refusal is a refusal at full length",
          brain._is_insufficient(padded_refusal))
    fake = FakeOllama(_split(padded_refusal))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("padded refusal: not streamed, nothing emitted",
          streamed is False and not sink_parts and not begins)

    # --- 7b. factual_short answers are shorter than the holdback ------------
    # The factual_short intent directive asks for one fact plus at most two
    # short sentences, so a correct answer now routinely ends BELOW
    # STREAM_HOLDBACK_CHARS. That lands in the "stream ended before the
    # threshold" branch, which must decide on the complete text and emit it --
    # if it ever failed open, every short factual answer would be swallowed and
    # the turn would fall through to the web.
    short_answer = ("Gallaudet University is in Washington, D.C. It is the only "
                    "university built from the ground up for deaf students "
                    "(Organizations/Gallaudet University.md).")
    check("short factual answer is under the holdback",
          brain._normalized_len(short_answer) < brain.STREAM_HOLDBACK_CHARS)
    check("short factual answer is not a refusal",
          not brain._is_insufficient(short_answer))
    fake = FakeOllama(_split(short_answer))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("short factual answer streams in full",
          streamed is True and text == short_answer
          and "".join(sink_parts) == short_answer and len(begins) == 1)

    # The mirror case: a refusal SHORTER than the holdback must still be caught.
    fake = FakeOllama(_split("INSUFFICIENT"))
    _install(fake)
    sink_parts = []
    begins = []
    text, streamed = brain.chat_stream_holdback(
        "m", [], brain._is_insufficient, begin)
    check("short refusal is still withheld",
          streamed is False and not sink_parts and not begins)

    # --- 8. stats reset: aborted stream leaves no stale counters -------------
    brain.LAST_CHAT_STATS = {"prompt_eval_count": 2900, "eval_count": 180}
    brain._reset_chat_stats()
    check("reset clears previous call's counters", brain.LAST_CHAT_STATS == {})

    print()
    if failures:
        print("%d check(s) FAILED: %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
