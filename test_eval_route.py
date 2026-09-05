"""Unit tests for eval_route.py -- no network, no index, no Ollama.

    python3 test_eval_route.py

Covers the parts that decide whether the measurement can lie: a strict-mode
cloud error is scored 0 and never judged; a leaked local fallback in the cloud
arm aborts; the judge transport seam feeds eval_brain.judge() unchanged;
aggregate/paired arithmetic; the --answers-out JSONL shape; the budget stop.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_brain  # noqa: E402
import eval_route  # noqa: E402

GOLD = [
    {"id": "q1", "question": "Where is Gallaudet?", "must_cover": ["Gallaudet is in Washington, DC"],
     "must_not": ["Gallaudet is in Maryland"], "_gold_file": "factual_gold.jsonl"},
    {"id": "q2", "question": "When was WFD founded?", "must_cover": ["WFD was founded in 1951", "WFD is in Helsinki"],
     "must_not": [], "_gold_file": "factual_gold.jsonl"},
]


def fake_build(question, persona="default", index=None):
    return ([{"role": "system", "content": "sys"},
             {"role": "user", "content": "Context:\n[a.md] text\n\nQuestion: " + question}],
            [{"source": "a.md", "title": "A", "text": "text"}])


def make_gen(cloud_error_ids=(), leak=False):
    def gen(messages, tier, strict=False, fallback="local-fast", rate_limit_wait_s=0.0):
        q = messages[1]["content"].rsplit("Question: ", 1)[1]
        qid = "q1" if "Gallaudet" in q else "q2"
        if tier == "cloud-speed":
            if leak:
                return {"text": "local text", "served_by": "local-fast", "model": "llama3.2:3b",
                        "seconds": 5.0, "attempts": 0, "error": None}
            if qid in cloud_error_ids:
                return {"text": "", "served_by": "cloud-error", "model": "openai/gpt-oss-20b",
                        "seconds": 8.1, "attempts": 2, "error": "empty content on 2 attempt(s)"}
            return {"text": "Gallaudet is in Washington, DC. WFD 1951, Helsinki.", "served_by": "cloud",
                    "model": "openai/gpt-oss-20b", "seconds": 0.4, "attempts": 1, "error": None,
                    "finish_reason": "stop", "usage": {"output_tokens": 30}}
        return {"text": "Gallaudet is in Washington, DC.", "served_by": "local-fast",
                "model": "llama3.2:3b", "seconds": 6.0, "attempts": 0, "error": None}
    return gen


class FakeJudgeTransport(object):
    """Stands in for eval_brain._ollama_chat: credits a fact when a keyword
    from it appears in the answer, so the verdict depends on the ANSWER text."""
    def __init__(self):
        self.calls = 0

    def __call__(self, model, messages, host, want_json, timeout):
        self.calls += 1
        prompt = messages[1]["content"]
        answer = prompt.split("ANSWER TO GRADE:\n", 1)[1].split("\n\nRespond", 1)[0]
        facts = prompt.split("MUST-COVER facts", 1)[1].split("MUST-NOT", 1)[0]
        cover = []
        for line in facts.splitlines():
            if line[:1].isdigit():
                i, fact = line.split(". ", 1)
                key = fact.split()[-1].strip(".,")
                cover.append({"i": int(i), "present": key in answer})
        return json.dumps({"cover": cover, "violate": [{"j": 1, "asserted": False}]})


class EvalRouteTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeJudgeTransport()
        self._orig = eval_brain._ollama_chat
        eval_brain._ollama_chat = self.transport
        eval_route.RATE_LIMIT.update({"hits": 0, "slept_s": 0.0})

    def tearDown(self):
        eval_brain._ollama_chat = self._orig

    def _run(self, gen, **kw):
        return eval_route.run(GOLD, None, ["local", "cloud"], "default", "groq", "fake-judge",
                              "fake-host", gen_fn=gen, build_fn=fake_build, parallel=False, **kw)

    def test_cloud_error_scores_zero_without_judging(self):
        results, partial, _ = self._run(make_gen(cloud_error_ids={"q2"}))
        self.assertFalse(partial)
        err = [r for r in results if r["arm"] == "cloud" and r["id"] == "q2"][0]
        self.assertEqual(err["served_by"], "cloud-error")
        self.assertEqual(err["score"], 0)
        self.assertEqual(err["judge_raw"], "generation-error")
        self.assertEqual(err["missing"], GOLD[1]["must_cover"])
        self.assertTrue(err["over_prod_timeout"])
        # 4 rows, one is an error -> exactly 3 judge calls
        self.assertEqual(self.transport.calls, 3)

    def test_fallback_leak_aborts(self):
        with self.assertRaises(RuntimeError):
            self._run(make_gen(leak=True))

    def test_scores_come_from_eval_brain_judge(self):
        results, _, _ = self._run(make_gen())
        by = {(r["arm"], r["id"]): r for r in results}
        self.assertEqual(by[("local", "q1")]["score"], 100)      # DC present
        self.assertEqual(by[("local", "q2")]["score"], 0)        # neither 1951 nor Helsinki
        self.assertEqual(by[("cloud", "q2")]["score"], 100)
        self.assertEqual(by[("cloud", "q1")]["judge_raw"], "votes=1/1")

    def test_aggregate_and_paired(self):
        results, _, _ = self._run(make_gen(cloud_error_ids={"q1"}))
        agg_l = eval_route.aggregate(results, "local")
        agg_c = eval_route.aggregate(results, "cloud")
        self.assertEqual(agg_l["n_scored"], 2)
        self.assertEqual(agg_l["mean_score"], 50.0)
        self.assertEqual(agg_l["pass_at_90"], 1)
        self.assertEqual(agg_l["fact_hit_rate"], round(1 / 3, 3))
        self.assertEqual(agg_l["latency_median_s"], 6.0)
        self.assertEqual(agg_c["n_generation_errors"], 1)
        self.assertEqual(agg_c["n_generated"], 1)
        self.assertEqual(agg_c["latency_mean_s"], 0.4)          # errors excluded from latency
        p = eval_route.paired(results, "local", "cloud")
        self.assertEqual(p["wins"], {"local": 1, "cloud": 1, "tie": 0, "unscored": 0})

    def test_answers_jsonl_shape_for_grade_only(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(make_gen(cloud_error_ids={"q2"}), answers_dir=d, out_stem="T")
            rows = eval_brain.load_answers(os.path.join(d, "T.cloud.jsonl"))
            self.assertEqual(set(rows), {"q1", "q2"})
            self.assertEqual(rows["q1"]["served_by"], "rag")
            self.assertEqual(rows["q1"]["backend"], "cloud")
            self.assertEqual(rows["q2"]["served_by"], "error")
            for k in ("id", "question", "answer", "sources", "served_by"):
                self.assertIn(k, rows["q1"])

    def test_budget_stops_before_next_question(self):
        clock = {"t": 1000.0}
        with mock.patch("eval_route.time.time", side_effect=lambda: clock.__setitem__("t", clock["t"] + 400) or clock["t"]):
            results, partial, _ = self._run(make_gen(), budget_s=600)
        self.assertTrue(partial)
        self.assertEqual({r["id"] for r in results}, {"q1"})

    def test_parse_judge(self):
        self.assertEqual(eval_route.parse_judge("groq:qwen/qwen3.8-27b"), ("groq", "qwen/qwen3.8-27b"))
        self.assertEqual(eval_route.parse_judge("ollama:qwen2.5:7b"), ("ollama", "qwen2.5:7b"))
        self.assertEqual(eval_route.parse_judge("none"), ("none", None))
        with self.assertRaises(ValueError):
            eval_route.parse_judge("bogus")

    def test_groq_chat_retries_429_then_returns_content(self):
        responses = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import urllib.error
        hdr = mock.Mock()
        hdr.get = lambda k: "1" if k == "retry-after" else None
        err429 = urllib.error.HTTPError("u", 429, "rate", hdr, io.BytesIO(b"{}"))
        ok = Resp(json.dumps({"choices": [{"message": {"content": '{"cover":[{"i":1,"present":true}]}'}}]}).encode())
        responses.extend([err429, ok])

        def fake_urlopen(req, timeout=None):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k", "BRAIN_SEED": "1"}), \
             mock.patch("eval_route.urllib.request.urlopen", fake_urlopen), \
             mock.patch("eval_route.time.sleep") as slp:
            out = eval_route._groq_chat("qwen/qwen3.8-27b", [{"role": "system", "content": "s"},
                                                            {"role": "user", "content": "u"}],
                                        "h", True, 10)
        self.assertIn('"present": true', out.replace('"present":true', '"present": true'))
        self.assertEqual(eval_route.RATE_LIMIT["hits"], 1)
        slp.assert_called_once_with(1.0)

    def test_groq_chat_requires_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(eval_brain.JudgeTransportError):
                eval_route._groq_chat("m", [], "h", True, 5)


class GenerateStrictTests(unittest.TestCase):
    """langchain_agent.generate(): strict mode must never reach the local model."""

    def setUp(self):
        # The Mac clone has no langchain installed; the Jetson does. Stub the
        # two message classes generate() imports so the test is about routing,
        # not about which machine runs it.
        import types
        if "langchain_core.messages" not in sys.modules:
            stub = types.ModuleType("langchain_core.messages")
            stub.HumanMessage = stub.SystemMessage = lambda content: content
            pkg = types.ModuleType("langchain_core")
            pkg.messages = stub
            self._stubbed = ("langchain_core", "langchain_core.messages")
            sys.modules["langchain_core"] = pkg
            sys.modules["langchain_core.messages"] = stub
        else:
            self._stubbed = ()
        import langchain_agent as la
        self.la = la
        self.msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)

    def _fake_cloud(self, contents):
        class R(object):
            def __init__(self, c):
                self.content = c
                self.response_metadata = {"finish_reason": "stop"}
                self.usage_metadata = {"output_tokens": 3}

        class M(object):
            def invoke(self, lc_in):
                return R(contents.pop(0))
        return lambda: M()

    def test_empty_then_text_counts_two_attempts(self):
        with mock.patch.object(self.la, "_LANGCHAIN_AVAILABLE", True), \
             mock.patch.object(self.la, "make_cloud_chat_model", self._fake_cloud(["", "  ", "ok"]), create=True), \
             mock.patch("brain.chat_once", side_effect=AssertionError("local must not run")):
            # two attempts max: "" then "  " -> both empty -> strict error
            r = self.la.generate(self.msgs, "cloud-speed", strict=True)
        self.assertEqual(r["served_by"], "cloud-error")
        self.assertEqual(r["attempts"], 2)
        self.assertIn("empty content", r["error"])
        self.assertEqual(r["text"], "")

    def test_exception_strict_vs_fallback(self):
        def boom():
            raise RuntimeError("groq down")
        with mock.patch.object(self.la, "make_cloud_chat_model", boom, create=True), \
             mock.patch("brain.chat_once", return_value="local answer") as co:
            strict = self.la.generate(self.msgs, "cloud-speed", strict=True)
            self.assertEqual(strict["served_by"], "cloud-error")
            self.assertIn("RuntimeError", strict["error"])
            co.assert_not_called()
            prod = self.la.generate(self.msgs, "cloud-speed", strict=False)
            self.assertEqual(prod["served_by"], "local-fast")
            self.assertEqual(prod["text"], "local answer")
            co.assert_called_once()

    def test_local_error_string_is_flagged(self):
        with mock.patch("brain.chat_once", return_value="[error] could not reach the model: x"):
            r = self.la.generate(self.msgs, "local-fast", strict=True)
        self.assertEqual(r["served_by"], "local-fast-error")
        self.assertIsNotNone(r["error"])

    def test_content_blocks_normalized(self):
        class R(object):
            content = [{"type": "reasoning", "text": "hmm"}, {"type": "text", "text": "final"}]
        self.assertEqual(self.la._content_text(R()), "final")




class RateLimitTests(unittest.TestCase):
    """generate(): a Groq 429 is waited out (Retry-After honored, capped) only
    when the caller allows it; production (0s) falls back at once."""

    def setUp(self):
        GenerateStrictTests.setUp(self)

    def tearDown(self):
        GenerateStrictTests.tearDown(self)

    def _limited_then_ok(self, n429):
        class RateLimitError(Exception):
            pass
        calls = {"n": 0}

        class R(object):
            content = "answer"
            response_metadata = {"finish_reason": "stop"}
            usage_metadata = {}

        class M(object):
            def invoke(self, lc_in):
                calls["n"] += 1
                if calls["n"] <= n429:
                    raise RateLimitError("Rate limit reached for model. Please try again in 2.5s.")
                return R()
        return (lambda: M()), calls

    def test_waits_when_allowed(self):
        mk, calls = self._limited_then_ok(2)
        with mock.patch.object(self.la, "make_cloud_chat_model", mk, create=True), \
             mock.patch("langchain_agent.time.sleep") as slp:
            r = self.la.generate(self.msgs, "cloud-speed", strict=True, rate_limit_wait_s=60)
        self.assertEqual(r["served_by"], "cloud")
        self.assertEqual(r["rate_limited"], 2)
        self.assertEqual(r["waited_s"], 5.0)
        self.assertEqual(calls["n"], 3)
        slp.assert_called_with(2.5)

    def test_production_does_not_wait(self):
        mk, calls = self._limited_then_ok(1)
        with mock.patch.object(self.la, "make_cloud_chat_model", mk, create=True), \
             mock.patch("langchain_agent.time.sleep") as slp, \
             mock.patch("brain.chat_once", return_value="local"):
            strict = self.la.generate(self.msgs, "cloud-speed", strict=True)
            self.assertEqual(strict["served_by"], "cloud-error")
            self.assertIn("RateLimitError", strict["error"])
        mk2, _ = self._limited_then_ok(1)   # fresh counter: the 429 must hit the prod call too
        with mock.patch.object(self.la, "make_cloud_chat_model", mk2, create=True), \
             mock.patch("langchain_agent.time.sleep") as slp, \
             mock.patch("brain.chat_once", return_value="local"):
            prod = self.la.generate(self.msgs, "cloud-speed", strict=False)
            self.assertEqual(prod["served_by"], "local-fast")
        slp.assert_not_called()

    def test_wait_is_capped(self):
        mk, calls = self._limited_then_ok(5)
        with mock.patch.object(self.la, "make_cloud_chat_model", mk, create=True), \
             mock.patch("langchain_agent.time.sleep"):
            r = self.la.generate(self.msgs, "cloud-speed", strict=True, rate_limit_wait_s=4)
        self.assertEqual(r["served_by"], "cloud-error")
        self.assertLessEqual(r["waited_s"], 4.0)




class RetryAfterParseTests(unittest.TestCase):
    def test_header_wins(self):
        h = mock.Mock(); h.get = lambda k: "3" if k == "retry-after" else None
        self.assertEqual(eval_route._retry_after(h, "try again in 9s"), 3.0)

    def test_body_phrase(self):
        h = mock.Mock(); h.get = lambda k: None
        self.assertEqual(eval_route._retry_after(h, 'Rate limit reached ... Please try again in 7.35s. Visit'), 7.6)
        self.assertEqual(eval_route._retry_after(h, "try again in 850ms"), 1.1)

    def test_default_and_cap(self):
        self.assertEqual(eval_route._retry_after(None, "", default=0.1), 1.0)
        self.assertEqual(eval_route._retry_after(None, "try again in 500s"), 90.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
