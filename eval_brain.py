#!/usr/bin/env python3
"""eval_brain.py - measure the Jetson brain's ANSWER ACCURACY, judged per question.

I am the scorekeeper for a brain that speaks about deaf lives, so I hold it to a
real standard, not a vibe. For each gold question I ask the brain exactly the way
a user would (cache first, then vault retrieval and the model), take the served
answer, and hand it to a local judge model that scores it 0 to 100 against a
FACT RUBRIC: the specific claims a correct, honest answer must cover. A question
PASSES at >= threshold (default 90). The whole run passes only when EVERY
question passes, because on this subject a wrong answer is not a rounding error.

Why a rubric and not string similarity: an answer can be worded a hundred ways
and still be right, and it can echo the reference wording and still be wrong. The
judge checks whether the covered FACTS are present and whether any claim is false
or audist, so the score tracks correctness, not phrasing.

Honest measurement note. The gold questions are HELD-OUT paraphrases, not the
phrasings prime_cache.py seeds, so a high score is not a tautology. If a held-out
paraphrase happens to fuzzy-match a primed entry, that is a real win: priming
covered it. If it does not, the vault retrieval and the model have to earn the
score live. Either way I judge the answer the user would actually receive.

This is the 4th script in the pipeline: build_index.py builds the vault index,
prime_cache.py warms the answer bank, brain.py answers, and I grade. Run me on the
Jetson where Ollama is local. Standard library only. Never crash. No em dashes.

Exit codes:
  0  every question scored >= threshold (or --dry-run listed the plan).
  2  usage error (bad flags or an empty gold set).
  3  import error: brain.py or answer_cache.py not importable beside me.
  4  environment error: index missing, or the judge model unreachable.
  5  at least one question scored below threshold (the loop should keep going).
  1  an unexpected error escaped the guard (should not happen).
"""

import os
import sys
import json
import time
import argparse
import urllib.request


# -- import the brain that lives beside me -----------------------------------
# Working directory is never assumed. I put my own folder first on sys.path so
# brain.py and answer_cache.py resolve no matter where python was launched.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import brain
except Exception as _e:
    sys.stderr.write(
        "I could not import brain.py, so there is nothing to grade.\n"
        "Looked in: " + SCRIPT_DIR + "\nReason: " + repr(_e) + "\n"
    )
    sys.exit(3)

try:
    from answer_cache import AnswerCache
except Exception:
    AnswerCache = None

# Footer marker plus its stripper, imported rather than restated, the same way
# brain.py does it. The judge must grade the ANSWER BODY: the citation footer
# is bare author/year/doi lines, not prose, so feeding it to the judge both
# dilutes the per-fact read and lets a listed title be scored as an ASSERTED
# must-not claim, which costs a full VIOLATION_PENALTY. If brain_depth is
# missing there is no footer to remove and grading proceeds unchanged.
try:
    from brain_depth import CITE_SENTINEL, strip_citations as _strip_citations
except Exception:      # brain_depth missing -> nothing to strip, grade as-is
    CITE_SENTINEL = None

    def _strip_citations(text):
        return text or ""


# -- config, mirrored from brain.py so the eval matches the deployed system ---
OLLAMA_HOST = getattr(brain, "OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = getattr(brain, "DEFAULT_MODEL", "llama3.2:3b")
# Judge on the SAME 3B model as the answerer. On the 8GB Orin Nano the 3B
# answerer and a 7B judge cannot co-reside, so a separate 7B judge forced a
# model swap on every --no-cache question: the dominant cost of a battery and
# the recurring OOM that killed Ollama mid-run. Judging on 3B keeps one model
# resident -- far faster, no swap, no OOM. The judge already grades one fact at
# a time via isolated yes/no calls (judge_isolated), the design that lets a
# small model grade reliably. Trade-off: answerer and judge are now the same
# model (self-grading), tolerable because grading is per-fact factual presence,
# not holistic quality. Pass --judge-model qwen2.5:7b to restore the 7B judge.
DEFAULT_JUDGE = DEFAULT_MODEL
CACHE_DIR = getattr(brain, "CACHE_DIR", "/mnt/nvme/brain/cache")
DEFAULT_GOLD = os.path.join(SCRIPT_DIR, "gold_questions.jsonl")
PASS_THRESHOLD = 90


# -- a silent Run so brain.handle_question answers without drawing a UI -------
class _SilentRun(brain._NullRun):
    """brain.handle_question calls run.* to render progress. I want only the
    returned answer text, so every stage and sink is a no-op. The answer comes
    from handle_question's RETURN value, not from what the sink prints."""

    def _emit(self, piece):
        return None

    def begin_answer(self):
        return self._emit

    def sources(self, *args, **kwargs):
        return None


# -- the brain under test -----------------------------------------------------
class Brain:
    """Loads the index once and answers a question the way the app does:
    cache first (unless disabled), then vault retrieval and the model."""

    def __init__(self, model, cache_dir, use_cache, web_enabled, semantic=None,
                 persona="default"):
        self.model = model
        self.web_enabled = web_enabled
        self.semantic = semantic  # optional SemanticCache, or None
        self.persona = persona    # "default" or "parent" (brain.PARENT_SYSTEM)
        # load_index() calls sys.exit() (SystemExit) when the index is missing
        # or corrupt. Catch that here so ready() can report False and main()
        # surfaces the documented exit-4 message instead of dying mid-init.
        try:
            self.chunks, self.bm25, self.meta = brain.load_index()
        except SystemExit:
            self.chunks, self.bm25, self.meta = [], None, {}
        self.cache = None
        if use_cache and AnswerCache is not None:
            try:
                self.cache = AnswerCache(cache_dir)
            except Exception:
                self.cache = None

    def ready(self):
        return bool(self.chunks) and self.bm25 is not None

    def ask(self, question):
        """Return (answer_text, sources_list, served_by). Never raises."""
        # Cache path, exactly as brain.main() serves it.
        if self.cache is not None:
            try:
                hit = self.cache.lookup(question)
            except Exception:
                hit = None
            if hit:
                ans = hit.get("answer", "") or ""
                if ans.strip():
                    return ans, (hit.get("sources") or []), "cache"
        # Semantic concept-router: widens coverage past lexical overlap. Only
        # fires on a confident, unambiguous match; otherwise falls through to RAG.
        if self.semantic is not None:
            try:
                m = self.semantic.match(question)
            except Exception:
                m = None
            if m and (m.get("answer") or "").strip():
                return m["answer"], (m.get("sources") or []), "semantic"
        # Fresh retrieval plus model.
        try:
            answer, sources = brain.handle_question(
                self.model, question, [], self.bm25, self.chunks,
                self.meta, self.web_enabled, _SilentRun(),
                persona=self.persona,
            )
            return (answer or ""), (sources or []), "rag"
        except Exception as e:
            return ("[error] " + repr(e)), [], "error"


# -- the judge ----------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a careful, fair grader of answers about deaf people, sign language, and "
    "language deprivation. You read the answer closely and check each fact one at a "
    "time. You credit a fact whenever its meaning is present in any wording. You flag a "
    "forbidden claim ONLY when the answer itself asserts it as true. You output only JSON."
)

# Per-fact grading. Asking the judge for one boolean per checklist item is far more
# reliable than a holistic 0-100: a small model under-credits present facts and
# invents "errors" when scoring holistically, but it reads accurately when forced to
# rule on each item. The numeric score is computed in Python from the booleans, so it
# is deterministic given the verdicts and cannot drift.
JUDGE_TEMPLATE = (
    "Grade the ANSWER against a checklist for this QUESTION.\n\n"
    "QUESTION:\n{question}\n\n"
    "MUST-COVER facts (numbered). For each, decide present=true if the answer conveys "
    "that fact in ANY wording. Paraphrase, synonyms, and meaning split across sentences "
    "ALL count as present. Exact words are NOT required.\n{must_cover}\n\n"
    "MUST-NOT claims (numbered). For each, decide asserted=true ONLY if the ANSWER "
    "ITSELF states this claim as true. If the answer refutes it, contradicts it, or does "
    "not mention it, asserted=false. Do NOT mark a claim asserted just because the answer "
    "discusses the same topic.\n{must_not}\n\n"
    "ANSWER TO GRADE:\n{answer}\n\n"
    "Respond with ONLY this JSON and nothing else:\n"
    '{{"cover": [{{"i": 1, "present": true}}, ...one per MUST-COVER fact...], '
    '"violate": [{{"j": 1, "asserted": false}}, ...one per MUST-NOT claim...]}}'
)

VIOLATION_PENALTY = 25   # points docked per must-not claim the answer actually asserts


# The judge must see the WHOLE answer it is grading. brain.py answers at
# num_ctx 6144 with num_predict 640, so a judge posted at 4096 silently drops
# the head of its own prompt -- the system prompt and the first facts of the
# checklist -- and grades a truncated view. Worse, no num_predict meant the
# verdict inherited Ollama's default 128 tokens, so a long MUST-COVER list got
# cut mid-JSON and parsed as a judge failure. Both defects punish LONG answers
# specifically, which is exactly what a run that got slower and more verbose
# produces: the harness would report a quality regression that is really its
# own truncation. Match the answerer's window; give the verdict room to close.
#
# DERIVED, not restated. This was hardcoded at 6144 and stayed there while the
# answerer moved 6144 -> 8192 -> 24576, so "match the answerer's window" quietly
# stopped being true and the judge graded a truncated view of exactly the long
# answers a good arm produces. Reading it from brain also keeps the judge on the
# SAME loaded model instance when it is the default self-judge: Ollama keys a
# loaded model by its options, so a different num_ctx forces a second load,
# which is how an 8 GB board ends up holding two copies of one 3B.
JUDGE_NUM_CTX = brain.GEN_OPTIONS["num_ctx"]
JUDGE_NUM_PREDICT = 512


# Every JudgeTransportError raised during a battery lands here, so a run that
# limped through a dead Ollama says so in its own report instead of looking
# like a quality result. `first` keeps the earliest message: the later ones are
# usually the same cause repeated once per retry.
TRANSPORT_ERRORS = {"count": 0, "first": None}


def _note_transport_error(exc):
    TRANSPORT_ERRORS["count"] += 1
    if TRANSPORT_ERRORS["first"] is None:
        TRANSPORT_ERRORS["first"] = str(exc)


class JudgeTransportError(Exception):
    """The judge could not be reached or did not return parseable transport
    payload. Distinct from a judge that answered with a bad verdict: this one
    means the measurement itself is void, not that the answer scored badly."""


def _ollama_chat(model, messages, host, want_json, timeout):
    """POST /api/chat to Ollama with urllib. Returns content str.

    Raises JudgeTransportError when Ollama is unreachable, the model is
    missing, or the response is not JSON. The previous bare
    `except Exception: return None` collapsed a dead Ollama, a 404 model, an
    OOM kill mid-generation and a malformed body into one indistinguishable
    None, which is how a battery with visibly correct answers was recorded as
    a clean zero. Callers that want the old tolerant behavior should catch
    JudgeTransportError explicitly and decide what it means."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": JUDGE_NUM_CTX,
            "num_predict": JUDGE_NUM_PREDICT,
        },
    }
    if want_json:
        body["format"] = "json"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        host + "/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise JudgeTransportError(
            "judge HTTP %s from %s: %s" % (e.code, host, detail)) from e
    except Exception as e:
        raise JudgeTransportError(
            "judge unreachable at %s: %r" % (host, e)) from e
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise JudgeTransportError(
            "judge returned non-JSON (%d bytes): %.200r" % (len(raw), raw)) from e
    return ((payload.get("message") or {}).get("content")) or ""


def _extract_json(text):
    """Pull the first balanced JSON object out of a model reply. None on fail."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _judge_once(prompt, judge_model, host, retries, timeout):
    """One judge sample. Returns (present:dict{i:bool}, asserted_set:set) or None.
    Retries only on a malformed / unreachable reply, not on a valid verdict."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for _attempt in range(max(1, retries)):
        try:
            content = _ollama_chat(judge_model, messages, host, True, timeout)
        except JudgeTransportError as e:
            _note_transport_error(e)
            continue
        obj = _extract_json(content)
        if not isinstance(obj, dict):
            continue
        cover = obj.get("cover")
        if not isinstance(cover, list) or not cover:
            continue
        present = {}
        for item in cover:
            if isinstance(item, dict) and "i" in item:
                try:
                    present[int(item["i"])] = bool(item.get("present"))
                except Exception:
                    continue
        if not present:
            continue
        asserted = set()
        for item in (obj.get("violate") or []):
            if isinstance(item, dict) and item.get("asserted"):
                try:
                    asserted.add(int(item.get("j")))
                except Exception:
                    continue
        return present, asserted
    return None


ISO_FACT_SYSTEM = (
    "You check whether one specific fact is conveyed by an answer. You credit it whenever "
    "its meaning is present in any wording, paraphrase, or synonym. You output only JSON."
)
ISO_FACT_TEMPLATE = (
    "ANSWER:\n{answer}\n\n"
    "FACT to look for:\n{fact}\n\n"
    "Does the ANSWER convey this fact, in any wording (paraphrase and synonyms count, exact "
    "words are NOT required)? Respond with ONLY this JSON: {{\"present\": true}} or "
    "{{\"present\": false}}."
)
ISO_VIOL_TEMPLATE = (
    "ANSWER:\n{answer}\n\n"
    "CLAIM:\n{claim}\n\n"
    "Does the ANSWER ITSELF assert this claim as true? If the answer refutes it, contradicts "
    "it, or does not state it, answer false. Respond with ONLY this JSON: {{\"asserted\": "
    "true}} or {{\"asserted\": false}}."
)


# Isolated-judge call tallies for the run. The conservative failure defaults
# below are correct per call but catastrophic in aggregate if the judge is
# simply DOWN: every fact defaults to "not present" and the run reports a clean,
# entirely believable 0/28 that looks like a system failure instead of a
# measurement failure. This happened for real. Answering pins llama3.2:3b with
# KEEP_ALIVE_FAST=-1, so on a 7GB Orin a --judge-model qwen2.5:7b cannot load
# afterwards; 165 judge calls errored out in 16 seconds total (~0.1s each, i.e.
# no inference at all) and the run scored 0/28 with every fact "missing" while
# the answers underneath were perfectly good. Counting outcomes lets main()
# tell the two apart and exit 4 (judge unreachable) instead of 5 (failed).
ISO_CALLS = {"ok": 0, "failed": 0}


def _iso_bool(template, key, filler, judge_model, host, retries, timeout, default):
    """Ask one isolated yes/no question and return the boolean. `default` on failure."""
    prompt = template.format(**filler)
    messages = [
        {"role": "system", "content": ISO_FACT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for _attempt in range(max(1, retries)):
        try:
            content = _ollama_chat(judge_model, messages, host, True, timeout)
        except JudgeTransportError as e:
            _note_transport_error(e)
            continue
        obj = _extract_json(content)
        if isinstance(obj, dict) and key in obj:
            ISO_CALLS["ok"] += 1
            return bool(obj.get(key))
    ISO_CALLS["failed"] += 1
    return default


def judge_isolated(question, answer, must_cover, must_not, judge_model, host, retries, timeout):
    """Grade each fact in its OWN focused call. A small judge checking one fact at a time is
    far more reliable than a 6-item checklist, where it loses track and drops present facts.
    A must-cover fact defaults to NOT present on judge failure (conservative); a must-not
    claim defaults to NOT asserted (do not invent violations). Returns the judge() tuple."""
    must_cover = list(must_cover or [])
    must_not = list(must_not or [])
    if not must_cover:
        # Not gradeable, and NOT a pass. This returned 100, which turned an
        # unlabeled row into a silent free win. load_gold drops such rows now,
        # so this is a backstop for any caller that builds rows itself.
        return -1, ["NO_MUST_COVER"], [], None
    missing = []
    ok = 0
    for fact in must_cover:
        present = _iso_bool(ISO_FACT_TEMPLATE, "present",
                            {"answer": answer or "(empty)", "fact": fact},
                            judge_model, host, retries, timeout, default=False)
        if present:
            ok += 1
        else:
            missing.append(fact)
    asserted = []
    for claim in must_not:
        if _iso_bool(ISO_VIOL_TEMPLATE, "asserted",
                     {"answer": answer or "(empty)", "claim": claim},
                     judge_model, host, retries, timeout, default=False):
            asserted.append(claim)
    score = int(round(100.0 * ok / len(must_cover))) - VIOLATION_PENALTY * len(asserted)
    score = max(0, min(100, score))
    return score, missing, asserted, "isolated"


def judge(question, answer, must_cover, must_not, judge_model, host, retries, timeout, votes=3):
    """Grade per-fact with MAJORITY-VOTE ensembling, compute the score in Python.
    Returns (score:int, missing:list, errors:list, raw). score -1 on total failure.

    A single small-judge sample under-credits present facts and occasionally invents a
    violation, and it is not deterministic across calls. So I sample the judge `votes`
    times and majority-vote each item: a MUST-COVER fact counts as present when more than
    half the valid samples mark it present, and a MUST-NOT claim counts as asserted only
    when a majority say so. This denoises both directions: a truly present fact (marked
    present most of the time) is credited, a truly absent one (marked absent every time)
    still fails. score = 100 * present/total - VIOLATION_PENALTY * asserted, clamped 0..100.
    """
    must_cover = list(must_cover or [])
    must_not = list(must_not or [])
    if not must_cover:
        # Same backstop as judge_isolated: no denominator means no score, not a
        # free 100. See load_gold.
        return -1, ["NO_MUST_COVER"], [], None
    cover_lines = "\n".join(str(i + 1) + ". " + f for i, f in enumerate(must_cover))
    not_lines = "\n".join(str(j + 1) + ". " + f for j, f in enumerate(must_not)) or "(none)"
    prompt = JUDGE_TEMPLATE.format(
        question=question, must_cover=cover_lines,
        must_not=not_lines, answer=(answer or "(empty)"),
    )

    n = len(must_cover)
    present_votes = {i: 0 for i in range(1, n + 1)}
    assert_votes = {j: 0 for j in range(1, len(must_not) + 1)}
    valid = 0
    for _v in range(max(1, votes)):
        res = _judge_once(prompt, judge_model, host, retries, timeout)
        if res is None:
            continue
        present, asserted = res
        valid += 1
        for i in range(1, n + 1):
            if present.get(i, False):
                present_votes[i] += 1
        for j in asserted:
            if j in assert_votes:
                assert_votes[j] += 1
    if valid == 0:
        return -1, ["JUDGE_UNREACHABLE"], [], None

    majority = valid / 2.0
    missing = [must_cover[i - 1] for i in range(1, n + 1) if present_votes[i] <= majority]
    asserted = [must_not[j - 1] for j in range(1, len(must_not) + 1) if assert_votes[j] > majority]
    n_present = n - len(missing)
    score = int(round(100.0 * n_present / n)) - VIOLATION_PENALTY * len(asserted)
    score = max(0, min(100, score))
    return score, missing, asserted, "votes=%d/%d" % (valid, votes)


# -- gold loading -------------------------------------------------------------
def report_gaps(results, the_brain):
    """For every must_cover fact the judge scored MISSING, count how many index
    chunks contain its distinctive terms.

    A failing question has two very different causes that look identical in the
    score: the retriever did not find the fact, or the fact is not in the vault
    at all. Only the first is fixable by tuning. "Who founded Gallaudet
    University and what law made it possible?" cannot be answered at any
    retrieval quality, because `Amos Kendall` and `Lincoln` appear in zero
    chunks. This separates the two so tuning effort goes where it can pay.

    Returns a list of {id, fact, terms, chunks}. Never raises."""
    out = []
    try:
        chunks = getattr(the_brain, "chunks", None) or []
        # Distinctive terms only: a fact's rare words decide whether it is in
        # the corpus. Scoring on common words would report every fact covered.
        doc_freq = {}
        tokenized = []
        for c in chunks:
            toks = set(brain.tokenize(c.get("text") or ""))
            tokenized.append(toks)
            for t in toks:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        n = max(len(chunks), 1)
        for r in results:
            for fact in r.get("missing") or []:
                terms = [t for t in brain.query_tokens(fact)
                         if doc_freq.get(t, 0) <= n * 0.10]
                terms = sorted(terms, key=lambda t: doc_freq.get(t, 0))[:4]
                if not terms:
                    continue
                hits = sum(1 for toks in tokenized
                           if all(t in toks for t in terms))
                out.append({"id": r["id"], "fact": fact,
                            "terms": terms, "chunks": hits})
    except Exception:
        return out
    return out


def load_gold(path):
    """Read any gold .jsonl (gold_questions, factual_gold, multihop_gold, ...)
    into a list of dicts. Skips blank/comment lines and malformed rows (with a
    stderr note). Returns [] if nothing loads. Never raises.

    A row with NO must_cover fact is dropped, not loaded. The score is
    100 * present / len(must_cover), so with no denominator the judges used to
    short-circuit to a free 100, and a whole file of such rows scored a constant
    100 while calling the judge zero times. retrieval_gold.jsonl (40 rows) and
    depth_retrieval_gold.jsonl (8 rows) are exactly that: expect_sources
    retrieval labels that belong to eval_retrieval.py, and their own header
    says so. Pointed at either, eval_brain now loads nothing and main() exits 2
    on the empty gold set rather than reporting a perfect score."""
    rows = []
    ungradeable = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    sys.stderr.write("skipping malformed gold line " + str(n) + "\n")
                    continue
                if not isinstance(obj, dict) or not obj.get("question"):
                    continue
                obj.setdefault("id", "q" + str(n))
                obj.setdefault("must_cover", [])
                obj.setdefault("must_not", [])
                if not obj["must_cover"]:
                    ungradeable.append(obj["id"])
                    continue
                rows.append(obj)
    except Exception as e:
        sys.stderr.write("could not read gold set: " + repr(e) + "\n")
    if ungradeable:
        shown = ", ".join(ungradeable[:10])
        if len(ungradeable) > 10:
            shown += ", ... (" + str(len(ungradeable) - 10) + " more)"
        sys.stderr.write(
            str(len(ungradeable)) + " row(s) in " + path + " carry no must_cover "
            "facts and were NOT graded: " + shown + "\n"
            "  A must_cover list is the only thing a score can be computed from.\n"
            "  If this is a retrieval label file (expect_sources), grade it with\n"
            "  eval_retrieval.py instead.\n")
    return rows


def load_answers(path):
    """Read a saved --answers-out JSONL back into {id: row}.

    Each row carries exactly id, question, answer, sources, served_by, which is
    everything the judge and the provenance gate need and nothing the answerer
    would have to be alive to supply. Tolerant like load_gold: blanks and
    comments skipped, malformed rows noted on stderr. Never raises."""
    rows = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    sys.stderr.write("skipping malformed answer line " + str(n) + "\n")
                    continue
                if not isinstance(obj, dict) or not obj.get("id"):
                    continue
                rows[obj["id"]] = obj
    except Exception as e:
        sys.stderr.write("could not read --grade-only file: " + repr(e) + "\n")
    return rows


def preflight_judge(judge_model, host, timeout):
    """Confirm the judge model answers at all. Returns True/False.

    Prints the transport reason on failure. The old version could only say
    "no reply", so a missing model, a stopped Ollama and a timeout all looked
    the same at the one moment the run could still be fixed cheaply."""
    try:
        reply = _ollama_chat(
            judge_model,
            [{"role": "user", "content": "Reply with the single word: ready"}],
            host, False, timeout,
        )
    except JudgeTransportError as e:
        sys.stderr.write("judge preflight failed: %s\n" % e)
        return False
    return bool(reply and reply.strip())


# -- reporting ----------------------------------------------------------------
def print_report(results, threshold, elapsed, gold_path, model, judge_model):
    line = "=" * 78
    print(line)
    print("BRAIN ANSWER-ACCURACY EVAL")
    print("  gold set  : " + gold_path)
    print("  answerer  : " + model + "   judge: " + judge_model)
    print("  threshold : every question must score >= " + str(threshold))
    print(line)
    passed = [r for r in results if r["score"] >= threshold]
    failed = [r for r in results if 0 <= r["score"] < threshold]
    broke = [r for r in results if r["score"] < 0]
    results_sorted = sorted(results, key=lambda r: (r["score"] if r["score"] >= 0 else -1))
    print("PER QUESTION (weakest first):")
    for r in results_sorted:
        tag = "PASS" if r["score"] >= threshold else ("JUDGE?" if r["score"] < 0 else "FAIL")
        sc = ("  n/a" if r["score"] < 0 else ("%5d" % r["score"]))
        print("  [%s] %s  via %-5s  %s" % (tag, sc, r["served_by"], r["id"]))
        print("         Q: " + r["question"][:70])
        if r["score"] < threshold:
            if r.get("missing"):
                print("         missing: " + "; ".join(str(m) for m in r["missing"][:4]))
            if r.get("errors"):
                print("         ERRORS : " + "; ".join(str(e) for e in r["errors"][:4]))
    print(line)
    n = len(results)
    worst = min((r["score"] for r in results if r["score"] >= 0), default=0)
    mean = (sum(r["score"] for r in results if r["score"] >= 0) /
            max(1, len([r for r in results if r["score"] >= 0])))
    print("SUMMARY")
    print("  questions      : " + str(n))
    print("  passed         : " + str(len(passed)) + " / " + str(n))
    print("  failed         : " + str(len(failed)))
    # A -1 has two causes now and they need different fixes: JUDGE_UNREACHABLE
    # is an environment problem (restart Ollama), NO_MUST_COVER is a data defect
    # in the gold file (nothing to grade against). Reporting both as "judge
    # unreachable" sent the operator to restart a service over a bad row.
    ungrad = [r for r in broke if "NO_MUST_COVER" in (r.get("errors") or [])]
    envfail = [r for r in broke if r not in ungrad]
    if envfail:
        print("  judge failures : " + str(len(envfail)) + " (judge unreachable, treat as env error)")
    if ungrad:
        print("  ungradeable    : " + str(len(ungrad)) + " (no must_cover facts; a data defect, not a judge problem)")
    print("  lowest score   : " + str(worst))
    print("  mean score     : %.1f" % mean)
    print("  elapsed        : %.1fs" % elapsed)
    verdict = "ALL PASS" if (not failed and not broke) else "NOT YET"
    print("  verdict        : " + verdict + " (every question >= " + str(threshold) + ")")
    print(line)
    if failed:
        print("Keep looping. For each FAIL: strengthen the vault note, or seed the")
        print("phrasing and its grounded answer into prime_cache.py, then re-run the pipeline.")
    else:
        print("Every question clears the bar. The brain answers these with dignity and fact.")


def provenance_failure(args, results):
    """True when --require-rag is on and something was not served by retrieval.

    Same semantics whichever path produced the results: served_by is stored in
    the --answers-out JSONL precisely so a re-grade cannot quietly launder a
    cache hit into a retrieval score."""
    if not args.require_rag:
        return False
    non_rag = [r for r in results if r.get("served_by") != "rag"]
    if not non_rag:
        return False
    sys.stderr.write(
        "PROVENANCE FAILURE: %d of %d answers were not served by retrieval.\n"
        % (len(non_rag), len(results)))
    for r in non_rag[:10]:
        sys.stderr.write("  %-30s served_by=%s\n" % (r["id"], r.get("served_by")))
    sys.stderr.write(
        "  These scores do not measure the retrieval pipeline. Pass "
        "--allow-cache-hits to grade them anyway.\n")
    return True


# -- main ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="eval_brain.py",
        description="LLM-judged answer-accuracy eval for the Jetson brain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.find("Exit codes:"):],
    )
    p.add_argument("--gold", default=DEFAULT_GOLD, help="gold_questions.jsonl path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="answerer model (brain)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE, help="judge model")
    p.add_argument("--threshold", type=int, default=PASS_THRESHOLD, help="pass score")
    p.add_argument("--cache-dir", default=CACHE_DIR, help="answer cache dir")
    p.add_argument("--no-cache", action="store_true", help="force fresh RAG, ignore cache")
    p.add_argument("--two-phase", action="store_true",
                   help="answer all, then judge all (one model load per phase; fast on 8GB)")
    p.add_argument("--report-gaps", action="store_true",
                   help="for each missed fact, count vault chunks containing it "
                        "(separates content gaps from retrieval gaps)")
    p.add_argument("--semantic", action="store_true",
                   help="enable the embedding concept-router (widens coverage past lexical match)")
    p.add_argument("--embed-model", default="nomic-embed-text", help="embedding model for --semantic")
    p.add_argument("--sem-threshold", type=float, default=0.70, help="min top cosine to serve semantically")
    p.add_argument("--sem-margin", type=float, default=0.06, help="min top-over-2nd margin to serve semantically")
    p.add_argument("--web", action="store_true", help="allow web fallback (default off)")
    p.add_argument("--host", default=OLLAMA_HOST, help="Ollama host")
    p.add_argument("--only", action="append", default=None, metavar="ID",
                   help="grade only this question id (repeatable)")
    p.add_argument("--retries", type=int, default=2, help="judge retries on a malformed reply")
    p.add_argument("--judge-votes", type=int, default=3,
                   help="judge samples per question, majority-voted (only used with --batch)")
    # Per-fact isolated grading is the DEFAULT: on the 8GB board's 3B self-judge the
    # batched checklist judge() goes negation-blind (false-positive violations) and
    # drops verbatim-present facts (false-negative coverage), so identical answers
    # score differently. Isolating one fact per call fixed a whole gold set from
    # 17/28 to 28/28 (mean 100) AND ran faster (no vote-ensembling). Use --batch to
    # restore the old majority-vote judge (only sensible with a stronger --judge-model).
    p.add_argument("--isolate", dest="isolate", action="store_true",
                   help="grade each fact in its own focused call (DEFAULT; most reliable for a small judge)")
    p.add_argument("--batch", dest="isolate", action="store_false",
                   help="use the batched majority-vote judge (--judge-votes) instead of per-fact isolation")
    p.set_defaults(isolate=True)
    # A question served from the answer cache or the semantic router is graded
    # exactly like a retrieved one, and nothing ever failed the run over it.
    # That is how a discredited "28/28" was produced: 100% cache hits, the
    # retrieval pipeline never executed. Provenance is now enforced by default,
    # and the configuration is rejected BEFORE answering rather than after the
    # 900s the battery takes to discover it.
    p.add_argument("--require-rag", dest="require_rag", action="store_true",
                   help="fail unless every answer is served by retrieval (DEFAULT)")
    p.add_argument("--allow-cache-hits", dest="require_rag", action="store_false",
                   help="permit cache/semantic-served answers to be graded")
    p.set_defaults(require_rag=True)
    p.add_argument("--answers-out", default=None, metavar="PATH",
                   help="write the answers as JSONL so a re-grade never has to "
                        "re-generate (a judged battery costs ~900-1700s)")
    # The other half of --answers-out, which was write-only until now: nothing
    # read those files back, so every rubric edit re-paid the full generation
    # battery. --grade-only re-judges a saved battery with no Brain, no
    # load_index and no answerer, which also means it runs on a laptop that has
    # neither the models nor the vault index.
    p.add_argument("--grade-only", default=None, metavar="PATH",
                   help="re-judge a saved --answers-out JSONL instead of generating "
                        "(skips the index and the answerer entirely)")
    p.add_argument("--timeout", type=int, default=180, help="per-call timeout seconds")
    p.add_argument("--dry-run", action="store_true", help="list the plan, grade nothing")
    p.add_argument("--json", dest="as_json", action="store_true", help="emit JSON report")
    p.add_argument("--mode", choices=("default", "parent", "scholar"), default="default",
                   help="answer persona: parent = warm parent-guidance register, "
                        "scholar = citation-bearing academic register")
    return p


def env_fingerprint(args):
    """Record which arm produced this artifact.

    eval_retrieval.py has carried one of these for a while; the judged harness
    did not, so its artifacts recorded a score with no way to tell what code
    path earned it. That is how AFTER_gold_questions_15of28.json ended up being
    read as a baseline for a retriever it never ran against.

    Every value is read off the BRAIN MODULE rather than re-derived from the
    environment with a default restated here. Restating is how an artifact
    mislabels its own arm: BRAIN_DECOMPOSE's default flipped 1->0 and
    BRAIN_HYDE's flipped 0->1, and any copy of the old default records an arm
    the run never used. brain's constants resolve at import time, so these are
    what the code WILL DO -- the only thing worth fingerprinting.

    The generation knobs matter here and nowhere else: eval_retrieval.py stops
    at retrieval, so num_ctx, num_predict and the context-fit budget are
    invisible to it. They are exactly the arms a judged run exists to price.

    Persona and seed are recorded because they were not: a --mode default run
    and a --mode parent run emitted BYTE-IDENTICAL headers, so no artifact
    already sitting in eval_results/ can be attributed to a persona at all.
    The persona is the resolved args.mode, i.e. what Brain was actually
    constructed with, not an env default restated here.

    The BRAIN_SCHOLAR_* flags are resolved off the module like every other arm.
    They were env-only when brain.py had not defined them; brain.py now does
    (SCHOLAR_PASS2/CITE/AUTO at 284-289, SCHOLAR_TOP_K/MAX_PER_SOURCE/
    FACETS_N at 253/257/260), and BRAIN_SCHOLAR_CITE defaults to "1", so an
    environ read of an unset variable would record null for the SHIPPED arm --
    a null that is byte-identical to the null a run that never touched the flag
    emits, and that reads as OFF next to the explicit "0" of a CITE-off arm.

    env_explicit lists the fingerprinted variables actually present in the
    process environment, so "unset, defaulted ON" stays distinguishable from
    "explicitly set to 1" without duplicating every key.
    """
    def flag(v):
        return "1" if v else "0"

    mod = globals().get("brain")

    def resolved(attr, env_name, fmt=flag):
        """The value the code WILL USE: brain's resolved constant when brain.py
        defines it, the raw environment only as a fallback.

        The fallback keeps this working if brain.py is older than the flag, or
        if brain could not be imported far enough to define it. The test is
        `is None`, not truthiness, precisely so a legitimate False/0 records as
        "0" instead of silently falling through to the environment."""
        v = getattr(mod, attr, None) if mod is not None else None
        if v is None:
            return os.environ.get(env_name)
        return fmt(v)

    gen = getattr(brain, "GEN_OPTIONS", {}) or {}
    fp = {
        "index": os.environ.get("BRAIN_INDEX", brain.INDEX_PATH),
        "persona": args.mode,
        "seed": os.environ.get("BRAIN_SEED"),
        # Retrieval arms.
        "BRAIN_HYBRID": flag(brain.HYBRID_ENABLED),
        "BRAIN_HYDE": flag(brain.HYDE_ENABLED),
        "BRAIN_DECOMPOSE": flag(brain.DECOMPOSE_ENABLED),
        "BRAIN_TOP_K_FAST": str(brain.TOP_K_FAST),
        "BRAIN_ANN": flag(brain.ANN_ENABLED),
        "BRAIN_OBSIDIAN_RANK": flag(brain.OBSIDIAN_RANK_ENABLED),
        "BRAIN_OBSIDIAN_BLOCKS": flag(brain.OBSIDIAN_BLOCKS_ENABLED),
        "BRAIN_SKILLS": flag(getattr(brain, "SKILLS_ENABLED", False)),
        "BRAIN_AUTOLEARN": flag(brain.AUTOLEARN_ENABLED),
        # Scholar arms, resolved through brain.py's defaults. CITE defaults ON,
        # so a raw environ read of an unset variable records null and loses the
        # shipped arm. Note SCHOLAR_FACETS_N: the env var is BRAIN_SCHOLAR_FACETS
        # but the module constant carries the _N suffix, and a wrong attribute
        # name here would getattr to None and revive the bug as a fallback.
        "BRAIN_SCHOLAR_PASS2": resolved("SCHOLAR_PASS2", "BRAIN_SCHOLAR_PASS2"),
        "BRAIN_SCHOLAR_CITE": resolved("SCHOLAR_CITE", "BRAIN_SCHOLAR_CITE"),
        "BRAIN_SCHOLAR_AUTO": resolved("SCHOLAR_AUTO", "BRAIN_SCHOLAR_AUTO"),
        "BRAIN_SCHOLAR_TOP_K": resolved("SCHOLAR_TOP_K", "BRAIN_SCHOLAR_TOP_K", str),
        "BRAIN_SCHOLAR_MAX_PER_SOURCE": resolved(
            "SCHOLAR_MAX_PER_SOURCE", "BRAIN_SCHOLAR_MAX_PER_SOURCE", str),
        "BRAIN_SCHOLAR_FACETS": resolved(
            "SCHOLAR_FACETS_N", "BRAIN_SCHOLAR_FACETS", str),
        # Generation arms -- judged-only territory.
        "num_ctx": gen.get("num_ctx"),
        "num_predict": gen.get("num_predict"),
        "temperature": gen.get("temperature"),
        "BRAIN_CTX_FIT": flag(brain.CTX_FIT_ENABLED),
        "CTX_RESERVE_TOKENS": getattr(brain, "CTX_RESERVE_TOKENS", None),
        "CHARS_PER_TOKEN": getattr(brain, "CHARS_PER_TOKEN", None),
    }
    # Which of the above the operator actually exported. Everything else in this
    # dict is now an EFFECTIVE value, so without this an arm that relied on the
    # default is indistinguishable from one that typed the default out.
    fp["env_explicit"] = sorted(
        k for k in list(fp) + ["BRAIN_INDEX", "BRAIN_SEED"]
        if k.startswith("BRAIN_") and k in os.environ)
    return fp


def grade_only(args, gold, run_judge, start):
    """Re-judge a saved --answers-out battery. No Brain, no load_index, no
    answerer model: only the judge runs.

    This is the point of the flag. Generating a judged battery costs 900-1700s,
    so every rubric edit used to re-pay for answers that had not changed, and
    the arms could only be re-scored on the Jetson itself. Reading the JSONL
    back means a rubric change costs judge time alone, on any machine.

    The --require-rag PRE-gate is skipped here on purpose: it inspects live
    cache and semantic-router state that does not exist in this mode. The
    POST-gate still runs, on the served_by recorded when the answers were
    generated, so provenance is enforced exactly as strictly as before."""
    answers = load_answers(args.grade_only)
    if not answers:
        sys.stderr.write("no answers to re-grade in " + args.grade_only + "\n")
        return 2

    if not preflight_judge(args.judge_model, args.host, min(60, args.timeout)):
        sys.stderr.write(
            "judge model " + args.judge_model + " is not answering at " + args.host
            + ". Is Ollama running and the model pulled?\n")
        return 4

    if not args.as_json:
        print("Grade-only: re-judging %d saved answer(s) from %s (no index, no answerer)."
              % (len(answers), args.grade_only))

    results = []
    unanswered = []
    for i, g in enumerate(gold, 1):
        row = answers.get(g["id"])
        if row is None:
            # A gold row with no saved answer is a HOLE IN THE ARTIFACT, not a
            # wrong answer. Scoring it 0 would drag the mean down and read as a
            # quality regression, so it is reported and left out of the scores.
            unanswered.append(g["id"])
            continue
        answer = row.get("answer") or ""
        sources = list(row.get("sources") or [])
        served_by = row.get("served_by") or "unknown"
        score, missing, errors, _raw = run_judge(g, answer)
        results.append({
            "id": g["id"], "question": g["question"], "score": score,
            "served_by": served_by, "missing": missing, "errors": errors,
            "answer_preview": answer[:200], "n_sources": len(sources),
            "sources": sources,
            # The saved JSONL carries no timings: the answer was generated in
            # some other run. None says "not measured here" rather than 0.0,
            # which would read as an impossibly fast answer.
            "answer_seconds": None,
            "timing": {},
        })
        if not args.as_json:
            tag = "PASS" if score >= args.threshold else ("JUDGE?" if score < 0 else "FAIL")
            print("  [%2d/%2d] %-6s %4s  %s" % (
                i, len(gold), tag, ("n/a" if score < 0 else str(score)), g["id"]))

    if unanswered:
        sys.stderr.write(
            "%d gold question(s) have no answer in %s and were NOT scored: %s\n"
            % (len(unanswered), args.grade_only, ", ".join(unanswered[:10])))
    if not results:
        sys.stderr.write("no gold id matched any saved answer; nothing to grade\n")
        return 2

    if provenance_failure(args, results):
        return 4

    # --report-gaps needs the vault index to count chunks, and this mode
    # deliberately never loads one. Say so instead of silently reporting none.
    if args.report_gaps and not args.as_json:
        print("(--report-gaps needs the vault index, which --grade-only does not load)")

    elapsed = time.time() - start
    return emit_and_verdict(args, results, [], elapsed)


def main():
    start = time.time()
    args = build_parser().parse_args()

    gold = load_gold(args.gold)
    if args.only:
        want = set(args.only)
        gold = [g for g in gold if g["id"] in want]
    if not gold:
        sys.stderr.write("no gold questions to grade (check --gold / --only)\n")
        return 2

    if args.dry_run:
        print("Dry run. Would grade " + str(len(gold)) + " question(s) with judge "
              + args.judge_model + " at threshold " + str(args.threshold) + ":")
        for g in gold:
            print("  - " + g["id"] + ": " + g["question"][:72])
        return 0

    # Defined before the brain is touched so --grade-only can borrow it: the
    # judging rules must be one piece of code, or a re-grade stops being
    # comparable to the run it re-grades.
    def run_judge(g, answer):
        # Grade the body only. The citation footer that brain.py appends when
        # BRAIN_SCHOLAR_CITE is on is metadata, and grading it penalises every
        # CITE-on arm against an otherwise byte-identical CITE-off arm. This is
        # the one chokepoint every path funnels through (live single-pass, live
        # two-phase, --grade-only), and it is deliberately NOT applied at the
        # ask() call sites: the saved answers file must keep its footer so a
        # re-grade and the footer-quality checks stay possible.
        answer = _strip_citations(answer)
        if args.isolate:
            return judge_isolated(
                g["question"], answer, g.get("must_cover", []), g.get("must_not", []),
                args.judge_model, args.host, args.retries, args.timeout)
        return judge(
            g["question"], answer, g.get("must_cover", []), g.get("must_not", []),
            args.judge_model, args.host, args.retries, args.timeout, args.judge_votes)

    # Re-grade a saved battery: no index, no answerer, no models beyond the
    # judge. Returns before any of the setup below runs.
    if args.grade_only:
        return grade_only(args, gold, run_judge, start)

    # Optional semantic concept-router (widens coverage past lexical overlap).
    semantic = None
    if args.semantic:
        try:
            from semantic_cache import SemanticCache
            import prime_cache
            semantic = SemanticCache(
                prime_cache.KNOWLEDGE, host=args.host, model=args.embed_model,
                threshold=args.sem_threshold, margin=args.sem_margin,
                cache_path=os.path.join(SCRIPT_DIR, ".semantic_vecs.json"))
            if not semantic.ready:
                sys.stderr.write("semantic router unavailable (embeddings failed); using RAG only\n")
                semantic = None
            elif not args.as_json:
                print("semantic router ready: %d concepts, threshold %.2f, margin %.2f"
                      % (len(semantic.ids), args.sem_threshold, args.sem_margin))
        except Exception as e:
            sys.stderr.write("could not build semantic router: " + repr(e) + "\n")
            semantic = None

    # Load the brain (index must exist).
    the_brain = Brain(args.model, args.cache_dir, not args.no_cache, args.web, semantic,
                      persona=args.mode)
    if not the_brain.ready():
        sys.stderr.write(
            "the vault index is not loaded. Run build_index.py first "
            "(expected at " + getattr(brain, "INDEX_PATH", "index/bm25.pkl") + ")\n")
        return 4

    # Provenance gate, checked HERE -- before a single question is answered --
    # because the failure it prevents is a whole battery whose numbers describe
    # the cache rather than the system.
    if args.require_rag:
        offenders = []
        if not args.no_cache:
            offenders.append("the answer cache is enabled (pass --no-cache)")
        if semantic is not None:
            offenders.append("the semantic router is enabled (drop --semantic)")
        if offenders:
            sys.stderr.write(
                "--require-rag is on (the default) but " + " and ".join(offenders)
                + ".\n  Answers could be served without ever running retrieval, "
                "which is how a\n  100%-cache-hit run was once recorded as a "
                "perfect score. Fix the flags,\n  or pass --allow-cache-hits to "
                "grade cache-served answers on purpose.\n")
            return 4

    # Confirm the judge is reachable before spending time answering.
    if not preflight_judge(args.judge_model, args.host, min(60, args.timeout)):
        sys.stderr.write(
            "judge model " + args.judge_model + " is not answering at " + args.host
            + ". Is Ollama running and the model pulled?\n")
        return 4

    if not args.as_json:
        print("Grading " + str(len(gold)) + " question(s). Answer then judge, one at a time.")

    results = []
    # Full answer text, kept separately because results[] stores only a 200-char
    # preview. This is what --answers-out writes so a judge change can be
    # re-scored against the SAME answers instead of regenerating them.
    answers_log = []
    if args.two_phase:
        # Phase 1: answer everything (answerer model stays resident), then Phase 2:
        # judge everything (judge model stays resident). One model load per phase
        # instead of a 3B<->judge swap on every question, which on an 8GB board that
        # cannot hold both models is the dominant cost of a --no-cache run.
        if not args.as_json:
            print("Two-phase: answering all, then judging all (avoids model swapping).")
        answered = []
        for i, g in enumerate(gold, 1):
            t0 = time.time()
            answer, sources, served_by = the_brain.ask(g["question"])
            ask_s = time.time() - t0
            timing = dict(getattr(brain, "LAST_TIMING", None) or {}) \
                if served_by == "rag" else {}
            answered.append((g, answer, sources, served_by, ask_s, timing))
            if not args.as_json:
                print("  answered [%2d/%2d] %-8s %s" % (i, len(gold), served_by, g["id"]))
        for i, (g, answer, sources, served_by, ask_s, timing) in enumerate(answered, 1):
            score, missing, errors, _raw = run_judge(g, answer)
            results.append({
                "id": g["id"], "question": g["question"], "score": score,
                "served_by": served_by, "missing": missing, "errors": errors,
                "answer_preview": (answer or "")[:200], "n_sources": len(sources or []),
                # Deterministic regression signals: WHICH sources were retrieved
                # (diffable across configs, unlike judge scores) and how long the
                # answer took, with brain.py's per-stage breakdown when RAG ran.
                "sources": list(sources or []),
                "answer_seconds": round(ask_s, 3),
                "timing": timing,
            })
            answers_log.append({
                "id": g["id"], "question": g["question"], "answer": answer or "",
                "sources": list(sources or []), "served_by": served_by,
            })
            if not args.as_json:
                tag = "PASS" if score >= args.threshold else ("JUDGE?" if score < 0 else "FAIL")
                print("  judged [%2d/%2d] %-6s %4s  %s" % (
                    i, len(gold), tag, ("n/a" if score < 0 else str(score)), g["id"]))
    else:
        for i, g in enumerate(gold, 1):
            t0 = time.time()
            answer, sources, served_by = the_brain.ask(g["question"])
            ask_s = time.time() - t0
            timing = dict(getattr(brain, "LAST_TIMING", None) or {}) \
                if served_by == "rag" else {}
            score, missing, errors, _raw = run_judge(g, answer)
            results.append({
                "id": g["id"], "question": g["question"], "score": score,
                "served_by": served_by, "missing": missing, "errors": errors,
                "answer_preview": (answer or "")[:200], "n_sources": len(sources or []),
                "sources": list(sources or []),
                "answer_seconds": round(ask_s, 3),
                "timing": timing,
            })
            answers_log.append({
                "id": g["id"], "question": g["question"], "answer": answer or "",
                "sources": list(sources or []), "served_by": served_by,
            })
            if not args.as_json:
                tag = "PASS" if score >= args.threshold else ("JUDGE?" if score < 0 else "FAIL")
                print("  [%2d/%2d] %-6s %4s  %s" % (
                    i, len(gold), tag, ("n/a" if score < 0 else str(score)), g["id"]))

    elapsed = time.time() - start

    # Write answers BEFORE any early return below, so an expensive battery that
    # trips a guard is still re-gradable without regenerating it.
    if args.answers_out:
        try:
            with open(args.answers_out, "w", encoding="utf-8") as fh:
                for row in answers_log:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not args.as_json:
                print("wrote %d answer(s) to %s" % (len(answers_log), args.answers_out))
        except Exception as e:
            sys.stderr.write("could not write --answers-out: " + repr(e) + "\n")

    # The configuration gate above rejects flags that PERMIT a cache hit; this
    # one catches an answer that actually came from somewhere other than
    # retrieval, whatever the flags claimed.
    if provenance_failure(args, results):
        return 4

    gaps = report_gaps(results, the_brain) if args.report_gaps else []
    if gaps and not args.as_json:
        print("")
        print("Vault coverage of missed facts (%d checked)" % len(gaps))
        print("-" * 72)
        uncovered = [g for g in gaps if g["chunks"] == 0]
        thin = [g for g in gaps if 0 < g["chunks"] <= 2]
        if uncovered:
            print("NOT IN THE VAULT -- no retrieval change can fix these:")
            for g in uncovered:
                print("  [%s] %s" % (g["id"], g["fact"][:88]))
                print("      0 chunks contain: %s" % ", ".join(g["terms"]))
        if thin:
            print("THIN (1-2 chunks) -- present but easy to miss:")
            for g in thin:
                print("  [%s] %-64s %d chunk(s)" % (g["id"], g["fact"][:64], g["chunks"]))
        if not uncovered and not thin:
            print("  Every missed fact is well covered in the vault; these are")
            print("  retrieval or generation failures, not content gaps.")

    return emit_and_verdict(args, results, gaps, elapsed)


def emit_and_verdict(args, results, gaps, elapsed):
    """Print the report (JSON or human) and return the process exit code.

    Shared by the generating run and by --grade-only so both emit the SAME
    artifact shape: an arm re-graded from saved answers has to be diffable
    against an arm that generated them, and it is not if the two paths write
    different keys. The guard sequence below is unchanged; it lives here so
    there is exactly one copy of it."""
    failed = [r for r in results if r["score"] < args.threshold]
    if args.as_json:
        print(json.dumps({
            "gold": args.gold, "model": args.model, "judge": args.judge_model,
            "env": env_fingerprint(args),
            "threshold": args.threshold, "results": results, "gaps": gaps,
            "passed": len([r for r in results if r["score"] >= args.threshold]),
            "failed": len([r for r in results if 0 <= r["score"] < args.threshold]),
            "judge_failures": len([r for r in results if r["score"] < 0]),
            "lowest": min((r["score"] for r in results if r["score"] >= 0), default=0),
            "all_pass": not failed, "elapsed_seconds": round(elapsed, 2),
        }, ensure_ascii=False, indent=2))
    else:
        print("")
        print_report(results, args.threshold, elapsed, args.gold, args.model, args.judge_model)

    # Exit 4 if the ONLY reason for failure is an unreachable judge.
    if any(r["score"] < 0 for r in results) and not any(0 <= r["score"] < args.threshold for r in results):
        return 4

    # Same verdict for the isolated judge, which cannot signal -1 for a TRANSPORT
    # failure: it degrades to "fact not present" per call, so a dead judge yields
    # a plausible-looking all-zeros run. A majority of failed calls means the
    # SCORES ARE INVALID and must not be read as a system result (see ISO_CALLS).
    # (It does return -1 for NO_MUST_COVER, but that is a data defect caught at
    # load time, not a transport failure, and load_gold drops those rows first.)
    total_iso = ISO_CALLS["ok"] + ISO_CALLS["failed"]
    if total_iso and ISO_CALLS["failed"] > total_iso / 2:
        msg = ("JUDGE UNRELIABLE: %d of %d isolated judge calls failed. Scores in "
               "this run are meaningless, not a system result."
               % (ISO_CALLS["failed"], total_iso))
        # Only suggest changing the judge when it is not ALREADY the default.
        # The old text said "re-run with --judge-model llama3.2:3b" even on runs
        # that were already using it, sending two sessions hunting a judge
        # change that had nothing left to change. When the judge is already the
        # default, the cause is transport, so print that instead.
        if args.judge_model != DEFAULT_JUDGE:
            msg += ("\n  A 7B judge cannot load after answering pins %s "
                    "(KEEP_ALIVE_FAST=-1) on an 8GB board. Re-run with the "
                    "default judge (--judge-model %s)."
                    % (DEFAULT_MODEL, DEFAULT_JUDGE))
        elif TRANSPORT_ERRORS["count"]:
            msg += ("\n  Judge is already the default (%s), so this is not a "
                    "model-swap problem. %d transport error(s); first: %s"
                    % (DEFAULT_JUDGE, TRANSPORT_ERRORS["count"],
                       TRANSPORT_ERRORS["first"]))
        else:
            msg += ("\n  Judge is already the default (%s) and no transport "
                    "errors were recorded, so Ollama answered but the verdicts "
                    "did not parse. Inspect the raw judge replies."
                    % DEFAULT_JUDGE)
        print(msg, file=sys.stderr)
        return 4

    # The guard above only catches a judge that THROWS. A judge that returns
    # well-formed but useless verdicts slips straight through it: the
    # qwen2.5:7b baseline run recorded errors: [], judge_failures: 0, and a
    # visibly correct answer_preview, yet scored a plausible-looking 0/28.
    # A real system does not score zero on every question while still producing
    # non-empty answers from retrieved sources -- that shape is a dead judge,
    # so refuse to report it as a result.
    # A third rather than a majority: on the run this guard is built from, only
    # 15 of 28 questions had both a preview and sources, which clears a majority
    # by one. Requiring a majority would make detection turn on noise.
    if results and all(r["score"] == 0 for r in results):
        answered = [r for r in results
                    if (r.get("answer_preview") or "").strip() and r.get("n_sources")]
        if len(answered) >= max(1, len(results) / 3):
            print("JUDGE UNRELIABLE: every one of %d questions scored 0, but %d of "
                  "them produced a non-empty answer from retrieved sources.\n"
                  "  A uniform zero with real answers is a judge outage, not a "
                  "system result. Re-run with --judge-model %s."
                  % (len(results), len(answered), args.model), file=sys.stderr)
            return 4

    return 5 if failed else 0


if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()
        sys.exit(code if isinstance(code, int) else 0)
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write("unexpected error: " + repr(exc) + "\n")
        sys.exit(1)
