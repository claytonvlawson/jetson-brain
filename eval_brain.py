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


# -- config, mirrored from brain.py so the eval matches the deployed system ---
OLLAMA_HOST = getattr(brain, "OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = getattr(brain, "DEFAULT_MODEL", "llama3.2:3b")
DEFAULT_JUDGE = getattr(brain, "DEEP_MODEL", "qwen2.5:7b")
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


def _ollama_chat(model, messages, host, want_json, timeout):
    """POST /api/chat to Ollama with urllib. Returns content str or None."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
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
            payload = json.loads(resp.read().decode("utf-8"))
        return ((payload.get("message") or {}).get("content")) or ""
    except Exception:
        return None


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
        content = _ollama_chat(judge_model, messages, host, True, timeout)
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


def _iso_bool(template, key, filler, judge_model, host, retries, timeout, default):
    """Ask one isolated yes/no question and return the boolean. `default` on failure."""
    prompt = template.format(**filler)
    messages = [
        {"role": "system", "content": ISO_FACT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for _attempt in range(max(1, retries)):
        content = _ollama_chat(judge_model, messages, host, True, timeout)
        obj = _extract_json(content)
        if isinstance(obj, dict) and key in obj:
            return bool(obj.get(key))
    return default


def judge_isolated(question, answer, must_cover, must_not, judge_model, host, retries, timeout):
    """Grade each fact in its OWN focused call. A small judge checking one fact at a time is
    far more reliable than a 6-item checklist, where it loses track and drops present facts.
    A must-cover fact defaults to NOT present on judge failure (conservative); a must-not
    claim defaults to NOT asserted (do not invent violations). Returns the judge() tuple."""
    must_cover = list(must_cover or [])
    must_not = list(must_not or [])
    if not must_cover:
        return 100, [], [], None
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
        return 100, [], [], None
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
def load_gold(path):
    """Read gold_questions.jsonl into a list of dicts. Skips blank/comment lines
    and malformed rows (with a stderr note). Returns []. Never raises."""
    rows = []
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
                rows.append(obj)
    except Exception as e:
        sys.stderr.write("could not read gold set: " + repr(e) + "\n")
    return rows


def preflight_judge(judge_model, host, timeout):
    """Confirm the judge model answers at all. Returns True/False."""
    reply = _ollama_chat(
        judge_model,
        [{"role": "user", "content": "Reply with the single word: ready"}],
        host, False, timeout,
    )
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
    if broke:
        print("  judge failures : " + str(len(broke)) + " (judge unreachable, treat as env error)")
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
    p.add_argument("--judge-votes", type=int, default=3, help="judge samples per question, majority-voted")
    p.add_argument("--isolate", action="store_true",
                   help="grade each fact in its own focused call (most reliable for a small judge)")
    p.add_argument("--timeout", type=int, default=180, help="per-call timeout seconds")
    p.add_argument("--dry-run", action="store_true", help="list the plan, grade nothing")
    p.add_argument("--json", dest="as_json", action="store_true", help="emit JSON report")
    p.add_argument("--mode", choices=("default", "parent"), default="default",
                   help="answer persona: parent = warm parent-guidance register")
    return p


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

    # Confirm the judge is reachable before spending time answering.
    if not preflight_judge(args.judge_model, args.host, min(60, args.timeout)):
        sys.stderr.write(
            "judge model " + args.judge_model + " is not answering at " + args.host
            + ". Is Ollama running and the model pulled?\n")
        return 4

    if not args.as_json:
        print("Grading " + str(len(gold)) + " question(s). Answer then judge, one at a time.")

    def run_judge(g, answer):
        if args.isolate:
            return judge_isolated(
                g["question"], answer, g.get("must_cover", []), g.get("must_not", []),
                args.judge_model, args.host, args.retries, args.timeout)
        return judge(
            g["question"], answer, g.get("must_cover", []), g.get("must_not", []),
            args.judge_model, args.host, args.retries, args.timeout, args.judge_votes)

    results = []
    if args.two_phase:
        # Phase 1: answer everything (answerer model stays resident), then Phase 2:
        # judge everything (judge model stays resident). One model load per phase
        # instead of a 3B<->judge swap on every question, which on an 8GB board that
        # cannot hold both models is the dominant cost of a --no-cache run.
        if not args.as_json:
            print("Two-phase: answering all, then judging all (avoids model swapping).")
        answered = []
        for i, g in enumerate(gold, 1):
            answer, sources, served_by = the_brain.ask(g["question"])
            answered.append((g, answer, sources, served_by))
            if not args.as_json:
                print("  answered [%2d/%2d] %-8s %s" % (i, len(gold), served_by, g["id"]))
        for i, (g, answer, sources, served_by) in enumerate(answered, 1):
            score, missing, errors, _raw = run_judge(g, answer)
            results.append({
                "id": g["id"], "question": g["question"], "score": score,
                "served_by": served_by, "missing": missing, "errors": errors,
                "answer_preview": (answer or "")[:200], "n_sources": len(sources or []),
            })
            if not args.as_json:
                tag = "PASS" if score >= args.threshold else ("JUDGE?" if score < 0 else "FAIL")
                print("  judged [%2d/%2d] %-6s %4s  %s" % (
                    i, len(gold), tag, ("n/a" if score < 0 else str(score)), g["id"]))
    else:
        for i, g in enumerate(gold, 1):
            answer, sources, served_by = the_brain.ask(g["question"])
            score, missing, errors, _raw = run_judge(g, answer)
            results.append({
                "id": g["id"], "question": g["question"], "score": score,
                "served_by": served_by, "missing": missing, "errors": errors,
                "answer_preview": (answer or "")[:200], "n_sources": len(sources or []),
            })
            if not args.as_json:
                tag = "PASS" if score >= args.threshold else ("JUDGE?" if score < 0 else "FAIL")
                print("  [%2d/%2d] %-6s %4s  %s" % (
                    i, len(gold), tag, ("n/a" if score < 0 else str(score)), g["id"]))

    elapsed = time.time() - start
    failed = [r for r in results if r["score"] < args.threshold]

    if args.as_json:
        print(json.dumps({
            "gold": args.gold, "model": args.model, "judge": args.judge_model,
            "threshold": args.threshold, "results": results,
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
