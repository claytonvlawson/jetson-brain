"""eval_route.py -- local vs cloud answer-generation A/B over FROZEN retrieval.

Measures what README.md's "Cloud speed tier" note says nobody had measured:
accuracy and latency of the two generators behind langchain_agent.py,

    local  = llama3.2:3b via brain.chat_once()      (the Jetson's own model)
    cloud  = Groq openai/gpt-oss-20b via ChatOpenAI   (the speed tier)

on IDENTICAL context. Retrieval runs exactly once per question, through
langchain_agent.build_messages() (-> vault_tools_local.search_vault ->
brain.retrieve_context), and the same messages go to both generators. That is
the only fair reading of "retrieval is unchanged, only generation moves": HyDE
inside retrieve_context() is itself a 3B generation, so two separate calls do
NOT see the same context unless it is frozen here.

Generation runs through langchain_agent.generate(strict=True). In strict mode a
cloud failure (timeout, exception, empty content twice) is a scored-zero row
tagged served_by="cloud-error" -- never the silent fall-back to local that
production answer() does, which would label a local answer as cloud.

Scoring reuses eval_brain.py's judge verbatim (JUDGE_TEMPLATE, judge(),
_extract_json, the 100*present/len(must_cover) - 25*asserted formula). The
only substitution is the transport: with --judge groq:<model> the judge
prompt goes to Groq instead of Ollama, so a 26-question, two-arm battery
grades in about a minute instead of the 15+ minutes a local qwen2.5:7b needs
(which also cannot load beside the pinned 3B on an 8 GB board). Pick a judge
from a DIFFERENT family than the cloud arm -- gpt-oss grading gpt-oss is the
same self-preference trap as the 3B self-judge this repo already documented.

Per-arm answers are also written as eval_brain --answers-out JSONL, so the
slower local judge can re-grade the SAME answers later:

    python3 eval_brain.py --grade-only eval_results/ROUTE_<ts>.cloud.jsonl \\
        --gold factual_gold.jsonl --judge-model qwen2.5:7b

Run on the Jetson (Ollama is local there; Groq key in groq.env.local):

    cd /mnt/nvme/brain && set -a && . ./groq.env.local && set +a
    BRAIN_SEED=1 BRAIN_TEMPERATURE=0 BRAIN_AUTOLEARN=0 BRAIN_PRIVACY_SCAN=0 \\
      python3 eval_route.py \\
        --gold factual_gold.jsonl,multihop_gold.jsonl,meta_gold.jsonl \\
        --arms local,cloud --judge groq:qwen/qwen3.8-27b \\
        --budget-s 600 --out eval_results/ROUTE_$(date +%Y%m%d_%H%M).json

Exit codes: 0 finished, 2 nothing to run, 3 finished but PARTIAL (budget hit
or judge unreachable for most rows), 4 configuration error (no GROQ_API_KEY
for a cloud arm / cloud judge).
"""

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

import brain
import eval_brain
import langchain_agent as la

ARM_TIER = {"local": "local-fast", "deep": "local-deep", "cloud": "cloud-speed"}
PASS_THRESHOLD = 90          # eval_brain's default --threshold
JUDGE_MAX_TOKENS = 2048      # reasoning judges spend tokens before the JSON; 512 truncates
# Groq's free tier meters TOKENS per minute (a few thousand), not just requests,
# and a judge prompt carries the whole answer. A 429 there says "try again in
# 7.3s" in the body; wait that long, cumulatively up to this budget per call.
JUDGE_RATE_LIMIT_WAIT_S = 120.0
RATE_LIMIT = {"hits": 0, "slept_s": 0.0}
# langchain_agent's production default. The eval may run the cloud arm at a
# longer --cloud-timeout so QUALITY is measured; every cloud row also records
# whether it would have been cut off (and fallen back to local) in production.
PROD_CLOUD_TIMEOUT_S = 8.0


def _retry_after(headers, body_text, default=2.0):
    """Seconds Groq asked us to wait: Retry-After header first, else the
    "Please try again in 7.35s" / "in 850ms" phrase in the error body."""
    import re
    try:
        ra = headers.get("retry-after") if headers else None
        if ra:
            return min(max(float(ra), 1.0), 90.0)
    except Exception:
        pass
    m = re.search(r"try again in ([0-9.]+)\s*(ms|s)", body_text or "")
    if m:
        v = float(m.group(1))
        v = v / 1000.0 if m.group(2) == "ms" else v
        return min(max(v + 0.25, 1.0), 90.0)
    return min(max(default, 1.0), 90.0)


# -- Groq judge transport (same signature as eval_brain._ollama_chat) -----------
def _groq_chat(model, messages, host, want_json, timeout):
    """POST /chat/completions to Groq with urllib. Drop-in for
    eval_brain._ollama_chat: same signature, same JudgeTransportError contract,
    so eval_brain.judge()/_judge_once() run unmodified. `host` is ignored
    (recorded for display only). Handles 429 with Retry-After inside the call
    so the judge's own retry loop never sees a rate limit as a dead judge."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise eval_brain.JudgeTransportError("GROQ_API_KEY is not set")
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": JUDGE_MAX_TOKENS}
    if os.environ.get("BRAIN_SEED"):
        try:
            body["seed"] = int(os.environ["BRAIN_SEED"])
        except ValueError:
            pass
    if want_json:
        body["response_format"] = {"type": "json_object"}
    # Qwen3 on Groq thinks before it answers; ask for the thinking to be
    # hidden so the content is the verdict JSON alone. Dropped on a 400 in
    # case the parameter is not accepted for this model id (see loop below).
    if "qwen" in model.lower():
        body["reasoning_format"] = "hidden"
    data = json.dumps(body).encode("utf-8")
    url = la.GROQ_BASE_URL + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key,
               # urllib does not decompress; and the Jetson's system Brotli
               # is the crash langchain_agent already works around.
               "Accept-Encoding": "identity"}
    waited = 0.0
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if e.code == 400 and "reasoning_format" in body:
                body.pop("reasoning_format", None)
                data = json.dumps(body).encode("utf-8")
                continue
            if e.code == 429:
                wait = _retry_after(e.headers, detail, default=2.0 * attempt)
                if waited + wait > JUDGE_RATE_LIMIT_WAIT_S or attempt > 12:
                    raise eval_brain.JudgeTransportError(
                        "judge rate-limited: waited %.0fs over %d tries; last: %s"
                        % (waited, attempt, detail)) from e
                RATE_LIMIT["hits"] += 1
                RATE_LIMIT["slept_s"] += wait
                waited += wait
                time.sleep(wait)
                continue
            raise eval_brain.JudgeTransportError(
                "judge HTTP %s from groq: %s" % (e.code, detail)) from e
        except eval_brain.JudgeTransportError:
            raise
        except Exception as e:
            raise eval_brain.JudgeTransportError(
                "judge unreachable at groq: %r" % (e,)) from e
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise eval_brain.JudgeTransportError(
            "judge returned non-JSON (%d bytes): %.200r" % (len(raw), raw)) from e
    choices = payload.get("choices") or []
    if not choices:
        raise eval_brain.JudgeTransportError("judge returned no choices: %.200r" % raw)
    return ((choices[0].get("message") or {}).get("content")) or ""


def groq_limits(model):
    """One 1-token call so Groq's x-ratelimit-* headers tell us the account's
    RPM/TPM for this model id. Recorded in the fingerprint: a battery paced
    under an 8k TPM free tier and one under a paid tier are different arms."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"error": "no key"}
    body = json.dumps({"model": model, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    req = urllib.request.Request(la.GROQ_BASE_URL + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key,
                                          "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            h = resp.headers
    except urllib.error.HTTPError as e:
        h = e.headers
        out = {"http": e.code}
    else:
        out = {"http": 200}
    for k in ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
              "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens", "retry-after"):
        v = h.get(k) if h else None
        if v is not None:
            out[k] = v
    return out


def parse_judge(spec):
    """'groq:qwen/qwen3.8-27b' | 'ollama:qwen2.5:7b' | 'none' -> (kind, model)."""
    spec = (spec or "none").strip()
    if spec.lower() == "none":
        return "none", None
    kind, _, model = spec.partition(":")
    kind = kind.lower()
    if kind not in ("groq", "ollama") or not model:
        raise ValueError("--judge must be groq:<model>, ollama:<model>, or none")
    return kind, model


def install_judge(kind):
    """Point eval_brain's judge at the chosen transport. Returns the `host`
    string eval_brain.judge() expects (display-only for groq)."""
    if kind == "groq":
        eval_brain._ollama_chat = _groq_chat
        return la.GROQ_BASE_URL
    return brain.OLLAMA_HOST


# -- fingerprint --------------------------------------------------------------
class _FpArgs(object):
    def __init__(self, mode):
        self.mode = mode


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def fingerprint(persona, judge_kind, judge_model, arms):
    fp = {}
    try:
        fp = eval_brain.env_fingerprint(_FpArgs(persona))
    except Exception as e:
        fp = {"env_fingerprint_error": repr(e)}
    idx = getattr(brain, "INDEX_PATH", None)
    fp.update({
        # The route keys eval_brain's fingerprint does not know about.
        "BRAIN_ROUTE_MODE": la._route_mode(),
        "BRAIN_CLOUD_MODEL": la._cloud_model_name(),
        "BRAIN_CLOUD_TIMEOUT_S": la._cloud_timeout_s(),
        "prod_cloud_timeout_s": PROD_CLOUD_TIMEOUT_S,
        "cloud_temperature": la._cloud_temperature(),
        "BRAIN_PRIVACY_SCAN": os.environ.get("BRAIN_PRIVACY_SCAN", "0"),
        "local_model": brain.DEFAULT_MODEL,
        "deep_model": brain.DEEP_MODEL,
        "arms": {a: ARM_TIER[a] for a in arms},
        "judge": {"kind": judge_kind, "model": judge_model,
                  "votes": 1, "template": "eval_brain.JUDGE_TEMPLATE"},
        "index_path": idx,
        "index_sha256": _sha256(idx) if idx else None,
        "python": sys.version.split()[0],
    })
    if "cloud" in arms or judge_kind == "groq":
        fp["groq_limits"] = {}
        for m in sorted({x for x in [la._cloud_model_name() if "cloud" in arms else None,
                                     judge_model if judge_kind == "groq" else None] if x}):
            try:
                fp["groq_limits"][m] = groq_limits(m)
            except Exception as e:
                fp["groq_limits"][m] = {"error": repr(e)[:120]}
    for mod in ("openai", "langchain_openai", "langchain_core", "ollama"):
        try:
            fp["ver_" + mod] = __import__(mod).__version__
        except Exception:
            fp["ver_" + mod] = None
    return fp


# -- the loop -----------------------------------------------------------------
def load_rows(paths):
    rows, seen = [], set()
    for p in paths:
        for r in eval_brain.load_gold(p):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            r["_gold_file"] = os.path.basename(p)
            rows.append(r)
    return rows


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def aggregate(results, arm):
    rs = [r for r in results if r["arm"] == arm]
    scored = [r for r in rs if r["score"] is not None and r["score"] >= 0]
    gen_ok = [r for r in rs if r["error"] is None]
    facts_total = sum(r["n_must_cover"] for r in scored)
    facts_hit = sum(r["n_must_cover"] - len(r["missing"]) for r in scored)
    lat = [r["seconds"] for r in gen_ok]
    return {
        "n_rows": len(rs),
        "n_generated": len(gen_ok),
        "n_generation_errors": len(rs) - len(gen_ok),
        # score None = --judge none (unjudged by design); -1 = the judge failed
        "n_judge_failures": sum(1 for r in rs if r["error"] is None and r["score"] is not None and r["score"] < 0),
        "n_scored": len(scored),
        "mean_score": round(statistics.mean([r["score"] for r in scored]), 1) if scored else None,
        "pass_at_%d" % PASS_THRESHOLD: sum(1 for r in scored if r["score"] >= PASS_THRESHOLD),
        "fact_hit_rate": round(facts_hit / facts_total, 3) if facts_total else None,
        "facts_hit": facts_hit, "facts_total": facts_total,
        "must_not_violations": sum(len(r["errors"]) for r in scored),
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
        "latency_median_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(_pct(lat, 0.95), 2) if lat else None,
        "answer_chars_mean": round(statistics.mean([r["answer_chars"] for r in gen_ok]), 0) if gen_ok else None,
        "cloud_attempts_total": sum(r.get("attempts") or 0 for r in rs),
        "rate_limited_calls": sum(r.get("rate_limited") or 0 for r in rs),
        "rate_limit_waited_s": round(sum(r.get("waited_s") or 0 for r in rs), 1),
    }


def paired(results, a, b):
    by = {}
    for r in results:
        by.setdefault(r["id"], {}).setdefault(r["arm"], []).append(r)
    out, wins = [], {a: 0, b: 0, "tie": 0, "unscored": 0}
    for qid, arms in by.items():
        if a not in arms or b not in arms:
            continue

        def _mean_score(rs):
            xs = [r["score"] for r in rs if r["score"] is not None and r["score"] >= 0]
            return round(statistics.mean(xs), 1) if xs else None

        def _mean_lat(rs):
            xs = [r["seconds"] for r in rs if r["error"] is None]
            return round(statistics.mean(xs), 2) if xs else None

        sa, sb = _mean_score(arms[a]), _mean_score(arms[b])
        if sa is None or sb is None:
            w = "unscored"
        elif sa > sb:
            w = a
        elif sb > sa:
            w = b
        else:
            w = "tie"
        wins[w] += 1
        out.append({"id": qid, a + "_score": sa, b + "_score": sb,
                    a + "_s": _mean_lat(arms[a]), b + "_s": _mean_lat(arms[b]),
                    "winner": w})
    return {"wins": wins, "by_id": out}


def run(rows, index, arms, persona, judge_kind, judge_model, judge_host,
        repeat=1, budget_s=None, answers_dir=None, out_stem="ROUTE",
        gen_fn=None, build_fn=None, log=None, judge_retries=2, judge_timeout=60,
        dry_run=False, parallel=True, checkpoint=None, cloud_rate_limit_wait_s=120.0):
    """The battery. Main thread owns everything that touches Ollama (retrieval
    incl. HyDE, the local arms); the cloud arm's generation+judging is pure
    network, so with parallel=True it runs on a worker thread while the 3B
    generates -- that overlap is what fits 26 questions in the budget. Each
    arm's judge call runs right after its own generation."""
    gen_fn = gen_fn or la.generate
    build_fn = build_fn or la.build_messages
    log = log or (lambda s: None)
    started = time.time()
    results = []
    partial = False
    answer_files = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2) if parallel else None
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)
        for arm in arms:
            answer_files[arm] = open(os.path.join(answers_dir, "%s.%s.jsonl" % (out_stem, arm)), "w")

    def one(g, arm, rep, messages, sources, ctx_chars, retr_s, auto_route):
        tier = ARM_TIER[arm]
        res = gen_fn(messages, tier, strict=True, rate_limit_wait_s=cloud_rate_limit_wait_s)
        served = res.get("served_by") or ""
        # strict mode never falls back, but a measurement that could mislabel
        # its arm is worse than one that aborts.
        if arm == "cloud" and not served.startswith("cloud"):
            raise RuntimeError("cloud arm served by %r -- fallback leaked into the eval" % served)
        if arm != "cloud" and served.startswith("cloud"):
            raise RuntimeError("local arm served by %r" % served)
        answer = res.get("text") or ""
        err = res.get("error")
        if err is not None:
            score, missing, errors, raw = 0, list(g["must_cover"]), [], "generation-error"
        elif judge_kind == "none":
            score, missing, errors, raw = None, [], [], "unjudged"
        else:
            score, missing, errors, raw = eval_brain.judge(
                g["question"], answer, g["must_cover"], g.get("must_not") or [],
                judge_model, judge_host, judge_retries, judge_timeout, votes=1)
        secs = round(res.get("seconds") or 0.0, 3)
        return {
            "id": g["id"], "gold_file": g.get("_gold_file"), "arm": arm, "rep": rep,
            "served_by": served, "model": res.get("model"),
            "score": score, "missing": missing, "errors": errors, "judge_raw": raw,
            "n_must_cover": len(g["must_cover"]),
            "seconds": secs, "retrieval_seconds": round(retr_s, 3),
            "over_prod_timeout": (secs > PROD_CLOUD_TIMEOUT_S) if arm == "cloud" else None,
            "attempts": res.get("attempts"), "error": err,
            "rate_limited": res.get("rate_limited"), "waited_s": res.get("waited_s"),
            "finish_reason": res.get("finish_reason"), "usage": res.get("usage"),
            "answer_chars": len(answer), "answer_preview": answer[:200],
            "sources": sources, "context_chars": ctx_chars, "auto_route": auto_route,
        }, answer

    def record(row, answer, g, line):
        results.append(row)
        arm = row["arm"]
        if arm in answer_files:
            answer_files[arm].write(json.dumps({
                "id": g["id"], "question": g["question"], "answer": answer, "sources": row["sources"],
                # eval_brain's provenance gate wants "rag" for a fresh
                # retrieval-grounded answer; the realized backend rides
                # alongside so a re-grade can still tell the arms apart.
                "served_by": "error" if row["error"] is not None else "rag",
                "backend": row["served_by"], "model": row["model"],
            }, ensure_ascii=False) + "\n")
            answer_files[arm].flush()
        sc = row["score"]
        tag = ("ERR" if row["error"] is not None else
               ("n/a" if sc is None else ("J?" if sc < 0 else "%3d" % sc)))
        line.append(" %s %s %.1fs" % (arm, tag, row["seconds"]))

    try:
        for i, g in enumerate(rows, 1):
            if budget_s is not None and time.time() - started > budget_s:
                partial = True
                log("BUDGET: %ds exceeded after %d/%d questions; stopping (partial)."
                    % (budget_s, i - 1, len(rows)))
                break
            q = g["question"]
            t0 = time.perf_counter()
            messages, hits = build_fn(q, persona=persona, index=index)
            retr_s = time.perf_counter() - t0
            sources = [h.get("source", "") for h in hits]
            ctx_chars = len(messages[1]["content"]) if len(messages) > 1 else 0
            auto_route = la.choose_tier(q, persona=persona)
            line = ["[%2d/%2d] %-28s ctx=%d src=%d %.1fs |" % (i, len(rows), g["id"][:28], ctx_chars, len(sources), retr_s)]
            if dry_run:
                line.append(" auto_route=%s" % auto_route)
                log("".join(line))
                continue
            futures = []
            inline = []
            for arm in arms:
                for rep in range(repeat):
                    args_ = (g, arm, rep, messages, sources, ctx_chars, retr_s, auto_route)
                    if pool is not None and arm == "cloud":
                        futures.append(pool.submit(one, *args_))
                    else:
                        inline.append(args_)
            done = []
            for args_ in inline:
                done.append(one(*args_))
            for f in futures:
                done.append(f.result())
            # keep rows in --arms order regardless of which finished first
            order = {a: n for n, a in enumerate(arms)}
            done.sort(key=lambda t: (order[t[0]["arm"]], t[0]["rep"]))
            for row, answer in done:
                record(row, answer, g, line)
            log("".join(line))
            if checkpoint is not None:
                try:
                    checkpoint(results, False)
                except Exception:
                    pass
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for fh in answer_files.values():
            fh.close()
    return results, partial, time.time() - started


def render(report):
    arms = list(report["aggregate"].keys())
    cols = ["n_scored", "mean_score", "pass_at_%d" % PASS_THRESHOLD, "fact_hit_rate",
            "must_not_violations", "n_generation_errors", "n_judge_failures",
            "latency_mean_s", "latency_median_s", "latency_p95_s", "answer_chars_mean",
            "rate_limited_calls", "rate_limit_waited_s"]
    w = max(len(c) for c in cols)
    lines = ["", "arm" + " " * (w - 3) + "".join("%14s" % a for a in arms)]
    for c in cols:
        lines.append(c.ljust(w) + "".join("%14s" % report["aggregate"][a].get(c) for a in arms))
    if report.get("paired"):
        p = report["paired"]
        lines.append("")
        lines.append("paired (per question, by score): " + ", ".join(
            "%s=%d" % (k, v) for k, v in p["wins"].items()))
    lines.append("")
    lines.append("judge=%s:%s  rate_limit_hits=%d (slept %.0fs)  judge_transport_errors=%d  elapsed=%.0fs%s"
                 % (report["judge"]["kind"], report["judge"]["model"],
                    report["rate_limit"]["hits"], report["rate_limit"]["slept_s"],
                    report["judge_transport_errors"]["count"], report["elapsed_seconds"],
                    "  PARTIAL" if report["partial"] else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--gold", default="factual_gold.jsonl,multihop_gold.jsonl,meta_gold.jsonl",
                    help="comma-separated gold .jsonl files (eval_brain.load_gold shape)")
    ap.add_argument("--arms", default="local,cloud", help="comma-separated: local, cloud, deep")
    ap.add_argument("--judge", default="groq:qwen/qwen3.8-27b",
                    help="groq:<model> | ollama:<model> | none")
    ap.add_argument("--persona", default="default", choices=["default", "parent", "scholar"])
    ap.add_argument("--repeat", type=int, default=1, help="generations per (question, arm)")
    ap.add_argument("--budget-s", type=float, default=600.0, help="stop starting new questions after this")
    ap.add_argument("--only", action="append", default=[], help="gold id to include (repeatable)")
    ap.add_argument("--out", default=None, help="report JSON path (default eval_results/ROUTE_<ts>.json)")
    ap.add_argument("--answers-dir", default=None,
                    help="where per-arm --answers-out JSONL goes (default: alongside --out)")
    ap.add_argument("--judge-timeout", type=int, default=60)
    ap.add_argument("--judge-retries", type=int, default=2)
    ap.add_argument("--cloud-timeout", type=float, default=30.0,
                    help="per-call Groq timeout for the eval (production uses %.0fs; rows record "
                         "over_prod_timeout so both numbers exist)" % PROD_CLOUD_TIMEOUT_S)
    ap.add_argument("--cloud-rate-limit-wait", type=float, default=120.0,
                    help="max seconds per cloud generation to sit out Groq 429s (honoring Retry-After); "
                         "waits are recorded, not counted as latency or as errors")
    ap.add_argument("--no-parallel", action="store_true",
                    help="run the cloud arm inline instead of overlapping it with local generation")
    ap.add_argument("--dry-run", action="store_true", help="retrieve + route only; no generation, no judge")
    args = ap.parse_args(argv)
    if args.cloud_timeout:
        os.environ["BRAIN_CLOUD_TIMEOUT_S"] = str(args.cloud_timeout)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_TIER:
            sys.stderr.write("unknown arm %r (choose from %s)\n" % (a, ", ".join(ARM_TIER)))
            return 2
    try:
        judge_kind, judge_model = parse_judge(args.judge)
    except ValueError as e:
        sys.stderr.write(str(e) + "\n")
        return 2
    if not args.dry_run and ("cloud" in arms or judge_kind == "groq") and not os.environ.get("GROQ_API_KEY"):
        sys.stderr.write("GROQ_API_KEY is not set. `set -a; . ./groq.env.local; set +a` first.\n")
        return 4

    rows = load_rows([p.strip() for p in args.gold.split(",") if p.strip()])
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    if not rows:
        sys.stderr.write("no gradeable gold rows loaded\n")
        return 2

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = args.out or os.path.join("eval_results", "ROUTE_%s.json" % ts)
    out_stem = os.path.splitext(os.path.basename(out))[0]
    answers_dir = args.answers_dir or os.path.dirname(out) or "."

    judge_host = install_judge(judge_kind)
    fp = fingerprint(args.persona, judge_kind, judge_model, arms)

    def log(s):
        print(s, flush=True)

    log("eval_route: %d questions x %s x%d, judge=%s:%s, budget=%ss, index=%s"
        % (len(rows), "+".join(arms), args.repeat, judge_kind, judge_model, args.budget_s,
           (fp.get("index_sha256") or "?")[:12]))
    index = brain.load_index()

    def build_report(results, partial, elapsed):
        return {
            "kind": "eval_route", "started": ts, "elapsed_seconds": round(elapsed, 1),
            "partial": partial, "gold": args.gold.split(","), "n_questions": len(rows),
            "n_completed": len({r["id"] for r in results}),
            "arms": {a: {"tier": ARM_TIER[a]} for a in arms},
            "judge": {"kind": judge_kind, "model": judge_model},
            "fingerprint": fp,
            "aggregate": {a: aggregate(results, a) for a in arms},
            "paired": paired(results, arms[0], arms[1]) if len(arms) >= 2 else None,
            "rate_limit": dict(RATE_LIMIT),
            "judge_transport_errors": dict(eval_brain.TRANSPORT_ERRORS),
            "answers_files": {a: os.path.join(answers_dir, "%s.%s.jsonl" % (out_stem, a)) for a in arms},
            "results": results,
        }

    t_run = time.time()

    def checkpoint(results, partial):
        # Crash-safe: the report on disk is always the latest completed question.
        rep = build_report(results, True, time.time() - t_run)
        rep["checkpoint"] = True
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)

    results, partial, elapsed = run(
        rows, index, arms, args.persona, judge_kind, judge_model, judge_host,
        repeat=args.repeat, budget_s=args.budget_s,
        answers_dir=None if args.dry_run else answers_dir, out_stem=out_stem, log=log,
        judge_retries=args.judge_retries, judge_timeout=args.judge_timeout,
        dry_run=args.dry_run, parallel=not args.no_parallel,
        checkpoint=None if args.dry_run else checkpoint,
        cloud_rate_limit_wait_s=args.cloud_rate_limit_wait)
    if args.dry_run:
        log("dry run: %d questions retrieved in %.0fs, nothing generated." % (len(rows), elapsed))
        return 0

    report = build_report(results, partial, elapsed)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    log(render(report))
    log("wrote " + out)
    judged = [r for r in results if r["error"] is None]
    unjudged = [r for r in judged if r["score"] is not None and r["score"] < 0]
    if partial or (judged and len(unjudged) > len(judged) / 2):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
