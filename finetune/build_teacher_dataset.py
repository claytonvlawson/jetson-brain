#!/usr/bin/env python3
"""Generate SFT training answers for a persona_gold split using a Claude model
as the teacher, instead of self-distilling from the local 3B+RAG brain.

Why: build_sft_dataset.py's existing sft_raw/*_answers.jsonl files come from
asking the brain ITSELF (stock 3B + RAG) to answer, then keeping the
high-scoring ones. A LoRA trained on that can at best match stock+RAG,
standalone -- it can never beat the system it was distilled from. v1 (27
examples) and v2 (38 examples) both lost to a freshly re-run stock baseline
for exactly this reason on top of too little data. This script swaps the
answer source for Claude, given the SAME retrieved vault context the live
brain would see plus the row's must_cover/must_not checklist spelled out
explicitly, so the teacher's answer has a real shot at covering more of the
checklist than the 3B does on its own.

Retrieval is reused verbatim from brain.py (retrieve_context +
build_first_pass_user) so the teacher sees the same passages -- and the same
crisis-question system-prompt addendum -- the live app would use. Only the
generator changes; grounding does not.

Two ways to call the teacher:
  * synchronous (default): one POST /v1/messages per row, judged as it lands.
  * --batch: all rows go up in one Message Batch (50% of the per-token price),
    the script polls until processing_status == "ended", downloads the JSONL
    results, then judges. The batch id is written to
    finetune/sft_raw/<out-name>_batch.json the moment it is created, so a
    killed poll resumes with --batch-id instead of paying twice.

Usage (run on the Mac, vault index must already be built):
    set -a && . ./anthropic.env.local && set +a
    BRAIN_INDEX=./index-mac-v2/bm25.pkl python3 finetune/build_teacher_dataset.py \\
        --split finetune/persona_gold/splits/train_800.jsonl \\
        --out-name persona_teacher --batch

Output (same shape build_sft_dataset.py already knows how to read):
    finetune/sft_raw/<out-name>_answers.jsonl   {id, question, answer, sources, served_by}
    finetune/sft_raw/<out-name>_scores.json     {"results": [{"id": ..., "score": ...}]}
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import brain  # noqa: E402
from eval_brain import judge_isolated  # noqa: E402

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TEACHER_MODEL = "claude-sonnet-5"
DEFAULT_JUDGE_MODEL = "qwen2.5:7b"
RAW_DIR = os.path.join(SCRIPT_DIR, "sft_raw")

# Batch custom_id rule from the API docs: ^[a-zA-Z0-9_-]{1,64}$
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def load_split(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def persona_note(persona):
    if not persona:
        return ""
    bits = [k.replace("_", " ") + ": " + str(v).replace("_", " ")
            for k, v in persona.items()]
    return "Parent context (for tone, not to be recited back): " + "; ".join(bits) + "\n\n"


def build_teacher_messages(row, chunks, meta, bm25):
    """Mirror brain.py's phase-1 vault-first turn: same retrieval, same
    context formatting, same crisis addendum -- Claude stands in for the 3B
    as the generator only."""
    question = row["question"]
    r = brain.retrieve_context(question, bm25, chunks, meta, persona="parent")
    retrieved = r["retrieved"]
    added, linked_ids = brain.expand_with_links(retrieved, r["scores"], meta)
    context_chunks = retrieved + added

    system_prompt = brain.PARENT_SYSTEM
    system_prompt = brain._shape_for_intent(system_prompt, r["intent"])
    if brain._is_crisis_question(question):
        system_prompt += brain.CRISIS_SYSTEM_ADDENDUM

    cover = row.get("must_cover") or []
    not_cover = row.get("must_not") or []
    checklist = (
        "\n\nYour answer will be graded against this exact checklist. Every "
        "item below MUST be conveyed somewhere in your answer, in your own "
        "words -- paraphrase is fine, omission is not:\n"
        + "\n".join("- " + c for c in cover)
    )
    if not_cover:
        checklist += (
            "\n\nDo NOT assert any of the following as true:\n"
            + "\n".join("- " + c for c in not_cover)
        )
    # Length guard. The batch smoke saw one answer run past 2048 output tokens
    # (~1400 words) on a question the same model answered in 411 words a run
    # earlier. That is a runaway, not thoroughness, and a 3B cannot learn a
    # 1400-word target anyway. The nine kept smoke answers were 326-522 words.
    checklist += (
        "\n\nKeep the whole answer under 450 words. Cover every checklist item, "
        "but do not pad, repeat, or add sections beyond what the question needs."
    )

    user_content = (
        persona_note(row.get("persona"))
        + brain.build_first_pass_user(question, context_chunks, linked_ids)
        + checklist
    )
    sources = brain.vault_sources(context_chunks)
    return system_prompt, user_content, sources


# ---------------------------------------------------------------------------
# Citation normalisation.
#
# PARENT_SYSTEM tells the model to "cite the sources by filename". The teacher
# obeys, inconsistently: "(LanguageDeprivationAgent/Parents FAQ.md)" in one
# answer, "(Parents FAQ)" in the next, "(First 90 Days After Diagnosis.md)"
# in a third, plus "(linked)" markers copied from the context labels. Full
# vault paths were also what pushed one smoke answer past max_tokens. The
# training target should be one consistent, short form: "(Parents FAQ)".
# Applied BEFORE judging so the judge scores exactly what the 3B will learn.
# ---------------------------------------------------------------------------
_LINKED_RE = re.compile(r"\s*\(linked\)")
# folder/sub folder/Title.md -> Title   (folders may contain spaces)
_PATH_MD_RE = re.compile(r"(?:[^\s/()][^/()\n]*?/)+([^/()\n]+?)\.md\b")
# any leftover bare "Title.md" right before a separator or close-paren
_BARE_MD_RE = re.compile(r"\.md(?=[\s,;:)])")


def normalize_citations(text):
    text = _LINKED_RE.sub("", text)
    text = _PATH_MD_RE.sub(r"\1", text)
    text = _BARE_MD_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Anthropic API, stdlib only (no `anthropic`/`requests` package in this env).
# ---------------------------------------------------------------------------
def _headers(api_key):
    return {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }


def _api_json(url, api_key, payload=None, retries=3, timeout=60):
    """One JSON round trip with backoff on 429/5xx. Returns the parsed body."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method="POST" if data is not None else "GET",
            headers=_headers(api_key))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            last_err = "HTTP %d: %s" % (e.code, body[:300])
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            break
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 ** attempt)
    raise RuntimeError("API call failed after retries: " + str(last_err))


def _message_params(system_prompt, user_content, model, max_tokens):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }


def _unpack_message(data):
    """(text, stop_reason, usage) from a Messages API response body."""
    parts = data.get("content") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return text.strip(), data.get("stop_reason"), data.get("usage") or {}


def call_claude(system_prompt, user_content, model, api_key, retries=3, timeout=60,
                max_tokens=2048):
    """Synchronous single call. Returns (text, stop_reason, usage). The caller
    must reject stop_reason == "max_tokens": the first smoke run cut an answer
    off mid-citation at 1024 and the judge still scored it 100, because every
    checklist fact landed before the cut."""
    data = _api_json(ANTHROPIC_URL, api_key,
                     _message_params(system_prompt, user_content, model, max_tokens),
                     retries=retries, timeout=timeout)
    return _unpack_message(data)


def batch_submit(prepared, model, max_tokens, api_key):
    """POST one Message Batch for every prepared row. Returns the batch id."""
    requests_ = []
    for row, system_prompt, user_content, _sources in prepared:
        qid = row["id"]
        if not _CUSTOM_ID_RE.match(qid):
            raise ValueError("row id %r is not a legal batch custom_id" % qid)
        requests_.append({
            "custom_id": qid,
            "params": _message_params(system_prompt, user_content, model, max_tokens),
        })
    data = _api_json(ANTHROPIC_BATCH_URL, api_key, {"requests": requests_}, timeout=300)
    return data["id"]


def batch_wait(batch_id, api_key, poll_seconds):
    """Poll until processing_status == "ended". Returns the final batch object."""
    while True:
        b = _api_json(ANTHROPIC_BATCH_URL + "/" + batch_id, api_key)
        counts = b.get("request_counts") or {}
        print("  batch %s: %s  processing=%s succeeded=%s errored=%s canceled=%s expired=%s"
              % (batch_id, b.get("processing_status"), counts.get("processing"),
                 counts.get("succeeded"), counts.get("errored"),
                 counts.get("canceled"), counts.get("expired")), flush=True)
        if b.get("processing_status") == "ended":
            return b
        time.sleep(poll_seconds)


def batch_results(results_url, api_key, timeout=300):
    """Download the JSONL results. Returns {custom_id: result_dict} where
    result_dict is the per-line "result" object ({"type": ..., "message": ...}
    or {"type": "errored", "error": ...})."""
    req = urllib.request.Request(results_url, headers=_headers(api_key))
    out = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp.read().decode("utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            out[line["custom_id"]] = line.get("result") or {}
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   help="persona_gold split jsonl (question + must_cover/must_not)")
    p.add_argument("--out-name", required=True,
                   help="base name for finetune/sft_raw/<name>_answers.jsonl / _scores.json")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-host", default="http://127.0.0.1:11434")
    p.add_argument("--min-score", type=int, default=90)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="teacher output cap; answers that hit it are dropped")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="seconds between synchronous Claude calls")
    p.add_argument("--batch", action="store_true",
                   help="use the Message Batches API (half price, async)")
    p.add_argument("--batch-id", default="",
                   help="resume: skip submission and poll/collect this batch id")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--no-normalize-citations", action="store_true",
                   help="keep the teacher's citation text verbatim")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ANTHROPIC_API_KEY is not set. Run:\n"
            "  set -a && . ./anthropic.env.local && set +a\n"
            "before this script.\n")
        return 4

    rows = load_split(args.split)
    if args.limit:
        rows = rows[:args.limit]
    print("loaded %d row(s) from %s" % (len(rows), args.split))

    chunks, bm25, meta = brain.load_index()
    if not chunks or bm25 is None:
        sys.stderr.write("vault index not loaded -- run build_index.py first\n")
        return 4

    os.makedirs(RAW_DIR, exist_ok=True)
    answers_path = os.path.join(RAW_DIR, args.out_name + "_answers.jsonl")
    scores_path = os.path.join(RAW_DIR, args.out_name + "_scores.json")
    batch_path = os.path.join(RAW_DIR, args.out_name + "_batch.json")
    # Truncated and sub-threshold answers land here for inspection. The
    # filename deliberately does NOT end in _answers.jsonl, so
    # build_sft_dataset.py's glob never picks it up.
    rejects_path = os.path.join(RAW_DIR, args.out_name + "_rejects.jsonl")

    # 1. Retrieval + prompt construction for every row, all local.
    prepared = []
    errors = 0
    for row in rows:
        try:
            system_prompt, user_content, sources = build_teacher_messages(
                row, chunks, meta, bm25)
        except Exception as e:
            sys.stderr.write("[%s] prompt build failed: %r\n" % (row["id"], e))
            errors += 1
            continue
        prepared.append((row, system_prompt, user_content, sources))
    print("prepared %d prompt(s), %d build error(s)" % (len(prepared), errors))

    # 2. Teacher generation -> {id: (text, stop_reason, usage)} or an error string.
    generated = {}
    if args.batch or args.batch_id:
        batch_id = args.batch_id
        if not batch_id:
            batch_id = batch_submit(prepared, args.teacher_model, args.max_tokens, api_key)
            with open(batch_path, "w", encoding="utf-8") as bf:
                json.dump({"batch_id": batch_id, "split": args.split,
                           "rows": len(prepared), "teacher_model": args.teacher_model,
                           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, bf, indent=2)
            print("submitted batch %s (%d requests); id saved to %s"
                  % (batch_id, len(prepared), batch_path), flush=True)
        else:
            print("resuming batch %s" % batch_id)
        final = batch_wait(batch_id, api_key, args.poll_seconds)
        results_url = final.get("results_url")
        if not results_url:
            sys.stderr.write("batch ended but results_url is empty: %r\n" % final)
            return 5
        results = batch_results(results_url, api_key)
        for row, _s, _u, _src in prepared:
            res = results.get(row["id"])
            if not res:
                generated[row["id"]] = "no result line for this custom_id"
            elif res.get("type") == "succeeded":
                generated[row["id"]] = _unpack_message(res.get("message") or {})
            else:
                generated[row["id"]] = "%s: %s" % (res.get("type"),
                                                  json.dumps(res.get("error"))[:300])
    else:
        for i, (row, system_prompt, user_content, _src) in enumerate(prepared):
            try:
                generated[row["id"]] = call_claude(
                    system_prompt, user_content, args.teacher_model, api_key,
                    max_tokens=args.max_tokens)
            except Exception as e:
                generated[row["id"]] = repr(e)
            if (i + 1) % 25 == 0:
                print("  generated %d/%d" % (i + 1, len(prepared)), flush=True)
            time.sleep(args.sleep)

    # 3. Normalise, reject truncations, judge, write. Same for both paths.
    kept = dropped = truncated = 0
    usage_in = usage_out = 0
    results = []
    with open(answers_path, "w", encoding="utf-8") as af, \
            open(rejects_path, "w", encoding="utf-8") as rf:
        for i, (row, _s, _u, sources) in enumerate(prepared):
            qid = row["id"]
            g = generated.get(qid)
            if not isinstance(g, tuple):
                sys.stderr.write("[%s] generation failed: %s\n" % (qid, g))
                errors += 1
                continue
            answer, stop_reason, usage = g
            usage_in += int(usage.get("input_tokens") or 0)
            usage_out += int(usage.get("output_tokens") or 0)
            if stop_reason == "max_tokens":
                # A cut-off answer is not training data, whatever it scores.
                sys.stderr.write("[%s] truncated at max_tokens=%d, dropped\n"
                                 % (qid, args.max_tokens))
                truncated += 1
                results.append({"id": qid, "score": -1, "missing": [],
                                 "asserted": [], "mode": "truncated"})
                rf.write(json.dumps({"id": qid, "reason": "truncated",
                                     "words": len(answer.split()),
                                     "answer": answer}, ensure_ascii=False) + "\n")
                continue
            if not args.no_normalize_citations:
                answer = normalize_citations(answer)

            score, missing, asserted, mode = judge_isolated(
                row["question"], answer, row.get("must_cover", []),
                row.get("must_not", []), args.judge_model, args.judge_host,
                retries=3, timeout=120)

            results.append({"id": qid, "score": score, "missing": missing,
                             "asserted": asserted, "mode": mode})
            if score >= args.min_score:
                af.write(json.dumps({
                    "id": qid, "question": row["question"], "answer": answer,
                    "sources": sources, "served_by": "claude-teacher",
                }, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
                rf.write(json.dumps({"id": qid, "reason": "score<%d" % args.min_score,
                                     "score": score, "missing": missing,
                                     "asserted": asserted, "answer": answer},
                                    ensure_ascii=False) + "\n")

            if (i + 1) % 25 == 0:
                print("  judged %d/%d  kept=%d dropped=%d truncated=%d errors=%d"
                      % (i + 1, len(prepared), kept, dropped, truncated, errors),
                      flush=True)

    with open(scores_path, "w", encoding="utf-8") as sf:
        json.dump({"results": results}, sf, indent=2)

    print("done: kept %d, dropped %d (score < %d), %d truncated, %d error(s)"
          % (kept, dropped, args.min_score, truncated, errors))
    print("teacher usage: %d input tokens, %d output tokens over %d answer(s)%s"
          % (usage_in, usage_out, kept + dropped + truncated,
             " [batch: half price]" if (args.batch or args.batch_id) else ""))
    print("wrote %s, %s" % (answers_path, scores_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
