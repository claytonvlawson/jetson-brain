#!/usr/bin/env python3
"""review_qa.py -- the curation step of the advisor's "curated growth" loop.

The advisor (advisor_server.py) logs every visitor exchange to qa_log.jsonl and
every thumbs vote to feedback.jsonl. Nothing a visitor types reaches the vault
on its own. THIS tool is the human gate: you read the exchanges, and the good
ones get promoted into the vault as notes so the next answer is better. That is
what "training" means for a BM25 + Ollama RAG: growing and sharpening the
grounded knowledge base, never letting anonymous input write to it directly.

Flow per exchange:
    approve  -> write a note to <vault>/_Advisor/<slug>.md (edit the text first
                if you want), then mark it handled
    reject   -> mark handled, write nothing
    skip     -> leave it for a later session
    quit     -> stop; approvals so far are kept

Handled ids are recorded in reviewed.jsonl so nothing you already judged comes
back. After a session with approvals it offers to re-run build_index.py so the
new notes become retrievable.

    python3 review_qa.py            # review everything not yet handled
    python3 review_qa.py --limit 10 # review at most 10 this session

Paths follow advisor_server.py and build_index.py, and honor the same env
overrides (ADVISOR_LOG_DIR, BRAIN_VAULT), so dev machines can point elsewhere.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("ADVISOR_LOG_DIR", "/mnt/nvme/brain/advisor")
VAULT_DIR = os.environ.get("BRAIN_VAULT", "/mnt/nvme/brain/vault")
QA_LOG = os.path.join(LOG_DIR, "qa_log.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")
REVIEWED_LOG = os.path.join(LOG_DIR, "reviewed.jsonl")
ADVISOR_NOTES = os.path.join(VAULT_DIR, "_Advisor")


def load_jsonl(path):
    """Return a list of parsed records, skipping any malformed line."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def slugify(text, maxlen=60):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    s = s[:maxlen].strip("-")
    return s or "note"


def priority(exchange, feedback_by_id):
    """Lower sorts first. Down-voted and commented answers are the ones most
    worth a human's eyes (a wrong answer to fix or a gap to fill)."""
    fbs = feedback_by_id.get(exchange["id"], [])
    if any(f.get("vote") == "down" for f in fbs):
        return 0
    if any((f.get("note") or "").strip() for f in fbs):
        return 1
    if any(f.get("vote") == "up" for f in fbs):
        return 3   # already liked; lower urgency to curate
    return 2       # no feedback at all


def read_multiline(prompt):
    """Read a replacement answer. A lone Enter keeps the original; otherwise
    type freely and end with a single '.' on its own line."""
    print(prompt)
    print("(Enter alone keeps the original. To replace it, type the new answer "
          "and finish with a single '.' on its own line.)")
    first = input()
    if first.strip() == "":
        return None
    if first.strip() == ".":
        return ""
    lines = [first]
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def write_note(exchange, answer_text):
    os.makedirs(ADVISOR_NOTES, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.localtime(exchange.get("ts", time.time())))
    question = (exchange.get("question") or "").strip()
    sources = exchange.get("sources") or []
    slug = slugify(question)
    fname = "%s-%s-%s.md" % (day, slug, exchange["id"][:6])
    path = os.path.join(ADVISOR_NOTES, fname)

    refs = "".join("\n  - %s" % s for s in sources) if sources else " []"
    src_line = ("\n\nSources: " + ", ".join(sources)) if sources else ""
    # Frontmatter keys stay simple so build_index.py's plain markdown walk indexes
    # the body cleanly. The question is repeated as an H1 so keyword retrieval has
    # the visitor's own phrasing to match on.
    body = (
        "---\n"
        'title: "%s"\n'
        "type: advisor-qa\n"
        "reviewed: %s\n"
        "refs:%s\n"
        "---\n\n"
        "# %s\n\n"
        "%s%s\n"
    ) % (question.replace('"', "'"), day, refs, question, answer_text.strip(), src_line)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def show(exchange, feedback_by_id, idx, total):
    fbs = feedback_by_id.get(exchange["id"], [])
    ups = sum(1 for f in fbs if f.get("vote") == "up")
    downs = sum(1 for f in fbs if f.get("vote") == "down")
    notes = [f.get("note", "").strip() for f in fbs if (f.get("note") or "").strip()]

    print("\n" + "=" * 72)
    print("[%d/%d]  id=%s   votes: +%d / -%d" % (idx, total, exchange["id"], ups, downs))
    print("-" * 72)
    print("Q: " + (exchange.get("question") or "").strip())
    print("-" * 72)
    print("A: " + (exchange.get("answer") or "").strip())
    if exchange.get("sources"):
        print("-" * 72)
        print("Sources: " + ", ".join(exchange["sources"]))
    for n in notes:
        print("-" * 72)
        print("Visitor correction: " + n)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Review advisor exchanges and promote the good ones into the vault.")
    parser.add_argument("--limit", type=int, default=0,
                        help="review at most N exchanges this session (0 = no cap)")
    args = parser.parse_args()

    exchanges = load_jsonl(QA_LOG)
    feedback = load_jsonl(FEEDBACK_LOG)
    reviewed_ids = {r.get("id") for r in load_jsonl(REVIEWED_LOG)}

    feedback_by_id = {}
    for f in feedback:
        feedback_by_id.setdefault(f.get("id"), []).append(f)

    queue = [e for e in exchanges if e.get("id") and e["id"] not in reviewed_ids]
    queue.sort(key=lambda e: (priority(e, feedback_by_id), -e.get("ts", 0)))
    if args.limit > 0:
        queue = queue[:args.limit]

    print("Advisor review")
    print("  logged exchanges : %d" % len(exchanges))
    print("  already handled  : %d" % len(reviewed_ids))
    print("  to review now    : %d" % len(queue))
    if not queue:
        print("\nNothing to review. When visitors ask questions on the site, "
              "they show up here.")
        return 0

    approvals = []
    for i, ex in enumerate(queue, 1):
        show(ex, feedback_by_id, i, len(queue))
        while True:
            choice = input("  [a]pprove  [r]eject  [s]kip  [q]uit > ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            print("  Please enter a, r, s, or q.")

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "r":
            append_jsonl(REVIEWED_LOG, {"id": ex["id"], "action": "reject", "ts": time.time()})
            print("  rejected (will not reappear).")
            continue

        # approve: optionally edit the answer, then write the note
        replacement = read_multiline("  Curate the answer text if you want:")
        answer_text = ex.get("answer", "") if replacement is None else replacement
        if not answer_text.strip():
            print("  empty answer, skipping this one.")
            continue
        path = write_note(ex, answer_text)
        append_jsonl(REVIEWED_LOG, {"id": ex["id"], "action": "approve",
                                    "note_path": path, "ts": time.time()})
        approvals.append(path)
        print("  approved -> " + path)

    print("\nDone. Approved %d note(s) this session." % len(approvals))
    if approvals:
        for p in approvals:
            print("  " + p)
        ans = input("\nRe-index the vault now so these become searchable? [y/N] ").strip().lower()
        if ans == "y":
            script = os.path.join(HERE, "build_index.py")
            print("Running build_index.py ...")
            rc = subprocess.run([sys.executable, script], cwd=HERE).returncode
            print("build_index.py exited with %d" % rc)
            return rc
        else:
            print("Skipped re-index. Run it later with:  python3 build_index.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(0)
