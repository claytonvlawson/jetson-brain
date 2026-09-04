"""Offline check that build_index.py carries frontmatter tags onto chunk rows
(no Ollama: runs the builder with --no-embed on a temp vault).

Tagged notes get a lowercase, de-duplicated "tags" list on every chunk; untagged
notes get no key at all, so older pickles and untagged vaults are unchanged.
Run: python3 test_index_tags.py
"""
import os
import pickle
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_index  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


check("chunk_tags normalizes case, '#', and duplicates",
      build_index.chunk_tags({"tags": ["Paper", "#computer-vision", "paper", " "]})
      == ["paper", "computer-vision"])
check("chunk_tags is empty without tags",
      build_index.chunk_tags({}) == [] and build_index.chunk_tags(None) == [])

TAGGED = """---
name: Albanie et al 2020 BSL-1K
type: source
tags: [source, paper, Computer-Vision, sign-language-recognition]
---
# BSL-1K

Scaling up co-articulated sign language recognition using mouthing cues.

## Method

A pose transformer is trained on a thousand signs from BBC broadcast footage.
"""
BLOCK = """---
tags:
  - legal
  - statute
---
# Northern Ireland Sign Language Bill

The bill gives BSL and ISL statutory recognition in Northern Ireland.
"""
PLAIN = """# What Is Language Deprivation

Language deprivation is the absence of accessible language input during the
critical period.
"""

with tempfile.TemporaryDirectory() as tmp:
    vault = os.path.join(tmp, "vault")
    os.makedirs(os.path.join(vault, "Sources"))
    os.makedirs(os.path.join(vault, "Core"))
    with open(os.path.join(vault, "Sources", "Albanie 2020 - BSL-1K.md"), "w") as f:
        f.write(TAGGED)
    with open(os.path.join(vault, "Sources", "NI Bill.md"), "w") as f:
        f.write(BLOCK)
    with open(os.path.join(vault, "Core", "What Is Language Deprivation.md"), "w") as f:
        f.write(PLAIN)
    out = os.path.join(tmp, "bm25.pkl")
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "build_index.py"),
         "--vault", vault, "--out", out, "--no-embed"],
        capture_output=True, text=True)
    check("build_index.py --no-embed exits 0 (stderr: %s)" % proc.stderr.strip()[-200:],
          proc.returncode == 0 and os.path.exists(out))
    if proc.returncode == 0:
        with open(out, "rb") as f:
            idx = pickle.load(f)
        chunks = idx["chunks"]
        by_src = {}
        for c in chunks:
            by_src.setdefault(c["source"], []).append(c)
        alb = by_src.get("Sources/Albanie 2020 - BSL-1K.md", [])
        ni = by_src.get("Sources/NI Bill.md", [])
        core = by_src.get("Core/What Is Language Deprivation.md", [])
        check("all three notes produced chunks", alb and ni and core)
        check("every chunk of the inline-tagged note carries lowercase tags",
              alb and all(c.get("tags") == ["source", "paper", "computer-vision",
                                             "sign-language-recognition"] for c in alb))
        check("block-list tags are carried too",
              ni and all(c.get("tags") == ["legal", "statute"] for c in ni))
        check("untagged note rows have no tags key",
              core and all("tags" not in c for c in core))
        check("ids are positional (scope masks index score arrays by position)",
              [c["id"] for c in chunks] == list(range(len(chunks))))
        check("tokenizer version unchanged by the schema change",
              idx.get("tokenizer_version") == build_index.TOKENIZER_VERSION)

print()
print("RESULT:", "ALL PASS" if not FAILS else ("FAILURES: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
