# Spark parity run: same settings as the Jetson, then the parent eval

The first thing to run on Spark hardware. Goal is narrow and worth stating:
**change nothing, and see whether the number matches.** Ollama, `llama3.2:3b`,
`qwen2.5:7b`, `nomic-embed-text`, the Jetson's own index file. Identical to
`/mnt/nvme/brain`, relocated with one env var.

Not vLLM. Not `ollama_compat.py`. Those are Phase 2. Running them now changes
four things at once and makes a mismatch undiagnosable.

Cost: about $0.75/hr, so the whole run lands under a dollar. Billing is
create-to-delete with no suspend, so **delete the instance when done.**

---

## Prep: already done (2026-08-27)

The Jetson's live index is pulled, hashed, and staged on the Mac, and the
retrieval baseline is measured against that exact file. Nothing here needs
repeating.

**Staged index:** `~/jetson-brain/index-jetson/bm25.pkl`

| | |
|---|---|
| Size | 12,161,932 bytes |
| Modified | 2026-08-23 19:28 |
| SHA-256 | `430b43d95c6216a8556515de188212b0a6788a9bfc08443c4d8d6517c5927222` |

That hash is the parity anchor. It must survive the trip to the Spark exactly.

**Do not use `~/jetson-brain/index-mac/bm25.pkl`.** It is 6,895,675 bytes from
2026-08-13, a little over half the size and ten days older. It is a side-by-side
A/B artifact, not the live index, and a mismatch against it could be a port bug
or index drift with no way to separate them.

**SSH key for Enverge:** `~/.ssh/id_ed25519.pub` already exists (comment
`clayton-mac-to-jetson`). `pbcopy < ~/.ssh/id_ed25519.pub` and paste. The same
key reaches the Jetson, which is what makes the two-hop index copy work.

---

## 1. Rent and connect  (~1 min of typing)

Enverge, single Spark, $0.75/hr. SSH relay only, first connect can take a minute.

```bash
ssh user@<host>.ssh.enverge.dev
```

Add it to `~/.ssh/config` as `spark` right away so `deploy.sh` and `rsync` can
reach it without retyping the relay host.

---

## 2. Ollama, native  (~1 min typing, ~5 min downloading)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If that fails for lack of sudo (the instance is an unprivileged sandbox), fall
back to the arm64 tarball into `$HOME`:

```bash
mkdir -p ~/.local && curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz \
  | tar -xz -C ~/.local && export PATH=$HOME/.local/bin:$PATH
```

Then start it and pull **the same three models the Jetson runs**:

```bash
ollama serve > ~/ollama.log 2>&1 &
ollama pull llama3.2:3b        # fast tier, ~2.0 GB
ollama pull qwen2.5:7b         # deep tier, ~4.7 GB
ollama pull nomic-embed-text   # embedder, ~274 MB
```

One port, 11434, doing chat and embeddings both. That is why this path needs no
shim.

**Check `ollama ps` after the first query.** Ollama silently splits layers to
CPU when it misjudges memory and it is invisible otherwise. On a 128 GB box with
a 3B model there is no excuse for anything but 100% GPU.

---

## 3. Deploy the code and vault  (~2 min)

From the Mac. `deploy.sh` was written for this:

```bash
cd ~/jetson-brain
JETSON=user@spark REMOTE=$HOME/brain ./deploy.sh
```

Two expected rough edges, neither fatal:

- `deploy.sh` runs `install.sh` at the end, which pip-installs `requirements.txt`
  including `mediapipe`. On aarch64 that may fail and abort the script **after**
  the files have already copied. If it does, the eval does not care. Install what
  brain actually needs by hand:
  ```bash
  ssh spark 'python3 -m pip install --user ollama rank_bm25 requests ddgs beautifulsoup4'
  ```
- `install.sh` also builds the index if the vault is present. Let it, but do not
  use its output. Step 4 overwrites it with the Jetson's.

---

## 4. Copy the staged index over the top  (~1 min)

`install.sh` will have built its own index during step 3. Overwrite it. The
staged copy is already on the Mac, so this is one push and one check:

```bash
# Mac
rsync -a ~/jetson-brain/index-jetson/bm25.pkl spark:~/brain/index/bm25.pkl
ssh spark 'shasum -a 256 ~/brain/index/bm25.pkl'
# must print 430b43d95c6216a8556515de188212b0a6788a9bfc08443c4d8d6517c5927222
```

If the hash differs, stop. Everything downstream is measuring a different index.

---

## 5. The env block: same settings, one relocation

On the Spark. This is the whole configuration.

```bash
export BRAIN_ROOT=$HOME/brain              # was /mnt/nvme/brain
export OLLAMA_HOST=http://127.0.0.1:11434  # unchanged, steers both code paths
export BRAIN_TEMPERATURE=0                 # determinism
export BRAIN_SEED=1                        # determinism
export BRAIN_AUTOLEARN=0                   # do not let the eval write the index
```

Models are **not** set. `brain.py` hardcodes `llama3.2:3b` / `qwen2.5:7b` /
`nomic-embed-text` as defaults, and leaving them alone is the point.

---

## 6. Gate on retrieval first  (~40 s)

Judge-free, deterministic, and the only thing that catches a silent embedder
rewrite. **If this does not match the Mac/Jetson, stop.** No generation number
below means anything.

```bash
cd ~/brain
BRAIN_INDEX=$HOME/brain/index/bm25.pkl BRAIN_AUTOLEARN=0 \
  python3 eval_retrieval.py --gold retrieval_gold.jsonl --json spark_retrieval.json
```

### The number it must return

Measured on the Mac 2026-08-27 against this exact index file, twice: once
unseeded, once under the full step-5 env. **Both returned the same thing**, so
the gate is seed-robust and not an artifact of one run.

```
in_context 32/40 (80.0%)   in_context_expanded 34/40 (85.0%)
MRR 0.5485   median rank 3
recall@k  @1=17  @3=24  @5=29  @8=33  @10=33  @20=38
found anywhere in ranking: 40/40
```

Read **`in_context`**, not MRR and not recall@k. Two identical runs move 8 of 40
questions by 1 to 5 ranks; `in_context` never moved. The only field that differed
between the two Mac runs was `LD sources/q` (0.80 vs 0.82), a breadth metric.

The 8 misses are stable and expected, not a defect to chase: all 40 targets are
found somewhere in the ranking, but these fall outside `keep_k=8` —
`bilingual_bimodal_2` (7), `deaf_community_transmission_1` (11),
`interpreting_access_1` (14), `legal_recognition_1` (14), `gallaudet_1` (25),
`interpreting_access_2` (37).

### Diff the fingerprint, not just the score

`--json` writes a `fingerprint` block. It is the stronger check, because it
catches a wrong index or a rewritten embedder even when the score happens to
land. The Mac baseline:

| Field | Must match |
|---|---|
| `chunks` / `dense_rows` | `2673` / `2673` |
| `dense_model` / `dense_dim` | `nomic-embed-text` / `768` |
| `index_version` | `5` |
| `tokenizer_version` | `tok-nfkd-acr-sstem` |
| `embed_prefix_version` | `none` |
| flags | `BRAIN_HYBRID=1 BRAIN_HYDE=1 BRAIN_RERANK=1 BRAIN_TOP_K_FAST=8`, and `ANN`/`DECOMPOSE`/`OBSIDIAN_BLOCKS`/`OBSIDIAN_RANK` all `0` |

Only `index` (a path) and `seed` are expected to differ.

`embed_prefix_version: none` matters: the `nomic-v1` prefix is a known
regression that costs 8 labels. If it reads anything else, the Spark is
embedding differently.

Reference copies on the Mac:
`eval_results/mac_retrieval_baseline_20260827.json` (unseeded) and
`eval_results/mac_retrieval_seeded_20260827.json` (full step-5 env).

---

## 7. The parent eval  (~5 min)

```bash
BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
  --mode parent --gold parent_gold.jsonl --json > spark_parent.json
```

**Expected: 8/8, mean 100.0.** Same as the Jetson.

Keep the default judge. `DEFAULT_JUDGE` is the answerer's own model, so this is
a 3B self-judge, which is exactly what produced the Jetson's 100.0. Swapping in
`--judge-model qwen2.5:7b` here would change two variables at once and destroy
the comparison. That swap is the *next* run, and it is the real reason to rent.

Note `--mode` and `--gold` are independent: `--mode` picks the persona, `--gold`
picks the questions, and `--gold` defaults unconditionally to
`gold_questions.jsonl` (`eval_brain.py:792`).

Then the rest of the trio, same session, same env. **Not optional:** all three
must land, because parent alone is 8 questions and cannot distinguish a clean
port from a lucky one.

```bash
BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
  --gold gold_questions.jsonl --json > spark_default.json    # 28/28, 100.0
BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
  --gold heldout_gold.jsonl --json > spark_heldout.json      # 13/14, 98.8
```

Start these and walk away. Roughly 25 minutes for all three, no supervision
needed once the retrieval gate has passed.

Do **not** point `eval_brain.py` at `retrieval_gold.jsonl`. It carries zero
`must_cover`, and `load_gold()` correctly exits 2 rather than reporting a free
100.

---

## 8. Pull everything back, then delete  (~1 min)

There is no persistent volume. Delete takes the disk.

```bash
# Mac
rsync -av spark:~/brain/spark_*.json ~/jetson-brain/eval_results/
rsync -av spark:~/ollama.log /tmp/spark-ollama.log
```

Then delete the instance in the Enverge console. **Restart does not stop
billing. Only delete does.**

---

## Reading the result

| Outcome | Means |
|---|---|
| All three match | Port is clean. Proceed to the judge swap and to vLLM. |
| `in_context` differs at step 6 | Retrieval broke in the copy. Wrong index, or the embedder resolved to something else. Fix before reading anything else. |
| Parent passes, heldout drops | Not necessarily a port bug: heldout is the only suite with headroom (13/14), so it is the one that can move on noise. Re-run before concluding anything. |
| Scores drop but retrieval matched | Generation-side drift: check `ollama ps` for a CPU split, and confirm `BRAIN_SEED` and `BRAIN_TEMPERATURE` are actually exported in the shell that ran the eval. |
| ~16 tok/s under Ollama | CPU split, not a bad model. Unload and reload to force 100% GPU. |

**What this run does not tell you.** The parent eval is saturated. 8/8 mean
100.0 on a 3B self-judge cannot show that a bigger model is better, only that
the port did not break. The honest eval, with a non-arm judge and
`parent_hard_gold.jsonl`, is the run after this one. See
`PARENT-SAFETY-FINDING.md` for the failure that compute does not fix.

---

Related: `spark/README.md` (the vLLM path, Phase 2),
`~/.claude/plans/with-the-current-plan-gentle-treasure.md`,
`~/.claude/plans/how-can-i-use-dazzling-reef.md`.
