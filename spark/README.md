# Spark lane: serve the brain on a DGX Spark

Phase 1 of `~/.claude/plans/how-can-i-use-dazzling-reef.md`. Everything here is
written and tested on the Mac; none of it has run on Spark hardware yet.

The goal of this directory is narrow and worth stating: **move `brain.py` to the
Spark without changing a line of its generation path**, so the same eval suites
score the same thing on both boxes and the port has an actual gate.

## What is here

| File | Role |
|---|---|
| `ollama_compat.py` | Ollama-shaped front door for vLLM. brain talks Ollama, vLLM talks OpenAI. |
| `test_ollama_compat.py` | 35 offline checks. No vLLM, no GPU, no network. |

The one change outside this directory is `brain.py`'s `OLLAMA_HOST`, which is
now env-aware. The Jetson default is unchanged and all 21 existing brain tests
still pass.

## Why a shim and not a refactor

`brain.py` reaches the model **two different ways**, and they read the host from
**two different places**:

- `chat()` / `chat_stream()` prefer the `ollama` Python client, which reads the
  `OLLAMA_HOST` *environment variable*
- both fall back to REST against `brain.OLLAMA_HOST`, which was a hardcoded
  literal

Repoint one and not the other and you get a split brain: the client reaches the
Spark while the fallback quietly dials a dead `11434`. It only surfaces on the
error path. Hence the one-line change, so a single env var steers both.

## Serve

```bash
# 1. vLLM, OpenAI API on :8000
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --port 8000 \
  --enable-prefix-caching \
  --max-num-seqs 8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.45
```

**FP8, not NVFP4.** GB10 is **SM121**, and SM121 does not have the
`cvt.rn.satfinite.e2m1x2` PTX instruction that SM120 (RTX 5090) and SM100 (B200)
have. NVFP4 there falls back to Marlin decompression: measured **40.8 tok/s vs
53.8 for FP8** on this same Qwen3.6-35B-A3B. Bigger and fancier loses here.

`--max-num-seqs 8`, not 256. The official vLLM DGX Spark guidance is 4, because
above roughly four concurrent decode streams the per-token bandwidth tax
outweighs the batching gain on a 273 GB/s box. 256 is a setting for a serving
fleet, not for one or two humans and a widget.

`--kv-cache-dtype fp8` is deliberately **absent**. The same vLLM guidance notes
it "can carry a noticeable performance cost" on this hardware; leave it unset
unless KV space actually becomes the binding constraint.

`0.45`, not the usual `0.85`. On unified memory vLLM **preallocates**, so the
common value claims ~100 GB and leaves QLoRA (Loop C) nothing to train in. Raise
it to **0.65** on any run that is not also training, and take the KV back.

`--max-model-len` must exceed the largest window brain budgets (fast 24576, deep
12288). Ollama truncates an over-long prompt from the top, taking the system
prompt with it; vLLM returns a 400 instead. Loud beats silent, but leave room.

```bash
# 2. A REAL embedder, on its own port. vLLM is serving a chat model and does
#    NOT expose nomic-embed-text on /v1/embeddings; without this, every
#    _dense_scores() and _rerank_by_embedding() call fails. nomic-embed-text is
#    137M params (~274 MB), so it costs nothing on a 128 GB box.
OLLAMA_HOST=127.0.0.1:11500 ollama serve &
OLLAMA_HOST=127.0.0.1:11500 ollama pull nomic-embed-text

# 3. The shim, on Ollama's port so nothing else has to know
export SPARK_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export SPARK_EMBED_URL=http://127.0.0.1:11500/v1   # NOT the chat vLLM
export SPARK_EMBED_MODEL=nomic-embed-text     # match what the INDEX was built with
python3 ollama_compat.py

# 4. Point brain at it. One variable, both code paths.
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Verify, in order

Each step gates the next. Do not skip 3.

0. **Does `--gpus` even work?** Minute three, before anything else, because it
   decides whether vLLM installs natively or from a container:
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0-base nvidia-smi
   ```
   On a hosted Spark (Enverge) the instance is an **unprivileged sandbox** and
   `--gpus` is documented as limited, so plan for `uv pip install vllm` against
   the preinstalled CUDA 13 stack and treat the container path as a bonus.
1. **Decode speed** — single-stream **45 to 55 tok/s** on vLLM FP8, **35 to 45**
   under Ollama. It is **not** 289: that number came from a leaderboard sweep on
   a specialized engine, and the published stock-vLLM measurements for this model
   class on GB10 are 48.2 (Ollama), 53.8 (vLLM FP8), 40.8 (vLLM NVFP4). Near
   6 tok/s means a dense model got loaded. **Near 16 tok/s under Ollama means a
   CPU split**, not a bad model: check `ollama ps`, then unload and reload to
   force 100% GPU. It is invisible unless you look.
2. **Resident size** — roughly 23 to 30 GB for a 35B-A3B at 4-to-8-bit. Wildly
   more means a dense model was resolved instead of the MoE.
3. **Embeddings resolve to the model the index was built with.** Watch the
   shim's one-time stderr warning on the first dense query. A silent rewrite is
   the highest-cost failure in this whole lane. Then prove retrieval survived
   the port, deterministically and without a judge, in about 40 seconds:
   ```bash
   BRAIN_INDEX=index/bm25.pkl BRAIN_AUTOLEARN=0 \
     python3 eval_retrieval.py --gold retrieval_gold.jsonl --json spark_retrieval.json
   ```
   Read **`in_context`**, not MRR or recall@k: two identical runs move 8 of 40
   questions by 1 to 5 ranks (brain.py:1089), and `in_context` never moved. If
   `in_context` does not match the Mac, stop. Retrieval is broken and no
   generation result below means anything.
4. **No regression.** This is the whole point of the shim. Note that `--mode`
   and `--gold` are **independent**: `--mode parent` selects the persona,
   `--gold` selects the questions, and `--gold` defaults unconditionally to
   `gold_questions.jsonl` (eval_brain.py:792). There is no `--heldout` flag.
   ```bash
   BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
     --mode parent --gold parent_gold.jsonl    # 8/8, mean 100.0
   BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
     --gold gold_questions.jsonl               # 28/28, mean 100.0
   BRAIN_TEMPERATURE=0 BRAIN_SEED=1 python3 eval_brain.py \
     --gold heldout_gold.jsonl                 # 13/14, mean 98.8
   ```
   Judge with a model that is **not** one of the arms, and not even the same
   family, via `--judge-model`. `eval_brain.py:88` defaults `DEFAULT_JUDGE` to
   the answerer's own model; every "depth" headline from the Jetson era turned
   out to be a 3B self-judge artifact. 128 GB is the first box you have had that
   can hold answerer and judge at once, which is most of why the port is worth
   doing.

   Do **not** point `eval_brain.py` at `retrieval_gold.jsonl`. It carries zero
   `must_cover`, and `load_gold()` correctly exits 2 rather than reporting a
   free 100.
4. **Prefill actually caches** — `gpu_prefix_cache_hit_rate` should be high on
   the parent lane. Near zero means something varies inside the prefix: a
   timestamp or session id in the system prompt, retrieved chunks placed before
   the stable block, Hermes injecting variable memory into the system prompt, or
   a per-request tool list.
5. **Concurrency under mixed load** — run a cron while streaming an answer. Both
   finish, no stall. This is the case the old two-70B layout could not survive.

## Traps found while building this

- **A chat vLLM cannot serve your embedder, and the default hides it.**
  `SPARK_EMBED_URL` defaults to `SPARK_BASE_URL`, which is only correct if one
  backend serves both. A vLLM started for Qwen does not expose
  `nomic-embed-text` on `/v1/embeddings`, so on the Spark this default breaks
  every dense query. Run a real embedding server on its own port and set
  `SPARK_EMBED_URL` explicitly. See step 2 in **Serve**.
- **`build_index.py` and `update_index.py` are not env-aware** (`build_index.py:81`,
  `update_index.py:160`, `:364` hardcode `http://127.0.0.1:11434`). That looks
  like a migration blocker and is not, because the shim *takes over* port 11434.
  The consequence is subtler: a rebuild or an auto-learn tick embeds through
  whatever `SPARK_EMBED_MODEL` resolves to, silently. Re-run `eval_retrieval.py`
  after any index write on this box.
- **The embedder is rewritten, and says so.** brain names its embedder in two
  places and neither is a tier alias, so a name-match rule lets them reach vLLM
  untranslated and it 400s. Every request now routes to the one served embedder
  — but if the on-disk index was built with a *different* model, that rewrite
  makes every dense score a cosine between vectors from two models: no error,
  just quietly wrong retrieval. The shim warns once per name on stderr. Watch
  for it on first deploy.
- **`seed` must survive the hop.** `BRAIN_SEED` is what makes an A/B decidable;
  unseeded judged runs flip 5 of 28. It is mapped straight through, and there is
  a test asserting upstream receives it.
- **`num_ctx` and `num_gpu` are dropped, not proxied.** vLLM 400s on unknown
  fields, and neither has an OpenAI equivalent.
- **`/api/generate` must answer 200.** brain only calls it to unload the fast
  model before a deep turn, because 9.4 GB does not fit in the Jetson's 7.4 GB.
  There is nothing to evict on the Spark, so it is a successful no-op. A 404
  would make brain log an eviction failure on every deep question.
