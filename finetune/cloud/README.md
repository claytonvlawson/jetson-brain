# Cloud LoRA retrain: rent → train → bring back to Jetson

Replaces v1's Mac-local MLX-LM pipeline with a rented CUDA GPU + Unsloth, since MLX only
runs on Apple Silicon. See `../../.claude/plans/how-can-i-train-luminous-perlis.md` and
`HANDOFF-parent-lora-cloud-retrain.md` in `~/.claude/plans/` for full background.

Dataset as of this write-up: `finetune/data/train.jsonl` (33 examples) /
`data/valid.jsonl` (5 examples) — grown from v1's 27/4 by mining new grounded questions
from the ThesisVault, still well under the ~150-example floor `build_sft_dataset.py`
warns is needed to avoid overfitting. Proceeding anyway (Clayton's call) mainly to
validate the new pipeline end-to-end and see if the modest data increase + a rank sweep
beats v1's 52.6 mean, not expecting a guaranteed win.

## 1. Rent the GPU (do this yourself — real money, real account)

**Verdict (prices checked 2026-09-06): RunPod Community Cloud, 1x H100 PCIe 80GB, $1.99/hr,
billed per second, $10 minimum deposit.** Cheapest H100 that is reliable, single-GPU,
self-serve, and runs Unsloth's official Docker image. Expect **$2-4 total** for this job
(about 10 min of training per rank, 60-90 min of session with image pull, upload, download).

| Provider | H100 $/hr | Billing | Notes |
|---|---|---|---|
| **RunPod Community, H100 PCIe** | **$1.99** | per-second | Pick. Secure Cloud $2.89; SXM $2.69 / $3.49. |
| Verda spot, H100 SXM | $1.63 (on-demand $3.25) | 10-min rounding | Cheapest real H100, but spot can be preempted. |
| Vast.ai H100 SXM | floor $1.49-1.87, typical much higher | per-hour listings | Floor is a moving target; hosts flake. |
| Modal H100 SXM | $3.95 | per-second | $30/mo free credit makes this run free, but needs a Modal wrapper, not SSH. |
| Thunder Compute | $3.20 | per-minute | Fine, just pricier. |
| Lambda 1x H100 PCIe | $3.29 | per-minute | Reliable, often sold out at 1x. |

An H100 is overkill for a 3B LoRA on 33 examples (an A40 or RTX 4090 at $0.35-0.45/hr
would do). Per-second billing makes the premium about $2, so keep the H100.

**RunPod click path**
1. Sign up at runpod.io, add **$10** by card. A pod needs at least one hour of credit at
   its rate to launch.
2. **Pods → Deploy**. Filter **Community Cloud**, GPU **H100 PCIe**. Take the cheapest
   listing with high uptime (target $1.99/hr).
3. Template: the latest **Runpod PyTorch 2.x** template works fine (Jupyter on 8888 and SSH
   are built in; run `pip install unsloth` once inside). Or **Custom** with image
   `unsloth/unsloth`, HTTP port `8888`, TCP port `22`, env `JUPYTER_PASSWORD=<pick one>`.
   Container disk **30 GB** (merged bf16 model is ~6.5 GB per run; GGUF build needs
   headroom). **No network volume**: it persists and bills after the pod is gone
   ($0.07/GB/mo), and its quota is invisible to `df`. If you end up with one anyway,
   delete it under **Storage** once the GGUFs are on the Mac.
4. Deploy. Image pull takes 1-3 min. Open **Connect → HTTP 8888** for Jupyter Lab, then
   open a terminal there. For scp, use **Connect → SSH over exposed TCP**; the image's SSH
   user is `unsloth`.
5. When the download is done, **Terminate** the pod (not Stop). Stop keeps billing disk.

**Fallbacks**
- Verda (verda.com) H100 SXM spot at $1.63/hr, no minimum deposit. Interruptible; fine
  for a 10-minute run, rerun if preempted.
- Vast.ai Unsloth Studio template (https://docs.vast.ai/examples/ai-ml-frameworks/unsloth-studio)
  if you want a no-code UI instead of the script. Sort by $/hr and check host reliability.
- Modal, free via the Starter plan's $30/mo credit: adapt
  `modal-examples/06_gpu_and_ml/unsloth_finetune.py`. Not written up here.
- RunPod's own Llama fine-tuning walkthrough, for orientation only:
  https://www.runpod.io/articles/guides/fine-tuning-llama-3-1-a-step-by-step-guide-for-efficient-model-customization

## 2. Upload the dataset + training script

From the Mac, upload these three files to the rented instance (via the provider's file
upload UI, `scp`, or `rsync` if SSH is exposed):

```
finetune/data/train.jsonl
finetune/data/valid.jsonl
finetune/cloud/train_unsloth.py
```

## 3. Install deps and train (on the rented instance)

The `unsloth/unsloth` image already has Unsloth installed. On a bare PyTorch pod instead:

```bash
pip install unsloth
# known-good pair from Unsloth's Llama-3.2 notebook if something drifts:
pip install transformers==4.56.2 && pip install --no-deps trl==0.22.2
```

Then run the sweep (training is ~25 s per run on an H100 at this dataset size; the GGUF
build adds ~5 min). The `--gguf` flag builds a Q4_K_M GGUF on the instance so you can skip
step 5. **Use `--epochs 10`**: the 2026-09-07 run showed 3 epochs is only 15 optimizer
steps (5 of them warmup) and leaves eval loss still falling at 1.84, i.e. undertrained.
Read the per-epoch `eval_loss` lines; the epoch where it bottoms out is the one to keep.

```bash
python3 train_unsloth.py --train-file train.jsonl --valid-file valid.jsonl \
    --rank 8  --alpha 16 --epochs 10 --out-dir out/parent-ft-v2-r8 --gguf
rm -rf out/*/merged out/*/checkpoint-*      # 6.5 GB per run; this is what fills the disk
python3 train_unsloth.py --train-file train.jsonl --valid-file valid.jsonl \
    --rank 16 --alpha 32 --epochs 10 --out-dir out/parent-ft-v2-r16 --gguf
rm -rf out/*/merged out/*/checkpoint-*
```

Each run writes a merged HF-format model to `out/<run>/merged/` and, if the on-instance
llama.cpp build succeeds, the file you actually want:

```
out/<run>/gguf_gguf/llama-3.2-3b-instruct.Q4_K_M.gguf      # ~2 GB (note: gguf_gguf, Unsloth's naming)
```

Delete `merged/` and `checkpoint-*` between runs as shown, otherwise the second run dies
with "disk quota exceeded". Only delete `merged/` after confirming the GGUF exists.

## 4. Download the merged model(s) back to the Mac

**If `--gguf` worked**: download only the ~2 GB
`out/<run>/gguf_gguf/llama-3.2-3b-instruct.Q4_K_M.gguf` to
`finetune/cloud/gguf-v2-r8/model-q4_K_M.gguf` and skip to step 6. Rename on arrival: every
run's GGUF has the same filename, so two downloads in a row become `… (1).gguf` in
`~/Downloads` with nothing but the md5 to tell them apart. The Mac also needs ~3 GB free
per model for `ollama create` (it writes a validation copy), or it fails with "no space
left on device".

**Otherwise**: download the `merged/` directory (or just the smaller LoRA adapter checkpoint
if you'd rather merge locally) for whichever run you want to evaluate first — start with
rank 8 since it matches v1's config most closely. Place it at, e.g.:

```
finetune/cloud/merged-v2-r8/
```

Fastest download from RunPod: zip in the terminal (`tar czf r8.tgz out/parent-ft-v2-r8/merged`)
and download from Jupyter's file browser, or `scp -P <port> unsloth@<ip>:r8.tgz .` via
the SSH-over-TCP details on the Connect panel.

**Then terminate the rented instance** — nothing further needs the GPU.

## 5. Convert to GGUF (back on the Mac, same tool v1 used)

Do NOT use `mlx_lm.fuse --export-gguf` — v1's `finetune/fuse_gguf.log` shows it fails
with `ValueError: [save_gguf] can only serialize row-major arrays`. Use llama.cpp's
converter instead, already present at `finetune/llama.cpp-src/`:

```bash
cd /Users/you.lawson/jetson-brain
source finetune/venv/bin/activate
python3 finetune/llama.cpp-src/convert_hf_to_gguf.py \
    finetune/cloud/merged-v2-r8 \
    --outfile finetune/cloud/gguf-v2-r8/model-f16.gguf \
    --outtype f16
```

Quantize with the Homebrew-installed `llama-quantize` (same binary v1 used):

```bash
llama-quantize \
    finetune/cloud/gguf-v2-r8/model-f16.gguf \
    finetune/cloud/gguf-v2-r8/model-q4_K_M.gguf \
    Q4_K_M
```

## 6. Build the Ollama model

Copy `finetune/Modelfile` and point it at the new GGUF (same system prompt — it's the
exact text `build_sft_dataset.py`'s `SYSTEM_PROMPT` trains against; do not swap in
`brain.py`'s `PARENT_SYSTEM`, which expects RAG snippets injected separately):

```bash
sed 's#./gguf/model-q4_K_M.gguf#./cloud/gguf-v2-r8/model-q4_K_M.gguf#' \
    finetune/Modelfile > finetune/cloud/Modelfile-v2-r8

ollama create llama3.2:3b-parent-ft-v2-r8 -f finetune/cloud/Modelfile-v2-r8
```

Smoke test: `ollama run llama3.2:3b-parent-ft-v2-r8 "We just found out our baby is deaf, what should we do first?"`

## 7. Evaluate against stock and v1 before deploying anything

```bash
export BRAIN_ROOT="$PWD/mac-brain-root" BRAIN_INDEX="$PWD/index-mac/bm25.pkl" \
       BRAIN_AUTOLEARN=0 BRAIN_SEED=42 BRAIN_TEMPERATURE=0
for m in llama3.2:3b llama3.2:3b-parent-ft llama3.2:3b-parent-ft-v2-r8; do
  echo "=== $m ==="
  python3 eval_brain.py --gold parent_gold.jsonl --model "$m" --judge-model qwen2.5:7b \
      --threshold 90 --no-cache --two-phase --mode parent --json
  python3 eval_brain.py --gold parent_hard_gold.jsonl --model "$m" --judge-model qwen2.5:7b \
      --threshold 90 --no-cache --two-phase --mode parent --json
done
```

Check specifically:
- Does v2 beat v1's 52.6 mean and stock's 54.6 mean, not just one of the two gold sets?
- Does v2 fix the crisis-adjacent question that failed on **both** v1 and stock
  (see `PARENT-SAFETY-FINDING.md`)? This is the real bar, not the aggregate score.

Only set `DEFAULT_MODEL` in `brain.py` to the new model if both checks pass. Either way,
record the outcome as a new memory note `jetson-brain-parent-finetune-v2-verdict`,
following the same format as the v1 verdict note.
