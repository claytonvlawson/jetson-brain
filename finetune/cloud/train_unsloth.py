#!/usr/bin/env python3
"""LoRA fine-tune of Llama-3.2-3B-Instruct on a rented CUDA GPU (Vast.ai / RunPod),
using Unsloth. Replaces the v1 MLX-LM (`mlx_lm.lora`) pipeline, which only runs on
Apple Silicon and can't use a rented A100/H100.

Consumes the same chat-format JSONL produced by finetune/build_sft_dataset.py:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage (on the rented instance, after `pip install unsloth`):
    python3 train_unsloth.py \
        --train-file data/train.jsonl \
        --valid-file data/valid.jsonl \
        --rank 8 --alpha 16 \
        --out-dir out/parent-ft-v2

Run it twice with --rank 8 and --rank 16 (both finish in minutes on an H100/A100 given
the dataset size) to sweep rank cheaply within the same rented session, per the
plan's step 3.

Add --gguf to also emit a Q4_K_M GGUF on the instance (Unsloth builds llama.cpp on the
fly). If that works you download ~2 GB instead of the ~6.5 GB merged model and skip the
Mac-side conversion. If it fails, merged/ is already written; fall back to the Mac path
in README.md step 5.

API notes (checked against Unsloth's Llama3.2 notebook, 2026-09): uses trl.SFTConfig,
not transformers.TrainingArguments; transformers >= 4.46 renamed evaluation_strategy
to eval_strategy. Known-good pins: transformers==4.56.2, trl==0.22.2.
"""
import argparse
import json
import os


def load_chat_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--valid-file", default="data/valid.jsonl")
    p.add_argument("--out-dir", default="out/parent-ft-v2")
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (v1 used 8)")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha (v1 scale was rank*2.5=20)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=10,
                   help="v2 run 2026-09-07: 3 epochs = 15 steps, eval loss still falling; 10 is the new floor")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gguf", action="store_true",
                   help="also write a Q4_K_M GGUF next to merged/ (needs llama.cpp build on the instance)")
    args = p.parse_args()

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect bf16/fp16
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def to_text(rows):
        return [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            for r in rows
        ]

    train_rows = load_chat_jsonl(args.train_file)
    valid_rows = load_chat_jsonl(args.valid_file)
    train_ds = Dataset.from_dict({"text": to_text(train_rows)})
    valid_ds = Dataset.from_dict({"text": to_text(valid_rows)}) if valid_rows else None

    print(f"train examples: {len(train_ds)}  valid examples: {len(valid_ds) if valid_ds else 0}")
    if len(train_ds) < 150:
        print(
            f"WARNING: only {len(train_ds)} training examples -- same overfitting risk "
            "flagged by build_sft_dataset.py for v1. Expect this run to mainly validate "
            "the pipeline, not necessarily beat the v1 verdict on quality.",
        )

    os.makedirs(args.out_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=5,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.001,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            eval_strategy="epoch" if valid_ds else "no",
            save_strategy="epoch",
            output_dir=args.out_dir,
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train()

    # Merge LoRA into the base weights so the checkpoint downloads as a single
    # standard HF model -- ready for llama.cpp's convert_hf_to_gguf.py, same
    # converter v1 used (mlx_lm.fuse --export-gguf was flagged buggy).
    merged_dir = os.path.join(args.out_dir, "merged")
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged HF model written to {merged_dir}. Download this directory, "
          f"then convert to GGUF on the Mac (see finetune/cloud/README.md).")

    if args.gguf:
        gguf_dir = os.path.join(args.out_dir, "gguf")
        try:
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
            # Unsloth appends "_gguf" to the directory you pass, and names the file after
            # the base model, e.g. out/<run>/gguf_gguf/llama-3.2-3b-instruct.Q4_K_M.gguf
            print(f"Q4_K_M GGUF written under {gguf_dir}_gguf/ (Unsloth adds the suffix). "
                  f"Download just that .gguf file and skip README step 5; go straight to step 6 "
                  f"(ollama create). Then `rm -rf {merged_dir}` before the next run to save disk.")
        except Exception as exc:  # llama.cpp build on the instance is the usual failure
            print(f"GGUF export failed ({exc!r}). merged/ is intact -- use the Mac path in "
                  f"README step 5 instead.")


if __name__ == "__main__":
    main()
