#!/usr/bin/env python
"""Full fine-tuning of the AR student on synthetic (question, response) pairs, PANGEA's recipe.

PANGEA (appendix B): Alpaca template, full-parameter SFT, AdamW, lr 1e-5, total batch 64,
sequence length 4096, cosine schedule with warm-up ratio 0.03, 3 epochs (FSDP on 4×H100).
Here it runs on one GPU: per-device batch × gradient accumulation = 64, bf16, gradient
checkpointing. The loss covers the response tokens only.

  python scripts/train_student.py --data data/synth/g1/medqa/train.jsonl --out ckpt/g1_medqa
  # data: jsonl / parquet with "instruction" and "output" columns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent

# Alpaca (no-input) template, as in PANGEA's training and eval_*.py.
ALPACA = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Response:\n")


def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True)
    df = df[["instruction", "output"]].dropna()
    return df[(df.instruction.str.strip() != "") & (df.output.str.strip() != "")]


def tokenize(tok, max_len):
    def fn(ex):
        prompt = tok(ALPACA.format(instruction=ex["instruction"].strip()), add_special_tokens=False)["input_ids"]
        resp = tok(ex["output"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = ([tok.bos_token_id] + prompt + resp)[:max_len]
        labels = ([-100] * (1 + len(prompt)) + resp)[:max_len]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "model" / "Llama-3.2-1B"))
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--total-batch", type=int, default=64)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    df = load_pairs(args.data)
    ds = Dataset.from_pandas(df, preserve_index=False).map(
        tokenize(tok, args.max_len), remove_columns=["instruction", "output"], num_proc=8)
    n_trunc = sum(len(x) == args.max_len for x in ds["input_ids"])
    print(f"{len(ds)} examples from {args.data} ({n_trunc} truncated at {args.max_len} tokens)")

    # fp32 master weights with bf16 autocast: at lr 1e-5 most updates are below bf16 resolution.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa")
    assert args.total_batch % args.per_device_batch == 0
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.total_batch // args.per_device_batch,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        group_by_length=True,
        logging_steps=10,
        save_strategy="no",  # only the final model (trainer.save_model); resume states cost 14G each
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
