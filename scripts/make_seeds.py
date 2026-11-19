#!/usr/bin/env python
"""Draw the domain seed set D_s: m random examples per domain from the train split.

Each seed is written as {id, question, pangea_query, answer, gold}: the question formatted
like the dLLM benchmark prompt (eval/tasks/local_*_utils.py), the same question in PANGEA's
query format (its test files and example data; used as the projection reference d_s), and
the seed's own answer text r_j:
  gsm8k  solution with the <<calc>> annotations removed, ending in "#### N"
  medqa  "D. Nitrofurantoin" (MedQA has no rationale)
  finqa  evidence sentences + calculation steps + answer

  python scripts/make_seeds.py [--m 100] [--seed 0]  ->  data/seeds/<domain>.jsonl

It also writes data/seeds/<domain>_sft.jsonl {instruction: pangea_query, output: answer}, the
seed-only SFT set for the S1 baseline (scripts/train_student.py --data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "tasks"))
import local_finqa_utils as finqa  # noqa: E402
import local_medqa_utils as medqa  # noqa: E402


MEDQA_HEADER = ("You are a clinical decision-making assistant. Your task is to read the provided "
                "clinical scenario carefully and choose the best treatment option from the provided choices.")
FINQA_HEADER = "Please answer the given financial question based on the context."


def gsm8k_record(row):
    answer = re.sub(r"<<[^>]*>>", "", row["answer"]).strip()
    return {"question": row["question"].strip(), "pangea_query": row["question"].strip(),
            "answer": answer, "gold": answer.split("####")[-1].strip()}


def medqa_record(row):
    opts = "\n".join(f"{k}. {row['options'][k]}" for k in "ABCD" if row["options"].get(k))
    query = f"{MEDQA_HEADER}\nQuestion: {row['question'].strip()}\nChoose one of the following:\n{opts}"
    return {"question": medqa.format_question(row), "pangea_query": query,
            "answer": medqa.format_answer(row), "gold": row["answer_idx"]}


def finqa_record(row):
    pre = " ".join(json.loads(row["pre_text"]))
    post = " ".join(json.loads(row["post_text"]))
    table = "\n".join("|" + "|".join(c.strip() for c in r) + "|" for r in json.loads(row["table"]))
    q = json.loads(row["qa"])["question"]
    query = f"{FINQA_HEADER}\nContext: {pre}\n{table}\n{post}\nQuestion: {q}"
    return {"question": finqa.format_question(row), "pangea_query": query,
            "answer": finqa.format_answer(row), "gold": finqa.answer_text(row)}


DOMAINS = {
    "gsm8k": ("train/gsm8k/train.parquet", gsm8k_record, None),
    "medqa": ("train/medqa/train.parquet", medqa_record, None),
    "finqa": ("train/finqa/train.parquet", finqa_record, "id"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    args = ap.parse_args()

    out_dir = ROOT / "data" / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.domains:
        path, fn, id_col = DOMAINS[name]
        df = pd.read_parquet(ROOT / "data" / path)
        picked = df.sample(n=args.m, random_state=args.seed)
        with open(out_dir / f"{name}.jsonl", "w") as fh, open(out_dir / f"{name}_sft.jsonl", "w") as sft:
            for idx, row in picked.iterrows():
                rec = {"id": f"{name}-{row[id_col] if id_col else idx}", **fn(row)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sft.write(json.dumps({"uid": rec["id"], "instruction": rec["pangea_query"],
                                      "output": rec["answer"]}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(picked)} seeds from {len(df)} -> {out_dir / f'{name}.jsonl'}")
    (out_dir / "SOURCE.json").write_text(json.dumps(
        {"m": args.m, "random_state": args.seed, "splits": {k: v[0] for k, v in DOMAINS.items()}},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
