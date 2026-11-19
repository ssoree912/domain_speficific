"""PANGEA's LLM answer-parsing step on lm-eval generations (pangea_<dataset> tasks).

Port of third_party/pangea/code/test_bench/data_parsing_{math,med,fin}.py. A chat model
reads each generation through PANGEA's extraction template (template/extract_*.txt) with
the assistant turn pre-filled by "\\n- **Model's Final Answer is:** ", and its short reply
is compared to the ground truth with PANGEA's rules: GSM8K takes the first number, rounded
to an int (a failed parse counts as 0); MedQA compares the text before the first "." with
the gold letter; FinQA cleans the reply (markdown, currency, %, negative hints), then
accepts a 100x percent/decimal mismatch, an absolute error <= 1 for |values| >= 1 and
<= 0.01 otherwise, also when only the sign differs.

The parser sampling is PANGEA's (GSM8K / FinQA: T=0.6, top_p=0.95, top_k=20, 10 new
tokens; MedQA: T=0.9, top_p=0.7, top_k=10, 100 new tokens; repetition penalty 1, which
also overrides Qwen's generation_config 1.05), and the scripts seed with
seed_everything(666) in __main__ (111 is only the function's default). HF sampling cannot
reproduce vLLM's random draws, so expect run-to-run noise of a few tenths of a point.
PANGEA's parsers were Llama-3.3-70B-Instruct (GSM8K, FinQA) and Llama-3.1-8B-Instruct
(MedQA); the default here is the local Qwen2.5-7B-Instruct.

    python scripts/pangea_parse.py results/student/Llama-3.2-1B/gsm8k/len1024

Reads the newest samples_pangea_*.jsonl under the given path (or the file itself) and
writes parsed_<...>.jsonl (per sample) and pangea_acc_<...>.json next to it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "pangea"))
from pangea_utils import ground_truth  # noqa: E402

TEMPLATE_DIR = os.path.join(ROOT, "third_party/pangea/code/test_bench/template")
TEMPLATES = {"gsm8k": "extract_template_math.txt", "medqa": "extract_answer_med.txt",
             "finqa": "extract_template_fin.txt"}
ASSISTANT_PREFIX = "\n- **Model's Final Answer is:** "
_NUMERIC = dict(max_new_tokens=10, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                repetition_penalty=1.0)
SAMPLING = {"gsm8k": _NUMERIC, "finqa": _NUMERIC,
            "medqa": dict(max_new_tokens=100, do_sample=True, temperature=0.9, top_p=0.7,
                          top_k=10, repetition_penalty=1.0)}


def seed_everything(seed):
    import numpy as np
    import torch
    from transformers import set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parser_conversation(template, generation):
    return [{"role": "user", "content": template.format(answer_text=generation.strip())}]


# --- GSM8K (data_parsing_math.py) ---

def extract_math(completion):
    match = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", completion.strip())
    if match:
        num_str = match.group()
        try:
            return round(float(Fraction(num_str))) if "/" in num_str else round(float(num_str))
        except Exception:
            return None
    return None


def score_gsm8k(reply, gt):
    value = extract_math(reply.replace(",", "").strip())
    value = 0 if value is None else int(value)
    return {"extracted": value, "correct": value == int(gt)}


# --- MedQA (data_parsing_med.py) ---

def score_medqa(reply, gt):
    value = reply.split(".")[0].strip()
    return {"extracted": value, "correct": value == gt.strip()}


# --- FinQA (data_parsing_fin.py) ---

def preprocess_fin(text):
    text = re.sub(r"[\*\`\#\~]+", "", text)
    text = re.sub(r"[$€¥£]", "", text)
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    has_decrease_hint = any(h in text.lower() for h in [
        "decreas", "loss", "deficit", "negativ", "fell", "decline", "drop", "reduc", "below"])
    percent_match = re.search(r"([-+]?\d+\.?\d*)\s*%", text)
    if percent_match:
        percent_value = float(percent_match.group(1)) / 100
        text = re.sub(r"([-+]?\d+\.?\d*)\s*%", str(percent_value), text)
    if has_decrease_hint:
        match = re.search(r"(?<!\-)\b\d+\.?\d*\b", text)
        if match and "increase" not in text.lower() and "growth" not in text.lower():
            text += " (potential negative number)"
    return text.strip()


def _to_number(num_str):
    num_str = num_str.replace(",", "")
    return float(Fraction(num_str)) if "/" in num_str else float(num_str)


def extract_fin(completion):
    """PANGEA computes a negative hint here but always keeps the parsed sign, so it is omitted."""
    text = completion.strip().lower()
    final = re.search(r"\*\*model's final answer is:\*\*\s*([-+]?[\d\.,]+(?:/[\d\.,]+)?)",
                      text, re.IGNORECASE)
    if final:
        try:
            return _to_number(final.group(1).strip())
        except Exception:
            pass
    if re.search(r"\*\*model's final answer is:\*\*\s*(yes|y)", text, re.IGNORECASE):
        return "yes"
    if re.search(r"\*\*model's final answer is:\*\*\s*(no|n)", text, re.IGNORECASE):
        return "no"
    if re.search(r"\b(yes|y)\b", text) or "correct" in text:
        return "yes"
    if re.search(r"\b(no|n)\b", text) or "incorrect" in text:
        return "no"
    for pattern in [r"the answer is[:\s]+([-+]?[\d\.,]+)", r"final answer[:\s]+([-+]?[\d\.,]+)",
                    r"answer[:\s]+([-+]?[\d\.,]+)", r"result is[:\s]+([-+]?[\d\.,]+)",
                    r"equals[:\s]+([-+]?[\d\.,]+)", r"comes to[:\s]+([-+]?[\d\.,]+)",
                    r"^\s*([-+]?[\d\.,]+)\s*$", r"([-+]?[\d\.,]+)\s*%",
                    r"([-+]?[\d\.,]+)\s*million", r"([-+]?[\d\.,]+)\s*billion",
                    r"is\s*([-+]?[\d\.,]+)"]:
        match = re.search(pattern, text)
        if match:
            try:
                return _to_number(match.group(1))
            except Exception:
                continue
    match = re.search(r"[-+]?[\d\.,]+(?:/[\d\.,]+)?", text)
    if match:
        try:
            return _to_number(match.group())
        except Exception:
            return None
    return None


def fin_gt(gt):
    """GT as data_parsing_fin.py reads it back from eval_finqa.py's CSV."""
    if not isinstance(gt, str):
        return gt
    gt = gt.strip().lower()
    if gt in ("yes", "no"):
        return gt
    try:
        return float(gt.replace(",", ""))
    except ValueError:
        return gt


def compare_fin(gen_val, gt_val):
    if gen_val is None:
        return False
    if isinstance(gen_val, str) and isinstance(gt_val, str):
        return gen_val.lower() == gt_val.lower()
    if not (isinstance(gen_val, (int, float)) or isinstance(gt_val, (int, float))):
        return False
    try:
        gen, gt = float(gen_val), float(gt_val)
    except (ValueError, TypeError):
        return False
    if abs(gen) > abs(gt):
        ratio = abs(gen / gt) if gt != 0 else float("inf")
        if 98 <= ratio <= 102:
            gen = gen / 100
    elif abs(gt) > abs(gen):
        ratio = abs(gt / gen) if gen != 0 else float("inf")
        if 98 <= ratio <= 102:
            gt = gt / 100
    if (gen > 0 > gt) or (gen < 0 < gt):
        gen, gt = abs(gen), abs(gt)
    if abs(gen) >= 1 and abs(gt) >= 1:
        return abs(gen - gt) <= 1.0
    return abs(gen - gt) <= 0.01


def score_finqa(reply, gt):
    cleaned = preprocess_fin(reply)
    value = extract_fin(cleaned)
    return {"cleaned": cleaned, "extracted": value, "correct": compare_fin(value, fin_gt(gt))}


SCORE = {"gsm8k": score_gsm8k, "medqa": score_medqa, "finqa": score_finqa}


def parse(samples, dataset, llm, batch=None):
    """Parser replies and PANGEA scores for lm-eval sample records (sorted by doc_id)."""
    with open(os.path.join(TEMPLATE_DIR, TEMPLATES[dataset]), encoding="utf-8") as f:
        template = f.read()
    gens = [s["resps"][0][0] or "" for s in samples]
    replies = llm.chat([parser_conversation(template, g) for g in gens],
                       assistant_prefix=ASSISTANT_PREFIX, **SAMPLING[dataset])
    rows = []
    for s, g, r in zip(samples, gens, replies):
        gt = ground_truth(dataset, s["doc"])
        rows.append({"doc_id": s["doc_id"], "ground_truth": gt, "generation": g, "parser_reply": r,
                     **SCORE[dataset](r, gt), "regex_acc": s.get("regex_acc")})
    return rows


def find_samples(path):
    if os.path.isfile(path):
        return path
    found = glob.glob(os.path.join(path, "**", "samples_pangea_*.jsonl"), recursive=True)
    if not found:
        raise SystemExit(f"no samples_pangea_*.jsonl under {path} (run with --log_samples)")
    return max(found, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="run directory (or a samples_pangea_*.jsonl file)")
    ap.add_argument("--parser", default=os.path.join(ROOT, "model", "Qwen2.5-7B-Instruct"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=666)
    args = ap.parse_args()

    samples_path = find_samples(args.path)
    dataset = re.search(r"samples_pangea_(gsm8k|medqa|finqa)_", os.path.basename(samples_path)).group(1)
    with open(samples_path) as f:
        samples = sorted((json.loads(l) for l in f), key=lambda s: s["doc_id"])

    seed_everything(args.seed)
    from llm import HFChat
    rows = parse(samples, dataset, HFChat(args.parser, batch_size=args.batch_size))

    stem = os.path.basename(samples_path)[len("samples_"):-len(".jsonl")]
    out_dir = os.path.dirname(samples_path)
    with open(os.path.join(out_dir, f"parsed_{stem}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    regex = [r["regex_acc"] for r in rows if r["regex_acc"] is not None]
    summary = {"dataset": dataset, "samples": samples_path, "n": len(rows),
               "acc": sum(r["correct"] for r in rows) / len(rows),
               "regex_acc": sum(regex) / len(regex) if regex else None,
               "parser": args.parser, "seed": args.seed, "sampling": SAMPLING[dataset],
               "assistant_prefix": ASSISTANT_PREFIX}
    with open(os.path.join(out_dir, f"pangea_acc_{stem}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("dataset", "n", "acc", "regex_acc")}))


if __name__ == "__main__":
    main()
