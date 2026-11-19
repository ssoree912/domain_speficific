#!/usr/bin/env python
"""Step 1 of the pipeline: from the m seeds of a domain, build the question profile η,
the answer skeleton Φ, and the per-section length budgets ℓ_k.

  1. η and Φ   One LLM call each over a sample of seeds. η follows PANGEA's question
               profile; Φ is 2-4 section names (plus a finer 4-6 variant for the K ablation).
  2. seed → Φ  Every seed answer is put into Φ's sections by the LLM. GSM8K / FinQA seeds
               have a solution, which is only rearranged ("rewrite"); MedQA seeds have only
               the correct option, so a short answer in Φ form is written for it ("write",
               the plan's small trial generation). The last section must hold the gold
               answer; seeds that fail this check are excluded from the statistics.
  3. ℓ_k       Token lengths of each section under the LLaDA and Dream tokenizers. ℓ_k is
               the q-quantile (default 0.9); if anchors + Σℓ_k exceed the canvas L, the
               budgets are scaled down to fit. The uniform split (L - anchors) / K is
               stored next to it for the ℓ_k ablation.

Skeleton variants written per domain: seed (Φ), fine (finer Φ), generic (Step 1 / Step 2
/ Answer) and answer_only (free reasoning, then one Answer anchor), so the H5 and K
ablations share one procedure.

  python scripts/build_profile.py --llm hf:model/Qwen2.5-7B-Instruct
  python scripts/build_profile.py --llm "openai:http://host:8000/v1|Qwen2.5-72B-Instruct" --domains medqa

Outputs, data/profiles/<domain>/:
  profile.json             η, every skeleton with anchor texts, ℓ_k per tokenizer, stats
  seed_<variant>.jsonl     each seed's sections, rendered r̃ and validity
  llm_*.txt                raw LLM replies for η and Φ (rerun skips steps whose files exist)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import load_llm, parse_json  # noqa: E402
from template import ANSWER_ONLY, GENERIC, anchor_texts, render  # noqa: E402

TOKENIZERS = {"llada": ROOT / "model" / "LLaDA-8B-Instruct",
              "dream": ROOT / "model" / "Dream-v0-Instruct-7B"}

DOMAINS = {
    "gsm8k": dict(
        name="grade-school math word problems",
        mode="rewrite",
        seed_answer="a step-by-step solution ending with '#### <final number>'",
        final="the final numeric answer only (a number, no units)"),
    "medqa": dict(
        name="USMLE-style medical exam questions with four options",
        mode="write",
        seed_answer="only the correct option letter and text; there is no explanation, so design "
                    "the skeleton from the question style and the conventions of clinical reasoning",
        final="the correct option as '<letter>. <option text>'"),
    "finqa": dict(
        name="numerical reasoning over financial reports (text and tables)",
        mode="rewrite",
        seed_answer="the evidence sentences from the report, the calculation steps, and the answer",
        final="the final answer value only (number, percentage, or yes/no)"),
}

SYSTEM = "You are an expert in building training data for language models. Reply with JSON only."

ETA_PROMPT = """Below are {n} example questions from one domain ({domain}).

{examples}

Write a profile of these questions that someone could use to write new questions of the same kind from unrelated source material. Reply with a JSON object with these keys:
- "domain": the domain in a few words
- "task_description": what the questions ask the solver to do, in 1-2 sentences
- "question_format": how a question is laid out (parts, order, length, style)
- "context_elements": the kinds of information a question provides (list of strings)
- "required_knowledge": domain knowledge needed to solve them (list of strings)
- "reasoning_skills": reasoning steps needed to solve them (list of strings)
- "answer_format": what form the final answer takes
- "difficulty": typical difficulty and what makes a question harder"""

PHI_PROMPT = """Below are {n} example question-answer pairs from one domain ({domain}). Each answer is {seed_answer}.

{examples}

Design an answer skeleton for this domain: an ordered list of section headers that every good answer in this domain can follow. A model will be given the headers in advance and fill the content under each one, so the headers must fit every question of this domain, not only these examples.

Rules:
- Each header is 1-3 words in Title Case, with no colon, number, or punctuation.
- The last header is where the final answer goes; its content is {final}.
- "anchors": 2-4 headers (a coarse skeleton).
- "fine_anchors": 4-6 headers (a finer skeleton of the same answers; the last one is also the final answer).
- "descriptions": for every header in both lists, one sentence on what goes under it.

Reply with a JSON object with keys "anchors", "fine_anchors", "descriptions"."""

REWRITE_PROMPT = """Question:
{question}

Reference answer:
{answer}

Rearrange the reference answer into the following sections, in order:
{sections}

Rules:
- Use only the content of the reference answer. Do not add new facts, steps, or explanations, and do not drop calculation steps.
- Remove formatting that does not belong to the content (such as '####').
- The last section ("{last}") contains only {final}; the gold final answer is: {gold}
- Every section must be non-empty.

Reply with a JSON object {{"sections": [<one string per section, in order>]}}."""

WRITE_PROMPT = """Question:
{question}

The correct answer is: {answer}

Write a concise expert answer to this question that arrives at the correct answer, using the following sections, in order:
{sections}

Rules:
- Keep it short: what an expert would write on an exam, at most a few sentences per section.
- The last section ("{last}") contains only {final}: {answer}
- Every section must be non-empty.

Reply with a JSON object {{"sections": [<one string per section, in order>]}}."""

GENERIC_DESC = {"Step 1": "The first part of the reasoning.",
                "Step 2": "The rest of the reasoning.",
                "Answer": "The final answer."}
LEAD_DESC = "(no header) The reasoning that leads to the answer."


def load_seeds(domain):
    with open(ROOT / "data" / "seeds" / f"{domain}.jsonl") as fh:
        return [json.loads(l) for l in fh]


def cut(s, n):
    return s if len(s) <= n else s[: n // 2] + "\n...\n" + s[-n // 2:]


def ask_json(llm, prompt, raw_path, max_new_tokens=1536):
    if raw_path.exists():
        reply = raw_path.read_text()
    else:
        reply = llm().chat([[{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}]], max_new_tokens)[0]
        raw_path.write_text(reply)
    obj = parse_json(reply)
    if obj is None:
        raise RuntimeError(f"no JSON in {raw_path}; delete it to retry")
    return obj


def clean_names(names):
    # Drop list numbering ("1. ") and trailing colons the LLM may add despite the rules.
    return [re.sub(r"^\s*\d+[\.\)]\s*", "", str(n)).strip().rstrip(":").strip() for n in names]


def norm(s):
    return re.sub(r"[\s,$]", "", str(s)).lower().rstrip(".")


def gold_ok(domain, last, gold):
    last_n = norm(last)
    if domain == "medqa":
        return re.match(rf"\(?{gold}\b", last.strip()) is not None
    if domain == "gsm8k":
        nums = re.findall(r"-?\d+(?:\.\d+)?", last.replace(",", ""))
        return any(float(x) == float(gold.replace(",", "")) for x in nums)
    return norm(gold) in last_n


def sectionize(llm, domain, seeds, names, lead, desc, cfg):
    items = ([LEAD_DESC] if lead else []) + [f"{n}: {desc.get(n, '')}" for n in names]
    sections = "\n".join(f"{k + 1}. {it}" for k, it in enumerate(items))
    tmpl = REWRITE_PROMPT if cfg["mode"] == "rewrite" else WRITE_PROMPT
    prompts = [tmpl.format(question=cut(s["question"], 4000), answer=s["answer"], gold=s["gold"],
                           sections=sections, last=names[-1], final=cfg["final"]) for s in seeds]
    replies = llm().chat([[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
                          for p in prompts], max_new_tokens=768)
    rows = []
    for s, reply in zip(seeds, replies):
        obj = parse_json(reply) or {}
        secs = obj.get("sections")
        ok = (isinstance(secs, list) and len(secs) == len(names) + lead
              and all(isinstance(c, str) and c.strip() for c in secs))
        ok = ok and gold_ok(domain, secs[-1], s["gold"])
        rows.append({"id": s["id"], "valid": bool(ok), "sections": secs if ok else None,
                     "response": render(names, secs, lead) if ok else None, "raw": reply})
    return rows


def budgets(names, lead, rows, tokenizers, L, q):
    """ℓ per content span (K, or K + 1 with lead, c_0 first)."""
    anchors = anchor_texts(names, lead)
    n_spans = len(names) + lead
    valid = [r["sections"] for r in rows if r["valid"]]
    out = {}
    for tname, tok in tokenizers.items():
        a_len = [len(tok.encode(a, add_special_tokens=False)) for a in anchors]
        lens = np.array([[len(tok.encode(c, add_special_tokens=False)) for c in secs]
                         for secs in valid])
        ell = [int(math.ceil(np.quantile(lens[:, k], q))) for k in range(n_spans)]
        room = L - sum(a_len)
        scaled = sum(ell) > room
        if scaled:
            ell = [max(1, int(e * room / sum(ell))) for e in ell]
        out[tname] = {
            "anchor_tokens": a_len,
            "ell": ell,
            "scaled_to_fit": scaled,
            "uniform_ell": [room // n_spans] * n_spans,
            "section_stats": [{"mean": round(float(lens[:, k].mean()), 1),
                               "median": float(np.median(lens[:, k])),
                               f"p{int(q * 100)}": float(np.quantile(lens[:, k], q)),
                               "max": int(lens[:, k].max())} for k in range(n_spans)],
            "full_response_tokens_p50": float(np.median(lens.sum(1) + sum(a_len))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="hf:<dir> or openai:<base url>|<model>")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    ap.add_argument("--L", type=int, default=256, help="response canvas length, anchors included")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--n-show", type=int, default=20, help="seeds shown in the η / Φ prompts")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizers = {k: AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
                  for k, p in TOKENIZERS.items()}
    _llm = []

    def llm():  # loaded on first use, so reruns that only redo ℓ_k need no GPU
        if not _llm:
            _llm.append(load_llm(args.llm))
        return _llm[0]

    for domain in args.domains:
        cfg = DOMAINS[domain]
        seeds = load_seeds(domain)
        out = ROOT / "data" / "profiles" / domain
        out.mkdir(parents=True, exist_ok=True)
        shown = seeds[: args.n_show]

        q_examples = "\n\n".join(f"### Example {i + 1}\n{cut(s['question'], 1500)}"
                                 for i, s in enumerate(shown))
        eta = ask_json(llm, ETA_PROMPT.format(n=len(shown), domain=cfg["name"], examples=q_examples),
                       out / "llm_eta.txt")

        qa_examples = "\n\n".join(
            f"### Example {i + 1}\nQuestion:\n{cut(s['question'], 1200)}\nAnswer:\n{cut(s['answer'], 800)}"
            for i, s in enumerate(shown))
        phi = ask_json(llm, PHI_PROMPT.format(n=len(shown), domain=cfg["name"], examples=qa_examples,
                                              seed_answer=cfg["seed_answer"], final=cfg["final"]),
                       out / "llm_phi.txt")
        desc = {clean_names([k])[0]: v for k, v in phi.get("descriptions", {}).items()}
        # variant -> (anchor names, lead)
        variants = {"seed": (clean_names(phi["anchors"]), False),
                    "fine": (clean_names(phi["fine_anchors"]), False),
                    "generic": (GENERIC, False), "answer_only": (ANSWER_ONLY, True)}
        desc_all = {**GENERIC_DESC, **desc}

        templates = {}
        for v, (names, lead) in variants.items():
            path = out / f"seed_{v}.jsonl"
            if path.exists():
                rows = [json.loads(l) for l in open(path)]
            else:
                rows = sectionize(llm, domain, seeds, names, lead, desc_all, cfg)
                with open(path, "w") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_valid = sum(r["valid"] for r in rows)
            templates[v] = {
                "anchors": names,
                "lead": lead,
                "anchor_texts": anchor_texts(names, lead),
                "descriptions": {n: desc_all.get(n, "") for n in names},
                "valid_seeds": n_valid,
                "budgets": budgets(names, lead, rows, tokenizers, args.L, args.quantile) if n_valid else None,
            }
            print(f"{domain}/{v}: {names} valid {n_valid}/{len(rows)}")

        profile = {"domain": domain, "llm": args.llm, "L": args.L, "quantile": args.quantile,
                   "n_seeds": len(seeds), "eta": eta, "templates": templates}
        (out / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
        for v, t in templates.items():
            if t["budgets"]:
                b = t["budgets"]["llada"]
                print(f"  {v}: anchors {b['anchor_tokens']} ell {b['ell']} uniform {b['uniform_ell']}"
                      f"{' (scaled)' if b['scaled_to_fit'] else ''}")


if __name__ == "__main__":
    main()
