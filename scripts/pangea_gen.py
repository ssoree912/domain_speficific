#!/usr/bin/env python
"""PANGEA's three generation stages, with the generator as the only per-condition variable.

  plan        sample n general items d_g (filtered CoT-Collection) and pair each with a random
              seed d_s; written once per (domain, n) and shared by every condition
  writer      d_g -> Synth-Guide Block τ                (generator of the condition)
  projection  τ + seed query -> synthetic question q̃    (generator of the condition)
  annotation  q̃ -> response                             (shared labeler, every condition)
  export      (q̃, response) -> train.jsonl {instruction, output} for scripts/train_student.py

Conditions (docs/PLAN.md):
  g1  Qwen2.5-7B-Instruct, autoregressive
  g2  Dream-v0-Instruct-7B, fixed canvas without anchors
  g3  Dream-v0-Instruct-7B, anchored canvas (scripts/anchors.py; GSM8K only so far)

Prompts are PANGEA's (pipeline/prompts), each reply primed with the stage's assistant prefix
as in the official code, sampled with PANGEA's T 0.9 / top-p 0.7 / top-k 10 for Qwen and
T 0.2 / top-p 0.95 for Dream. Stages append
to <out>/<stage>.jsonl and skip uids already written, so reruns resume.

  python scripts/pangea_gen.py plan --domain medqa --n 10000
  python scripts/pangea_gen.py writer --cond g1 --domain medqa --n 10000
  python scripts/pangea_gen.py projection --cond g1 --domain medqa --n 10000
  python scripts/pangea_gen.py annotation --cond g1 --domain medqa --n 10000
  python scripts/pangea_gen.py export --cond g1 --domain medqa --n 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPTS = ROOT / "pipeline" / "prompts"
QWEN = ROOT / "model" / "Qwen2.5-7B-Instruct"
DREAM = ROOT / "model" / "Dream-v0-Instruct-7B"
LABELER = QWEN

# Assistant prefixes: the official MedQA code primes each reply with the start of the
# required output format; GSM8K / FinQA use their prompts' required first line likewise.
PREFIX = {
    "writer": {"medqa": "Clinical Building Blocks:", "gsm8k": "Quantitative Building Blocks:",
               "finqa": "Financial Scenario:"},
    "projection": {"medqa": "**New Clinical Question:**\n", "gsm8k": "Transformed Domain Question:",
                   "finqa": "New Financial Context and Question:"},
    "annotation": {"medqa": "- **Answer**: ", "gsm8k": "- **Answer**: ", "finqa": "- **Answer**: "},
}
SAMPLING = dict(temperature=0.9, top_p=0.7, top_k=10)
# Dream (g2, g3): with PANGEA's sampling Dream copies the prompt's format placeholders or emits
# an empty canvas (docs/PLAN.md, pilot), so both Dream conditions use Dream's recommended
# low temperature instead (--dream-sampling low). With anchors (g3) PANGEA's sampling no longer
# collapses, and the low temperature cost diversity in the GSM8K 1k run (--dream-sampling pangea).
DREAM_SAMPLINGS = {"low": dict(temperature=0.2, top_p=0.95, top_k=None), "pangea": SAMPLING}
DREAM_SAMPLING = DREAM_SAMPLINGS["low"]
# AR length caps (PANGEA: 4096 for every stage) and Dream canvas lengths L per stage.
MAX_NEW = {"writer": 768, "projection": 1024, "annotation": 1536}
CANVAS = {"writer": {"medqa": 256, "gsm8k": 256, "finqa": 256},
          "projection": {"medqa": 384, "gsm8k": 160, "finqa": 768}}


def out_dir(args, cond=None):
    name = (cond or args.cond) + (f"_{args.tag}" if args.tag else "")
    d = ROOT / "data" / "synth" / name / args.domain / f"n{args.n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plan_path(args):
    return ROOT / "data" / "synth" / "plan" / f"{args.domain}_n{args.n}.jsonl"


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def make_plan(args):
    path = plan_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"{path} exists")
        return
    gen = pd.read_parquet(ROOT / "data" / "general" / "cot_collection" / args.general,
                          columns=["id", "source"])
    picked = gen.sample(n=args.n, random_state=args.seed)
    seeds = read_jsonl(ROOT / "data" / "seeds" / f"{args.domain}.jsonl")
    rng = random.Random(args.seed)
    with open(path, "w") as fh:
        for k, r in enumerate(picked.itertuples()):
            s = rng.choice(seeds)
            fh.write(json.dumps({"uid": f"{args.domain}-{k:06d}", "dg_id": r.id, "general": r.source,
                                 "seed_id": s["id"], "seed": s["pangea_query"]}, ensure_ascii=False) + "\n")
    print(f"plan: {args.n} (d_g, seed) pairs from {len(gen)} general items -> {path}")


def generator(args, stage):
    if stage == "annotation" or args.cond == "g1":
        from llm import HFChat
        return HFChat(str(LABELER if stage == "annotation" else QWEN), batch_size=args.batch_size)
    from dream_chat import DreamChat
    return DreamChat(str(DREAM), batch_size=args.batch_size)


def dream_g3(gen, args, stage, rows, convs, prefix):
    """Anchored canvases; Dream batches one canvas length at a time, so group rows by L."""
    import anchors
    n_tokens = lambda t: len(gen.anchor_ids(t))
    plans = [anchors.writer_plan(args.domain) if stage == "writer"
             else anchors.projection_plan(args.domain, r["writer"], r["general"], args.projection_anchors)
             for r in rows]
    canv = [anchors.canvas(p, n_tokens) for p in plans]
    outs = [None] * len(rows)
    for L in sorted({L for _, L in canv}):
        idx = [i for i, (_, l) in enumerate(canv) if l == L]
        # PANGEA's projection rule "no intermediate calculations": with τ's values anchored,
        # Dream otherwise copies τ's "A = 100 people" notation into half the questions
        ban = ["="] if stage == "projection" and args.projection_anchors == "rich" else None
        # contrastive CFG against the same prompt with τ masked (projection only)
        cfg = (dict(cfg_w=args.cfg_w, cfg_late=args.cfg_late, cfg_texts=[rows[i]["writer"].strip() for i in idx])
               if stage == "projection" and args.cfg_w else {})
        res = gen.chat([convs[i] for i in idx], gen_length=L, steps=args.steps or L,
                       assistant_prefix=prefix, anchors=[canv[i][0] for i in idx], ban=ban,
                       **DREAM_SAMPLING, **cfg)
        for i, o in zip(idx, res):
            outs[i] = anchors.clean(o) if stage == "projection" else o
    return outs, [[t for t, _ in p if t] for p in plans]


def run_stage(args, stage):
    template = (PROMPTS / f"{args.domain}_{stage}.txt").read_text()
    if stage == "writer":
        src = read_jsonl(plan_path(args))
        fill = lambda r: template.format(general=r["general"])
    elif stage == "projection":
        wdir = (ROOT / "data" / "synth" / args.writer_from / args.domain / f"n{args.n}"
                if args.writer_from else out_dir(args))
        src = read_jsonl(wdir / "writer.jsonl")
        fill = lambda r: template.format(block=r["writer"].strip(), seed=r["seed"])
    else:
        src = read_jsonl(out_dir(args) / "projection.jsonl")
        fill = lambda r: template.format(question=r["projection"].strip())
    if not src:
        raise SystemExit(f"nothing to do: input for {stage} is missing")
    dst = out_dir(args) / f"{stage}.jsonl"
    done = {r["uid"] for r in read_jsonl(dst)}
    todo = [r for r in src if r["uid"] not in done][: args.limit or None]
    print(f"{args.cond}/{args.domain}/{stage}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    gen = generator(args, stage)
    prefix = PREFIX[stage][args.domain]
    with open(dst, "a") as fh:
        for s in range(0, len(todo), args.chunk):
            rows = todo[s:s + args.chunk]
            convs = [[{"role": "user", "content": fill(r)}] for r in rows]
            extra = [{}] * len(rows)
            if args.cond == "g3" and stage != "annotation":
                outs, used = dream_g3(gen, args, stage, rows, convs, prefix)
                extra = [{f"{stage}_anchors": u} for u in used]
            elif hasattr(gen, "mask_id"):  # Dream without anchors (g2)
                outs = gen.chat(convs, gen_length=args.canvas or CANVAS[stage][args.domain],
                                steps=args.steps, assistant_prefix=prefix, **DREAM_SAMPLING)
            else:
                outs = gen.chat(convs, max_new_tokens=MAX_NEW[stage], assistant_prefix=prefix,
                                do_sample=True, repetition_penalty=1.0, **SAMPLING)
            for r, o, e in zip(rows, outs, extra):
                fh.write(json.dumps({**r, stage: o, f"{stage}_generator": gen.name, **e},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {s + len(rows)}/{len(todo)}", flush=True)


def export(args):
    rows = read_jsonl(out_dir(args) / "annotation.jsonl")
    kept = [r for r in rows if r["projection"].strip() and r["annotation"].strip()]
    path = out_dir(args) / "train.jsonl"
    with open(path, "w") as fh:
        for r in kept:
            fh.write(json.dumps({"uid": r["uid"], "seed_id": r["seed_id"],
                                 "instruction": r["projection"].strip(),
                                 "output": r["annotation"].strip()}, ensure_ascii=False) + "\n")
    print(f"export: {len(kept)}/{len(rows)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "writer", "projection", "annotation", "export"])
    ap.add_argument("--domain", required=True, choices=["gsm8k", "medqa", "finqa"])
    ap.add_argument("--cond", choices=["g1", "g2", "g3"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--general", default="filtered.parquet", help="file under data/general/cot_collection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="process at most this many pending rows")
    ap.add_argument("--chunk", type=int, default=256, help="rows per write")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--canvas", type=int, help="Dream canvas length L (default per stage/domain)")
    ap.add_argument("--steps", type=int, help="Dream denoising steps (default L)")
    ap.add_argument("--dream-sampling", choices=list(DREAM_SAMPLINGS), default="low")
    ap.add_argument("--tag", help="suffix of the condition's output dir, e.g. t09 -> data/synth/g3_t09")
    ap.add_argument("--writer-from", help="projection: take τ from this condition's dir (e.g. g1), "
                                          "so the conditions share the Writer")
    ap.add_argument("--projection-anchors", choices=["numbers", "rich"], default="numbers",
                    help="g3 projection anchors (scripts/anchors.py)")
    ap.add_argument("--cfg-w", type=float, default=0.0,
                    help="g3 projection: contrastive CFG weight, negative = prompt with τ masked")
    ap.add_argument("--cfg-late", type=float, default=0.5,
                    help="apply CFG once the canvas's remaining mask fraction is at most this (1 = all steps)")
    args = ap.parse_args()
    global DREAM_SAMPLING
    DREAM_SAMPLING = DREAM_SAMPLINGS[args.dream_sampling]
    if args.stage == "plan":
        make_plan(args)
    elif args.cond is None:
        ap.error("--cond is required for this stage")
    elif args.stage == "export":
        export(args)
    else:
        run_stage(args, args.stage)


if __name__ == "__main__":
    main()
