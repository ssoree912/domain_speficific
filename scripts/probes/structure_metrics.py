"""Reasoning-structure and label-quality checks on synthetic GSM8K train sets (CPU only).

  python scripts/probes/structure_metrics.py [--resolve results/label_resolve_gsm8k_n1000.jsonl] name=train.jsonl ...

Per condition:
  label: head answer (first line) vs "The answer is" agree; answer is a number
  structure (from the labeler's solution): arithmetic steps, operation types, operator-sequence
    templates (distinct ratio, entropy); GSM8K test gold (<<a*b=c>> calcs) as reference
  fidelity (rows with a writer τ): question uses τ's given values, invents numbers, leaks the Target value
  resolve (optional): independent Qwen greedy answer equals the label answer"""
import argparse, collections as C, json, math, re, statistics as st
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import anchors

NUM = r"-?\d[\d,]*(?:\.\d+)?"
UNUM = r"\d[\d,]*(?:\.\d+)?"  # unsigned: "16-3-4" is two subtractions
def num(s):
    try: return float(s.replace(",", ""))
    except Exception: return None
def nums(s): return [n for n in (num(x) for x in re.findall(NUM, s)) if n is not None]
def same(a, b): return a is not None and b is not None and abs(a - b) <= 1e-4 * max(1, abs(b))

def final_answer(text):
    m = re.findall(r"answer is[:\s]*\**\s*\$?\s*(" + NUM + ")", text, re.I)
    if m: return num(m[-1])
    b = re.findall(r"\\boxed\{\s*\$?(" + NUM + ")", text)
    return num(b[-1]) if b else None

def head_answer(text):
    first = text.strip().split("\n", 1)[0]
    if re.match(r"\s*(\d+\.\s|\*\*|#)", first) or len(first.split()) > 15:
        return None  # a list item / heading, not a stated answer
    n = nums(first)
    return n[0] if n else None

def normalize(t):
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", t)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("\\div", "/").replace("÷", "/")
    t = re.sub(r"\\text\{[^}]*\}", " ", t)          # units in LaTeX: 3000 \text{ km}
    t = t.replace("\\$", "").replace("$", "").replace("\\%", "%").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"(?<=[\d%)])\s*[A-Za-z][A-Za-z.]*(?:\s+[A-Za-z]+)?\s*(?=[-+*/×]\s*[\d(]|=)", " ", t)  # 3000 km - 1000 km
    return t

TAIL = re.compile(r"[\d\s.,+\-*/x%()]+$")   # arithmetic expression at the end of a segment (linear, no nesting)
HEADNUM = re.compile(r"\s*" + NUM)
def calcs(text):
    """Operator string for each 'a op b ... = c' in the solution, deduplicated by expression."""
    seen, out = set(), []
    for line in normalize(text).split("\n"):
        segs = line.split("=")
        for left, right in zip(segs, segs[1:]):
            m = TAIL.search(left)
            if not m or not HEADNUM.match(right): continue
            e = re.sub(r"\s+", "", m.group(0)).strip(",.")
            if e in seen or not re.search(r"\d", e): continue
            ops = "".join(c for c in re.sub(UNUM, "N", e) if c in "+-*/x%").replace("x", "*")
            if e.startswith("-"): ops = ops[1:]  # a leading minus sign is not an operation
            if ops:
                seen.add(e); out.append(ops)
    return out

def gold_calcs(ans):
    return ["".join(c for c in re.sub(UNUM, "N", e) if c in "+-*/%") for e in re.findall(r"<<([^=>]+)=", ans)]

OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "pct"}
def structure(steps_list):
    n = len(steps_list)
    k = [len(s) for s in steps_list]
    tmpl = C.Counter(">".join(s) for s in steps_list if s)
    tot = sum(tmpl.values())
    ent = -sum(c / tot * math.log2(c / tot) for c in tmpl.values()) if tot else 0
    ops = C.Counter(o for s in steps_list for o in set("".join(s)))
    return {"steps_med": st.median(k), "steps>=4": sum(x >= 4 for x in k) / n, "steps0": sum(x == 0 for x in k) / n,
            "tmpl_distinct": len(tmpl) / max(1, tot), "tmpl_entropy": ent,
            **{f"op_{v}": ops[o] / n for o, v in OPS.items()}}

def given_values(writer):
    """τ's given (not computed) values and its Target value, using the anchor rules."""
    given = [anchors._num(q) for q in anchors.tau_quantities(writer, given_only=True)]
    return [g for g in given if g is not None], anchors.derived_answer(writer)

ap = argparse.ArgumentParser()
ap.add_argument("--resolve")
ap.add_argument("conds", nargs="+")
args = ap.parse_args()
resolved = {}
if args.resolve:
    for r in map(json.loads, open(args.resolve)):
        resolved[(r["cond"], r["uid"])] = final_answer(r["resolve"])

rows_out = {}
gold = pd.read_parquet("data/eval/gsm8k/test.parquet")
rows_out["GSM8K test (gold)"] = {**structure([gold_calcs(a) for a in gold.answer]), "n": len(gold)}

for spec in args.conds:
    name, path = spec.split("=", 1)
    rs = [json.loads(l) for l in open(path)]
    ann = {}
    try:
        ann = {r["uid"]: r for r in map(json.loads, open(path.replace("train.jsonl", "annotation.jsonl")))}
    except FileNotFoundError: pass
    n = len(rs)
    heads = [head_answer(r["output"]) for r in rs]
    finals = [final_answer(r["output"]) for r in rs]
    m = {"n": n,
         "label_numeric": sum(f is not None for f in finals) / n,
         "head_stated": sum(h is not None for h in heads) / n,
         "head_eq_final|stated": sum(same(h, f) for h, f in zip(heads, finals)) / max(1, sum(h is not None for h in heads)),
         **structure([calcs(r["output"]) for r in rs])}
    fid = []
    for r in rs:
        w = ann.get(r["uid"], {}).get("writer")
        if not w: continue
        given, target = given_values(w)
        qn = nums(r["instruction"])
        fid.append((all(any(same(q, g) for q in qn) for g in given) if given else None,
                    any(not any(same(q, g) for g in given) for q in qn) if given else None,
                    target is not None and any(same(q, target) for q in qn) and not any(same(target, g) for g in given)))
    if fid:
        m["fid_all_given_used"] = sum(bool(a) for a, _, _ in fid) / len(fid)
        m["fid_invented_number"] = sum(bool(b) for _, b, _ in fid) / len(fid)
        m["fid_target_leak"] = sum(c for _, _, c in fid) / len(fid)
    if resolved:
        pairs = [(resolved.get((name, r["uid"])), f) for r, f in zip(rs, finals) if (name, r["uid"]) in resolved]
        if pairs:
            m["resolve_n"] = len(pairs)
            num_pairs = [(a, b) for a, b in pairs if b is not None]
            m["resolve_agree"] = sum(same(a, b) for a, b in pairs) / len(pairs)
            m["resolve_agree|label_numeric"] = sum(same(a, b) for a, b in num_pairs) / max(1, len(num_pairs))
            m["usable (calc & numeric & agree)"] = sum(
                same(resolved.get((name, r["uid"])), f) and bool(calcs(r["output"]))
                for r, f in zip(rs, finals)) / n
    rows_out[name] = m

keys = []
for m in rows_out.values():
    keys += [k for k in m if k not in keys]
print("| metric | " + " | ".join(rows_out) + " |")
print("|---|" + "---|" * len(rows_out))
for k in keys:
    vals = [rows_out[c].get(k) for c in rows_out]
    print(f"| {k} | " + " | ".join("–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in vals) + " |")
