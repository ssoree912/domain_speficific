"""Skeleton-first GSM8K generation: a verified math skeleton, then the question text around it.

  python scripts/skeleton_gen.py skeleton  --n 64     # Qwen: (general, seed) -> JSON skeleton, executed and checked
  python scripts/skeleton_gen.py write --gen qwen     # Qwen AR writes the question from the skeleton
  python scripts/skeleton_gen.py write --gen dream    # Dream fills a canvas with the skeleton's phrases anchored
  python scripts/skeleton_gen.py annotate --gen qwen|dream
  python scripts/skeleton_gen.py report

Skeleton (one Qwen call per (general example, seed) pair of data/synth/plan/gsm8k_n1000.jsonl):
  {"setting": ..., "given": [{"var", "value", "phrase"}], "program": ["b = a * 3", ..., "answer = ..."],
   "question": "...?"}
The program is executed, so every kept skeleton has a known answer. The number of steps asked
for is drawn per row from the GSM8K test gold solutions (<<...>> calculations). Both writers get
the same skeleton and the same prompt; Dream additionally has the given phrases and the final
question fixed on its canvas. The labeler is the pipeline's (Qwen, PANGEA annotation prompt),
and its answer can be checked against the skeleton's.

Output: data/synth/skel/gsm8k/n<N>/{skeleton,write_<gen>,annotate_<gen>}.jsonl"""
import argparse, json, math, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchors
from pangea_gen import DREAM, PREFIX, PROMPTS, QWEN, SAMPLING

SKELETON_PROMPT = '''You design the math skeleton of a GSM8K-style word problem.

General example (take the setting, the people or objects and the context from it; do not solve it):
"""
{general}
"""

Reference GSM8K problem (for difficulty only; do not copy its content):
"""
{seed}
"""

Requirements:
- Solving the problem must take exactly {k} arithmetic steps.
- Use 2 to 5 given quantities. Each is a concrete number with a unit or object that fits the setting.
- No given quantity may be the answer or an intermediate result.
- Every number the program uses must be one of the given quantities. A relation such as "twice as many",
  "half of" or "3 times" is a given quantity too (value 2 with phrase "twice as many", value 0.5 with phrase "half").
- The setting must use the people, objects or place of the general example.
- The final question asks for one number.

Output only this JSON:
{{"setting": "<one sentence>",
 "given": [{{"var": "a", "value": <number>, "phrase": "<the number with its unit, e.g. 12 chocolate eggs>"}}, ...],
 "program": ["<var> = <expression>", ..., "answer = <expression>"],
 "question": "<the final question, ending with ?>"}}
Each program line assigns one new variable from given variables and earlier variables{literals}; the last line assigns answer.'''

SOLVE_PROMPT = ("Solve the following math word problem step by step. "
                "End with a final line of the form \"The answer is: <number>\".\n\nProblem: {q}")

WRITE_PROMPT = '''Write a GSM8K-style math word problem from this skeleton.

Setting: {setting}
Given quantities (state each one exactly once, as written): {phrases}
Final question (end the problem with exactly this): {question}

Rules:
- 2 to 4 sentences.
- Do not state any computed value or the answer.
- Do not use variable names, symbols or equations.
- Output only the problem.'''

WRITE_PREFIX = "Problem:"
NUM = r"-?\d[\d,]*(?:\.\d+)?"
TAG = ""
OUT = lambda n: ROOT / "data" / "synth" / f"skel{('_' + TAG) if TAG else ''}" / "gsm8k" / f"n{n}"
# number words a given phrase may use instead of digits (v2: relations are given quantities)
WORDS = {2: ["twice", "double", "two"], 3: ["three", "triple", "thrice"], 4: ["four", "quadruple"],
         0.5: ["half"], 0.25: ["quarter"], 1 / 3: ["third"], 10: ["ten"], 12: ["dozen"]}


def read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def step_counts():
    import pandas as pd
    gold = pd.read_parquet(ROOT / "data/eval/gsm8k/test.parquet")
    return [max(2, min(6, len(re.findall(r"<<", a)))) for a in gold.answer]


def fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


V2 = False
V3 = False  # implies V2 checks
STOP = {"there", "their", "which", "about", "would", "these", "other", "where", "after", "being", "every",
        "total", "number", "people", "question", "answer", "following", "options"}


def check(sk, k, general=""):
    """Execute and validate a skeleton; returns (ok, reason, answer)."""
    try:
        given = sk["given"]
        prog, q = sk["program"], sk["question"].strip()
        if not (2 <= len(given) <= 5) or not prog or not q.endswith("?"):
            return False, "shape", None
        env = {}
        for g in given:
            v = float(g["value"])
            words = [w for k_, ws in WORDS.items() if abs(k_ - v) < 1e-6 for w in ws]
            in_phrase = (any(abs(float(x) - v) < 1e-9 for x in re.findall(r"\d+(?:\.\d+)?", g["phrase"].replace(",", "")))
                         if V3 else fmt(v) in g["phrase"].replace(",", ""))
            if not re.fullmatch(r"[a-z]\w*", g["var"]) or (not in_phrase and not any(w in g["phrase"].lower() for w in words)):
                return False, "given", None
            env[g["var"]] = v
        lines = [l for l in prog if l.strip()]
        for line in lines:
            m = re.fullmatch(r"\s*([a-z_]\w*)\s*=\s*([\w\s.+\-*/()]+)", line)
            if not m:
                return False, "program_syntax", None
            env[m.group(1)] = float(eval(m.group(2), {"__builtins__": {}}, dict(env)))
        ans = env.get("answer")
        if ans is None or not math.isfinite(ans) or ans <= 0:
            return False, "answer_value", None
        if abs(ans - round(ans, 2)) > 1e-9:
            return False, "answer_not_round", None
        if any(abs(ans - float(g["value"])) < 1e-9 for g in given):
            return False, "answer_is_given", None
        used = " ".join(l.split("=", 1)[1] for l in lines)
        if any(not re.search(rf"\b{g['var']}\b", used) for g in given):
            return False, "given_unused", None
        if len(lines) < max(2, k - 1):
            return False, "too_few_steps", None
        if V2 or V3:
            if len(q.split()) < 4:
                return False, "question_short", None
            pct = any("%" in g["phrase"] or "percent" in g["phrase"].lower() for g in given)
            for lit in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", used):
                L_ = float(lit)
                if V3 and not (L_ == 1 or (L_ == 100 and pct)):
                    return False, "numeric_literal", None
                if not (any(abs(L_ - float(g["value"])) < 1e-9 for g in given) or L_ == 1 or (L_ == 100 and pct)):
                    return False, "hidden_constant", None
            gw = lambda t: set(re.findall(r"[a-z]{5,}", t.lower())) - STOP
            if not (gw(sk.get("setting", "")) & gw(general)):
                return False, "ungrounded_setting", None
        if re.search(rf"(?<![\d.]){re.escape(fmt(ans))}(?![\d.])", q):
            return False, "answer_in_question", None
        return True, "ok", round(ans, 2)
    except Exception as e:  # malformed JSON fields, eval errors (division by zero, names)
        return False, f"error:{type(e).__name__}", None


def solved_answer(text):
    m = re.findall(r"answer is[:\s]*\**\s*\$?\s*(" + NUM + ")", text, re.I) or re.findall(r"\\boxed\{\s*\$?(" + NUM + ")", text)
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


def stage_verify(args):
    """Greedy independent solve of each written question; agreement with the program = math preserved."""
    from llm import HFChat
    llm = HFChat(str(QWEN), batch_size=32)
    for gen in args.gens:
        qs = read(OUT(args.n) / f"write_{gen}.jsonl")
        outs = llm.chat([[{"role": "user", "content": SOLVE_PROMPT.format(q=q["question"])}] for q in qs], max_new_tokens=768)
        with open(OUT(args.n) / f"verify_{gen}.jsonl", "w") as fh:
            for q, o in zip(qs, outs):
                fh.write(json.dumps({"uid": q["uid"], "solve": o, "solve_answer": solved_answer(o)}) + "\n")
        print(f"verify {gen}: {len(qs)}")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def stage_skeleton(args):
    from llm import HFChat
    plan = read(ROOT / "data/synth/plan/gsm8k_n1000.jsonl")
    rng = random.Random(0)
    ks = step_counts()
    rows = [{**r, "k": rng.choice(ks)} for r in plan[: args.n * args.cand]]
    llm = HFChat(str(QWEN), batch_size=16)
    convs = [[{"role": "user", "content": SKELETON_PROMPT.format(general=r["general"], seed=r["seed"], k=r["k"],
                       literals=" only (no number literals except 1, and 100 for a percent)" if V3 else ' and small constants (like 2 for "twice")')}]
             for r in rows]
    outs = llm.chat(convs, max_new_tokens=512, do_sample=True, **SAMPLING)
    kept, reasons = [], {}
    for r, o in zip(rows, outs):
        sk = parse_json(o)
        ok, why, ans = check(sk, r["k"], r["general"]) if sk else (False, "json", None)
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            kept.append({"uid": r["uid"], "general": r["general"], "seed": r["seed"], "k": r["k"],
                         "skeleton": sk, "answer": ans, "raw": o})
    if V3:  # the skeleton must be self-consistent: a plain rendering, solved greedily, gives the program's answer
        plain = [f"{r['skeleton']['setting'].strip()} Given: " + "; ".join(g_["phrase"] for g_ in r["skeleton"]["given"])
                 + ". " + r["skeleton"]["question"].strip() for r in kept]
        sol = llm.chat([[{"role": "user", "content": SOLVE_PROMPT.format(q=p_)}] for p_ in plain], max_new_tokens=768)
        consistent = []
        for r, p_, o in zip(kept, plain, sol):
            a_ = solved_answer(o)
            if a_ is not None and abs(a_ - r["answer"]) <= 1e-4 * max(1, abs(r["answer"])):
                consistent.append({**r, "plain": p_})
        reasons["ok"] = len(consistent)
        reasons["inconsistent_skeleton"] = len(kept) - len(consistent)
        kept = consistent
    kept = kept[: args.n]
    OUT(args.n).mkdir(parents=True, exist_ok=True)
    with open(OUT(args.n) / "skeleton.jsonl", "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"candidates": len(rows), "kept": len(kept), "reasons": reasons},
              open(OUT(args.n) / "skeleton_stats.json", "w"), indent=1)
    print(f"skeletons: {len(kept)} kept of {len(rows)}; {reasons}")


def write_conv(sk):
    phrases = "; ".join(g["phrase"] for g in sk["given"])
    return [{"role": "user", "content": WRITE_PROMPT.format(setting=sk["setting"], phrases=phrases,
                                                            question=sk["question"].strip())}]


def stage_write(args):
    rows = read(OUT(args.n) / "skeleton.jsonl")
    convs = [write_conv(r["skeleton"]) for r in rows]
    if args.gen == "qwen":
        from llm import HFChat
        gen = HFChat(str(QWEN), batch_size=16)
        outs = gen.chat(convs, max_new_tokens=256, assistant_prefix=WRITE_PREFIX, do_sample=True, **SAMPLING)
        outs = [o.strip() for o in outs]
    else:
        from dream_chat import DreamChat
        gen = DreamChat(str(DREAM), batch_size=4)
        n_tok = lambda t: len(gen.anchor_ids(t))
        plans = []
        for r in rows:
            texts = [" " + g["phrase"] for g in r["skeleton"]["given"]]
            span = max(4, args.budget // (len(texts) + 1))
            plans.append([(t, span) for t in texts] + [(" " + r["skeleton"]["question"].strip(), span)])
        canv = [anchors.canvas(p, n_tok) for p in plans]
        outs = [None] * len(rows)
        for L in sorted({L for _, L in canv}):
            idx = [i for i, (_, l) in enumerate(canv) if l == L]
            res = gen.chat([convs[i] for i in idx], gen_length=L, steps=L, assistant_prefix=WRITE_PREFIX,
                           anchors=[canv[i][0] for i in idx], ban=["="], **SAMPLING)
            for i, o in zip(idx, res):
                outs[i] = anchors.clean(o)
    with open(OUT(args.n) / f"write_{args.gen}.jsonl", "w") as fh:
        for r, o in zip(rows, outs):
            fh.write(json.dumps({"uid": r["uid"], "question": o, "generator": gen.name}, ensure_ascii=False) + "\n")
    print(f"write {args.gen}: {len(rows)}")


def stage_annotate(args):
    from llm import HFChat
    qs = read(OUT(args.n) / f"write_{args.gen}.jsonl")
    template = (PROMPTS / "gsm8k_annotation.txt").read_text()
    llm = HFChat(str(QWEN), batch_size=16)
    outs = llm.chat([[{"role": "user", "content": template.format(question=q["question"])}] for q in qs],
                    max_new_tokens=1536, assistant_prefix=PREFIX["annotation"]["gsm8k"], do_sample=True, **SAMPLING)
    with open(OUT(args.n) / f"annotate_{args.gen}.jsonl", "w") as fh:
        for q, o in zip(qs, outs):
            fh.write(json.dumps({"uid": q["uid"], "instruction": q["question"], "output": o}, ensure_ascii=False) + "\n")
    print(f"annotate {args.gen}: {len(qs)}")


def stage_report(args):
    sys.argv = ["x", "a=b"]
    g = {"__file__": str(ROOT / "scripts/probes/structure_metrics.py")}
    exec((ROOT / "scripts/probes/structure_metrics.py").read_text().split("ap = argparse")[0], g)
    sk = {r["uid"]: r for r in read(OUT(args.n) / "skeleton.jsonl")}
    print(json.load(open(OUT(args.n) / "skeleton_stats.json")))
    print("| metric | " + " | ".join(args.gens) + " |\n|---|" + "---|" * len(args.gens))
    res = {}
    for gen in args.gens:
        ann = read(OUT(args.n) / f"annotate_{gen}.jsonl")
        n = len(ann)
        m = {}
        def frac(f): return sum(map(bool, map(f, ann))) / n
        q = lambda r: r["instruction"]
        s = lambda r: sk[r["uid"]]
        m["all given phrases present"] = frac(lambda r: all(g_["phrase"].lower() in q(r).lower() for g_ in s(r)["skeleton"]["given"]))
        m["all given numbers present"] = frac(lambda r: all(re.search(rf"(?<![\d.]){re.escape(fmt(float(g_['value'])))}(?![\d.])", q(r).replace(",", "")) for g_ in s(r)["skeleton"]["given"]))
        m["ends with skeleton question"] = frac(lambda r: q(r).strip().lower().endswith(s(r)["skeleton"]["question"].strip().lower()))
        m["answer leaked"] = frac(lambda r: re.search(rf"(?<![\d.]){re.escape(fmt(s(r)['answer']))}(?![\d.])", q(r).replace(",", "")))
        m["symbol/equation leak"] = frac(lambda r: re.search(r"\b[a-e]\s*=|\b[B-E]\b(?!\.)|=", q(r)))
        m["2-4 sentences"] = frac(lambda r: 2 <= len(re.findall(r"[.?!](?:\s|$)", q(r))) <= 4)
        m["label = true answer"] = frac(lambda r: g["same"](g["final_answer"](r["output"]), s(r)["answer"]))
        ver = {v["uid"]: v["solve_answer"] for v in read(OUT(args.n) / f"verify_{gen}.jsonl")}
        if ver:
            m["verified (solve = program)"] = frac(lambda r: g["same"](ver.get(r["uid"]), s(r)["answer"]))
            m["verified & label correct"] = frac(lambda r: g["same"](ver.get(r["uid"]), s(r)["answer"])
                                                 and g["same"](g["final_answer"](r["output"]), s(r)["answer"]))
        grams = [t for r in ann for t in (lambda w: [tuple(w[i:i + 3]) for i in range(len(w) - 2)])(re.findall(r"\w+", q(r).lower()))]
        m["distinct 3-gram"] = len(set(grams)) / max(1, len(grams))
        gw = lambda t: set(re.findall(r"[a-z]{5,}", t.lower()))
        m["shares word w/ general"] = frac(lambda r: gw(q(r)) & gw(s(r)["general"]))
        m["words (median)"] = sorted(len(q(r).split()) for r in ann)[n // 2]
        res[gen] = m
    for k in res[args.gens[0]]:
        print(f"| {k} | " + " | ".join(f"{res[g_][k]:.2f}" if isinstance(res[g_][k], float) else str(res[g_][k]) for g_ in args.gens) + " |")
    st = g["structure"]([g["calcs"](r["output"]) for r in read(OUT(args.n) / f"annotate_{args.gens[0]}.jsonl")])
    ks = [r["k"] for r in sk.values()]
    print(f"\nskeleton steps asked: median {sorted(ks)[len(ks) // 2]}, >=4: {sum(k >= 4 for k in ks) / len(ks):.2f}; "
          f"program lines median {sorted(len(r['skeleton']['program']) for r in sk.values())[len(sk) // 2]}")


ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["skeleton", "write", "annotate", "verify", "report"])
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--gen", choices=["qwen", "dream"])
ap.add_argument("--gens", nargs="+", default=["qwen", "dream"])
ap.add_argument("--tag", default="", help="output dir suffix: data/synth/skel_<tag>/")
ap.add_argument("--v2", action="store_true", help="no hidden constants, question >= 4 words, setting grounded in the general example")
ap.add_argument("--v3", action="store_true", help="v2 + no number literals, numeric phrase check, self-consistent skeletons")
ap.add_argument("--cand", type=int, default=2, help="candidate skeletons per kept row")
ap.add_argument("--budget", type=int, default=64, help="Dream: canvas tokens shared by the spans between anchors")
args = ap.parse_args()
TAG, V2, V3 = args.tag, args.v2 or args.v3, args.v3
{"skeleton": stage_skeleton, "write": stage_write, "annotate": stage_annotate, "verify": stage_verify,
 "report": stage_report}[args.stage](args)
