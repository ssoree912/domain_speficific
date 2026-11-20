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
{program_rule}'''

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
V5 = False  # v4 + steps = arithmetic lines, one operation per line, uniform k, stratified select
V4 = False  # v3 + relaxed rules: uppercase vars, last line is the answer, literals equal to a given value, fraction values
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
            if not re.fullmatch(r"[A-Za-z]\w*" if V4 else r"[a-z]\w*", g["var"]):
                return False, "given_var", None
            if not in_phrase and not any(w in g["phrase"].lower() for w in words):
                return False, f"given_phrase:{g['var']}", None
            env[g["var"]] = v
        lines = [l for l in prog if l.strip()]
        for line in lines:
            m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*([\w\s.+\-*/()]+)", line)
            if not m:
                return False, "program_syntax", None
            env[m.group(1)] = last = float(eval(m.group(2), {"__builtins__": {}}, dict(env)))
        ans = env.get("answer", last if V4 else None)
        if ans is None or not math.isfinite(ans) or ans <= 0:
            return False, "answer_value", None
        if abs(ans - round(ans, 2)) > 1e-9:
            return False, "answer_not_round", None
        if any(abs(ans - float(g["value"])) < 1e-9 for g in given):
            return False, "answer_is_given", None
        used = " ".join(l.split("=", 1)[1] for l in lines)
        unused = [g["var"] for g in given if not re.search(rf"\b{g['var']}\b", used)]
        if unused:
            return False, f"given_unused:{unused[0]}", None
        arith = sum(bool(re.search(r"[+\-*/]", l.split("=", 1)[1])) for l in lines)
        if (arith if V5 else len(lines)) < max(2, k - 1):
            return False, "too_few_steps", None
        if V2 or V3:
            if len(q.split()) < 4:
                return False, "question_short", None
            pct = any("%" in g["phrase"] or "percent" in g["phrase"].lower() for g in given)
            for lit in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", used):
                L_ = float(lit)
                is_given = any(abs(L_ - float(g["value"])) < 1e-9 for g in given)
                if V3 and not (L_ == 1 or (L_ == 100 and pct) or (V4 and is_given)):
                    return False, f"numeric_literal:{lit}", None
                if not (any(abs(L_ - float(g["value"])) < 1e-9 for g in given) or L_ == 1 or (L_ == 100 and pct)):
                    return False, "hidden_constant", None
            gw = lambda t: set(re.findall(r"[a-z]{5,}", t.lower())) - STOP
            if not (gw(sk.get("setting", "")) & gw(general)):
                return False, "ungrounded_setting", None
        if re.search(rf"(?<![\d.]){re.escape(fmt(ans))}(?![\d.])", q):
            return False, "answer_in_question", None
        return True, "ok", round(ans, 2)
    except Exception as e:  # malformed JSON fields, eval errors (division by zero, names)
        return False, f"error:{type(e).__name__}: {str(e)[:80]}", None


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
    if V4:  # "value": 1/5 -> 0.2
        text = re.sub(r'("value"\s*:\s*)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',
                      lambda m: m.group(1) + repr(float(m.group(2)) / float(m.group(3))), text)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def feedback(why, sk, k):
    """Tell the skeleton writer what failed, for a repair round."""
    head, _, det = why.partition(":")
    var = {g.get("var"): g for g in (sk or {}).get("given", []) if isinstance(g, dict)}
    msg = {
        "json": "Your output was not valid JSON. Every value must be a plain number (write 0.2, not 1/5).",
        "shape": "Use 2 to 5 given quantities, a non-empty program and a final question ending with ?.",
        "given_var": "Every var must be a plain identifier such as a or n_cups.",
        "given_phrase": (f'The phrase of given quantity {det} ("{var.get(det, {}).get("phrase", "")}") does not contain its value '
                         f'{var.get(det, {}).get("value", "")}. Every phrase must state its number, e.g. "1000 miles between the two grounds".'),
        "program_syntax": "Every program line must be <var> = <expression> using only variables, + - * / and parentheses.",
        "answer_value": "The program must end with answer = <expression>, and the answer must be a positive number.",
        "answer_not_round": "The answer must be a whole number or have at most 2 decimals; change the given values.",
        "answer_is_given": "The answer equals one of the given quantities; change the problem so the answer is a new number.",
        "given_unused": f"Given quantity {det} is not used by the program; use every given quantity or remove it.",
        "too_few_steps": f"The program must have {k} lines, each performing one arithmetic operation (+, -, * or /); do not write lines that only copy a variable.",
        "question_short": "The final question must be a full question about the setting.",
        "numeric_literal": (f"The program uses the number {det}, which is not a given quantity. Add it as a given quantity "
                            f'whose phrase states it (e.g. value 2, phrase "twice as many") and use its variable.'),
        "hidden_constant": f"The program uses the number {det}, which is not a given quantity.",
        "ungrounded_setting": "The setting must use the people, objects or place of the general example.",
        "answer_in_question": "The final question must not contain the answer.",
        "inconsistent_skeleton": (f"A solver who read only the setting, the given phrases and the question got {det} instead of "
                                  f"the program's answer. Reword the phrases and the question so that they state every relation "
                                  f"unambiguously and the problem has exactly the program's answer."),
    }.get(head, f"The program could not be executed ({det}).")
    return msg + " Output only the corrected JSON."


def arith_steps(sk):
    return sum(bool(re.search(r"[+\-*/]", l.split("=", 1)[1])) for l in sk["program"] if "=" in l)


def skeleton_rounds(llm, rows, repair):
    """Generate, check and repair skeletons for ``rows``; returns kept, rejects, per-round reasons."""
    kept, reasons, rejects, pending = [], {}, [], rows
    for rnd in range(repair + 1):
        if not pending:
            break
        outs = llm.chat([r["conv"] for r in pending], max_new_tokens=512, do_sample=True, **SAMPLING)
        passed, failed = [], []
        for r, o in zip(pending, outs):
            sk = parse_json(o)
            ok, why, ans = check(sk, r["k"], r["general"]) if sk else (False, "json", None)
            (passed if ok else failed).append((r, o, sk, why, ans))
        if V3 and passed:  # self-consistency: a plain rendering, solved greedily, gives the program's answer
            plain = [f"{sk['setting'].strip()} Given: " + "; ".join(g_["phrase"] for g_ in sk["given"]) + ". " + sk["question"].strip()
                     for _, _, sk, _, _ in passed]
            sol = llm.chat([[{"role": "user", "content": SOLVE_PROMPT.format(q=p_)}] for p_ in plain], max_new_tokens=768)
            ok_rows = []
            for (r, o, sk, why, ans), p_, so in zip(passed, plain, sol):
                a_ = solved_answer(so)
                if a_ is not None and abs(a_ - ans) <= 1e-4 * max(1, abs(ans)):
                    ok_rows.append((r, o, sk, {"plain": p_}, ans))
                else:
                    failed.append((r, o, sk, f"inconsistent_skeleton:{fmt(a_) if a_ is not None else 'no answer'}", ans))
            passed = ok_rows
        for r, o, sk, extra, ans in passed:
            kept.append({"uid": r["uid"], "general": r["general"], "seed": r["seed"], "k": r["k"], "skeleton": sk,
                         "answer": ans, "raw": o, "round": rnd, "steps": arith_steps(sk), **(extra if isinstance(extra, dict) else {})})
        rr = reasons.setdefault(f"round{rnd}", {"in": len(pending), "ok": len(passed)})
        for _, _, _, why, _ in failed:
            rr[why.split(":")[0]] = rr.get(why.split(":")[0], 0) + 1
        print(f"round {rnd}: {len(passed)} kept of {len(pending)}; {rr}", flush=True)
        for r, o, sk, why, _ in failed:
            rejects.append({"uid": r["uid"], "k": r["k"], "round": rnd, "reason": why, "raw": o})
            r["conv"] = r["conv"] + [{"role": "assistant", "content": o},
                                     {"role": "user", "content": feedback(why, sk, r["k"])}]
        pending = [r for r, *_ in failed]
    return kept, rejects, reasons


def stage_skeleton(args):
    from llm import HFChat
    plan = read(ROOT / "data/synth/plan" / args.plan)
    rng = random.Random(0)
    ks = [2, 3, 4, 5, 6] if V5 else step_counts()
    rows = [{**r, "k": rng.choice(ks)} for r in plan[: args.n * args.cand]]
    lit = " only (no number literals except 1, and 100 for a percent)" if V3 else ' and small constants (like 2 for "twice")'
    rule = ("Each program line assigns one new variable by exactly one arithmetic operation (+, -, * or /) on given variables "
            "and earlier variables{lit}, so the program has exactly {k} lines; the last line is answer = <expression>." if V5 else
            "Each program line assigns one new variable from given variables and earlier variables{lit}; the last line assigns answer.")
    for r in rows:
        r["conv"] = [{"role": "user", "content": SKELETON_PROMPT.format(general=r["general"], seed=r["seed"], k=r["k"],
                                                                        program_rule=rule.format(lit=lit, k=r["k"]))}]
    OUT(args.n).mkdir(parents=True, exist_ok=True)
    llm = HFChat(str(QWEN), batch_size=args.batch_size)
    if not args.keep_all:  # probe: one pass, first n kept
        kept, rejects, reasons = skeleton_rounds(llm, rows, args.repair)
        by_round = [sum(r["round"] <= i for r in kept) for i in range(args.repair + 1)]
        with open(OUT(args.n) / "skeleton.jsonl", "w") as fh:
            for r in kept[: args.n]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT(args.n) / "skeleton_rejects.jsonl", "w") as fh:
            for r in rejects:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"candidates": len(rows), "kept_cumulative_by_round": by_round, "kept_total": len(kept),
                   "kept": min(len(kept), args.n), "reasons": reasons}, open(OUT(args.n) / "skeleton_stats.json", "w"), indent=1)
        print(f"skeletons: kept by round (cumulative) {by_round} of {len(rows)}; wrote {min(len(kept), args.n)}")
        return
    # full run: chunks appended to skeleton.jsonl, resumable (done uids in skeleton_chunks.jsonl)
    log = OUT(args.n) / "skeleton_chunks.jsonl"
    done = {u for c in read(log) for u in c["uids"]}
    todo = [r for r in rows if r["uid"] not in done]
    print(f"skeleton: {len(done)} candidates done, {len(todo)} to go", flush=True)
    for c0 in range(0, len(todo), args.chunk):
        n_kept = sum(c["kept"] for c in read(log))
        if args.max_kept and n_kept >= args.max_kept:
            print(f"skeleton: {n_kept} kept >= --max-kept {args.max_kept}; stop", flush=True)
            break
        chunk = todo[c0:c0 + args.chunk]
        kept, rejects, reasons = skeleton_rounds(llm, chunk, args.repair)
        for name, items in [("skeleton", kept), ("skeleton_rejects", rejects)]:
            with open(OUT(args.n) / f"{name}.jsonl", "a") as fh:
                for r in items:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(log, "a") as fh:
            fh.write(json.dumps({"uids": [r["uid"] for r in chunk], "kept": len(kept), "reasons": reasons}) + "\n")
        tot = read(log)
        print(f"[chunk] {sum(len(c['uids']) for c in tot)} candidates, {sum(c['kept'] for c in tot)} kept", flush=True)
    tot = read(log)
    agg = {}
    for c in tot:
        for rnd, d in c["reasons"].items():
            for k_, v in d.items():
                agg.setdefault(rnd, {})[k_] = agg.get(rnd, {}).get(k_, 0) + v
    json.dump({"candidates": sum(len(c["uids"]) for c in tot), "kept": sum(c["kept"] for c in tot), "reasons": agg},
              open(OUT(args.n) / "skeleton_stats.json", "w"), indent=1)


# GSM8K test gold: share of problems by number of <<...>> calculations (<=2, 3, 4, 5, >=6)
GOLD_STEPS = {2: 440 / 1319, 3: 364 / 1319, 4: 290 / 1319, 5: 138 / 1319, 6: 87 / 1319}


def stage_select(args):
    """Keep rows whose text re-solves to the program answer and whose label is that answer, then draw
    --size rows per generator with the step distribution of GSM8K gold (a short bucket passes its
    deficit to the next shorter one). Writes train_<gen>.jsonl (instruction, output)."""
    sys.argv = ["x", "a=b"]
    g = {"__file__": str(ROOT / "scripts/probes/structure_metrics.py")}
    exec((ROOT / "scripts/probes/structure_metrics.py").read_text().split("ap = argparse")[0], g)
    sk = {r["uid"]: r for r in read(OUT(args.n) / "skeleton.jsonl")}
    bucket = lambda r: max(2, min(6, r.get("steps", arith_steps(r["skeleton"]))))
    rng = random.Random(0)
    summary = {}
    for gen in args.gens:
        ann = {r["uid"]: r for r in read(OUT(args.n) / f"annotate_{gen}.jsonl")}
        ver = {v["uid"]: v["solve_answer"] for v in read(OUT(args.n) / f"verify_{gen}.jsonl")}
        good = [u for u, a in ann.items() if g["same"](ver.get(u), sk[u]["answer"])
                and g["same"](g["final_answer"](a["output"]), sk[u]["answer"])]
        pools = {b: [u for u in good if bucket(sk[u]) == b] for b in GOLD_STEPS}
        for b in pools:
            rng.shuffle(pools[b])
        quota = {b: round(args.size * p) for b, p in GOLD_STEPS.items()}
        take, carry = {}, 0
        for b in sorted(GOLD_STEPS, reverse=True):  # long buckets first; deficits move to shorter ones
            want = quota[b] + carry
            take[b] = pools[b][:want]
            carry = want - len(take[b])
        chosen = [u for b in take for u in take[b]]
        with open(OUT(args.n) / f"train_{gen}.jsonl", "w") as fh:
            for u in chosen:
                fh.write(json.dumps({"instruction": ann[u]["instruction"], "output": ann[u]["output"], "uid": u,
                                     "steps": bucket(sk[u])}, ensure_ascii=False) + "\n")
        summary[gen] = {"annotated": len(ann), "verified_and_label": len(good), "pool_by_steps": {b: len(v) for b, v in pools.items()},
                        "selected": len(chosen), "selected_by_steps": {b: len(v) for b, v in take.items()}, "short_by": carry}
        print(gen, summary[gen])
    json.dump(summary, open(OUT(args.n) / "select_stats.json", "w"), indent=1)


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
    """PANGEA labeler, one sample per question. --resample R: questions whose label is not the skeleton's
    answer are labeled again, up to R more times, and the first matching solution is kept (rejection
    sampling against the program answer). ``label_first_ok`` records the single-sample result."""
    from llm import HFChat
    qs = read(OUT(args.n) / f"write_{args.gen}.jsonl")
    sk = {r["uid"]: r for r in read(OUT(args.n) / "skeleton.jsonl")}
    template = (PROMPTS / "gsm8k_annotation.txt").read_text()
    llm = HFChat(str(QWEN), batch_size=16)
    label = lambda idx: llm.chat([[{"role": "user", "content": template.format(question=qs[i]["question"])}] for i in idx],
                                 max_new_tokens=1536, assistant_prefix=PREFIX["annotation"]["gsm8k"], do_sample=True, **SAMPLING)
    outs = label(range(len(qs)))
    ok = lambda i, o: (lambda a: a is not None and abs(a - sk[qs[i]["uid"]]["answer"]) <= 1e-4 * max(1, abs(sk[qs[i]["uid"]]["answer"])))(solved_answer(o))
    first = [ok(i, o) for i, o in enumerate(outs)]
    tries = [1] * len(qs)
    for t in range(args.resample):
        bad = [i for i, o in enumerate(outs) if not ok(i, o)]
        if not bad:
            break
        for i, o in zip(bad, label(bad)):
            tries[i] += 1
            if ok(i, o):
                outs[i] = o
        print(f"annotate {args.gen} resample {t + 1}: {len(bad)} relabeled, {sum(not ok(i, o) for i, o in enumerate(outs))} still wrong", flush=True)
    with open(OUT(args.n) / f"annotate_{args.gen}.jsonl", "w") as fh:
        for i, (q, o) in enumerate(zip(qs, outs)):
            fh.write(json.dumps({"uid": q["uid"], "instruction": q["question"], "output": o, "label_first_ok": first[i],
                                 "label_tries": tries[i]}, ensure_ascii=False) + "\n")
    print(f"annotate {args.gen}: {len(qs)}; first-sample correct {sum(first) / max(1, len(qs)):.2f}")


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
    st_ = [max(2, min(6, arith_steps(r["skeleton"]))) for r in sk.values()]
    print("arithmetic steps (2=<=2, 6=>=6):", {b: round(st_.count(b) / len(st_), 2) for b in range(2, 7)},
          "| GSM8K gold:", {b: round(p, 2) for b, p in GOLD_STEPS.items()})


ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["skeleton", "write", "annotate", "verify", "report", "select"])
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--gen", choices=["qwen", "dream"])
ap.add_argument("--gens", nargs="+", default=["qwen", "dream"])
ap.add_argument("--tag", default="", help="output dir suffix: data/synth/skel_<tag>/")
ap.add_argument("--v2", action="store_true", help="no hidden constants, question >= 4 words, setting grounded in the general example")
ap.add_argument("--v3", action="store_true", help="v2 + no number literals, numeric phrase check, self-consistent skeletons")
ap.add_argument("--v4", action="store_true", help="v3 + relaxed rules (see V4); use with --repair")
ap.add_argument("--v5", action="store_true", help="v4 + steps counted as arithmetic lines, one operation per line, uniform k")
ap.add_argument("--plan", default="gsm8k_n1000.jsonl", help="plan file in data/synth/plan/")
ap.add_argument("--keep-all", action="store_true", help="skeleton: keep every passing skeleton, in resumable chunks")
ap.add_argument("--chunk", type=int, default=512, help="skeleton --keep-all: candidates per chunk")
ap.add_argument("--batch-size", type=int, default=16, help="skeleton: Qwen batch size")
ap.add_argument("--max-kept", type=int, default=0, help="skeleton --keep-all: stop once this many skeletons are kept")
ap.add_argument("--resample", type=int, default=0, help="annotate: relabel wrong labels up to this many more times")
ap.add_argument("--size", type=int, default=1000, help="select: rows per generator")
ap.add_argument("--repair", type=int, default=0, help="repair rounds: failed skeletons go back to Qwen with the failure reason")
ap.add_argument("--cand", type=int, default=2, help="candidate skeletons per kept row")
ap.add_argument("--budget", type=int, default=64, help="Dream: canvas tokens shared by the spans between anchors")
args = ap.parse_args()
args.v4 = args.v4 or args.v5
TAG, V2, V3, V4, V5 = args.tag, args.v2 or args.v3 or args.v4, args.v3 or args.v4, args.v4, args.v5
{"skeleton": stage_skeleton, "write": stage_write, "annotate": stage_annotate, "verify": stage_verify,
 "report": stage_report, "select": stage_select}[args.stage](args)
