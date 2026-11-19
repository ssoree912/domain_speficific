"""GSM8K Projection probe for the 2×2: τ (Writer output of each condition) + seed -> question.

  g1   Qwen, PANGEA prompt, AR                     (τ from G1)
  g1p  Qwen, same anchors as g3, span by span      (τ from G1' rich)
  g2   Dream, PANGEA prompt, canvas without anchors (τ from G2)
  g3   Dream, anchors on the canvas, infilled      (τ from G3 rich)

Anchors (g1p, g3): the values of τ's Symbols in order (at most 6), then "?" at the end; the
spans between them share a budget of 64 tokens (seed GSM8K question median 60 tokens).

  python scripts/probes/projection_probe.py <g1|g1p|g2|g3> <writer jsonl> <n>
"""
import json, os, re, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
ROOT = "/home/M2026107/domain"
PREFIX = "Transformed Domain Question:"
BUDGET, MAX_NUMS, G2_CANVAS = 64, 6, 160
NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def tau_numbers(tau):
    m = re.search(r"Symbols\s*[-:–]\s*(.*)", tau)
    sym = m.group(1) if m else ""
    vals = [NUM.match(v.strip()).group(0) for v in re.findall(r"=\s*([^,;]*)", sym)
            if NUM.match(v.strip())] or NUM.findall(sym)
    out = []
    for v in vals:
        v = v.rstrip(",")
        if v not in out:
            out.append(v)
    return out[:MAX_NUMS]


def anchor_plan(tau):
    """[(anchor text, span before it)]: each number and the final '?' gets a preceding span."""
    nums = tau_numbers(tau)
    span = max(8, BUDGET // (len(nums) + 1))
    return [(" " + v, span) for v in nums] + [("?", span)]


def main():
    cond, src, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tpl = open(f"{ROOT}/pipeline/prompts/gsm8k_projection.txt").read()
    rows = [json.loads(l) for l in open(src)][:n]
    convs = [[{"role": "user", "content": tpl.format(block=r["writer"].strip(), seed=r["seed"])}]
             for r in rows]
    plans = [anchor_plan(r["writer"]) for r in rows]
    if cond in ("g1", "g1p"):
        from llm import HFChat
        g = HFChat(f"{ROOT}/model/Qwen2.5-7B-Instruct", batch_size=16)
        sampling = dict(do_sample=True, temperature=0.9, top_p=0.7, top_k=10)
    else:
        from dream_chat import DreamChat
        g = DreamChat(f"{ROOT}/model/Dream-v0-Instruct-7B", batch_size=4)
        sampling = dict(temperature=0.2, top_p=0.95, top_k=None)

    if cond == "g1":
        outs = g.chat(convs, max_new_tokens=1024, assistant_prefix=PREFIX, **sampling)
    elif cond == "g2":
        outs = g.chat(convs, gen_length=G2_CANVAS, steps=G2_CANVAS, assistant_prefix=PREFIX, **sampling)
    elif cond == "g3":
        anchors, lens = [], []
        for plan in plans:
            a, off = {}, 0
            for text, span in plan:
                off += span
                a[off] = text
                off += len(g.anchor_ids(text))
            anchors.append(a)
            lens.append(off)
        # rows are batched by canvas length, so group by L
        outs = [None] * len(rows)
        for L in sorted(set(lens)):
            idx = [i for i, l in enumerate(lens) if l == L]
            res = g.chat([convs[i] for i in idx], gen_length=L, steps=L, assistant_prefix=PREFIX,
                         anchors=[anchors[i] for i in idx], **sampling)
            for i, o in zip(idx, res):
                outs[i] = o
    else:  # g1p: per anchor, sample the span before it and stop where the model writes it itself
        texts = [""] * len(rows)
        for k in range(max(len(p) for p in plans)):
            live = [i for i, p in enumerate(plans) if k < len(p)]
            res = g.chat([convs[i] for i in live], max_new_tokens=max(plans[i][k][1] for i in live),
                          assistant_prefix=[PREFIX + texts[i] for i in live], **sampling)
            for i, o in zip(live, res):
                text, span = plans[i][k]
                o = o.split("\n")[0] if texts[i].strip() else o.lstrip("\n").split("\n")[0]
                stop = re.search(r"\?" if text == "?" else r"[\d$]", o)
                o = o[:stop.start()] if stop else o
                o = g.tok.decode(g.tok(o, add_special_tokens=False)["input_ids"][:span])
                texts[i] += o.rstrip() + text
        outs = texts

    out = f"{ROOT}/data/synth/probe_proj/gsm8k_{cond}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        for r, p, o in zip(rows, plans, outs):
            fh.write(json.dumps({**r, "anchors": [t for t, _ in p] if cond in ("g1p", "g3") else [],
                                 "projection": o.strip()}, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
