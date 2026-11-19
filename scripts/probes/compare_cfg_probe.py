"""Compare CFG probe dirs with the qtau baseline on the same uids (CPU only).

  python scripts/probes/compare_cfg_probe.py g3_cfg_w1_l05 g3_cfg_w2_l05 ...

Writes the baseline subset (same uids as the first probe dir) to data/synth/g3_qtau_probe64/,
then prints structure_metrics (math-ness, label quality, answer leak) and question_metrics
(grounding in the general example, 3-gram diversity), plus a τ-notation leak count."""
import json, re, subprocess, sys
from pathlib import Path

S = Path("data/synth")
dirs = sys.argv[1:]
uids = [json.loads(l)["uid"] for l in open(S / dirs[0] / "gsm8k/n1000/projection.jsonl")]
base = S / "g3_qtau_probe64/gsm8k/n1000"
base.mkdir(parents=True, exist_ok=True)
for f in ["projection", "annotation", "train"]:
    rows = {r["uid"]: r for r in map(json.loads, open(S / f"g3_qtau/gsm8k/n1000/{f}.jsonl"))}
    with open(base / f"{f}.jsonl", "w") as fh:
        for u in uids:
            if u in rows:
                fh.write(json.dumps(rows[u], ensure_ascii=False) + "\n")

conds = [("qtau(w0)", base)] + [(d.replace("g3_cfg_", ""), S / d / "gsm8k/n1000") for d in dirs]
py = sys.executable
subprocess.run([py, "scripts/probes/structure_metrics.py"] + [f"{n}={p}/train.jsonl" for n, p in conds], check=True)
print()
subprocess.run([py, "scripts/probes/question_metrics.py"] + [f"{n}={p}/projection.jsonl" for n, p in conds], check=True)

# τ notation in the question: "A, 100 people", "A is 20 minutes", "C is the remaining", "100 people - pilots"
LEAK = [re.compile(r"\b[A-E](,| is| =| equals)\s"), re.compile(r"(?<![.?!]\s)(?<!^)\b[B-E]\b"),
        re.compile(r"\d[\w ]{0,20} - [a-z]"), re.compile(r"- (Scenario|Symbols|Equations|Target|Plan) -")]
print("\n| cond | τ notation leak | τ format output |\n|---|---|---|")
for n, p in conds:
    q = [json.loads(l)["projection"] for l in open(p / "projection.jsonl")]
    leak = sum(any(r.search(x) for r in LEAK[:3]) for x in q)
    fmt = sum(bool(LEAK[3].search(x)) for x in q)
    print(f"| {n} | {leak}/{len(q)} | {fmt}/{len(q)} |")
