"""How often a question states a value τ computes (the answer or an intermediate result).
  python scripts/probes/answer_leak.py name=projection.jsonl ..."""
import json, re, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
import anchors
def nums(s): return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    rs = [json.loads(l) for l in open(path)][:64]
    ans = inter = 0
    for r in rs:
        q, tau = nums(r["projection"]), r["writer"]
        a = anchors.derived_answer(tau)
        letters, values = anchors.derived(tau)
        sym = dict(re.findall(r"\b([A-Z])\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)", anchors._field(tau, "Symbols")))
        dvals = values | {float(sym[l].replace(",", "")) for l in letters if l in sym}
        given = {float(v.replace(",", "")) for l, v in sym.items() if l not in letters}
        ans += a is not None and a in q and a not in given
        inter += bool((dvals - given) & q)
    print(f"{name}: answer value in question {ans}/{len(rs)}, any derived value in question {inter}/{len(rs)}")
