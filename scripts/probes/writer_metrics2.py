"""Extra GSM8K Writer checks for the rich-anchor comparison.
  python scripts/probes/writer_metrics2.py name=path.jsonl ..."""
import json, re, sys
def eqs(x):
    m = re.search(r"Equations - (.*)", x)
    return [p.strip(" ,.") for p in re.split(r"\(\d\)", m.group(1)) if p.strip(" ,.")] if m else []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    o = [json.loads(l)["writer"] for l in open(path)][:64]
    rep = sum(len(set(e)) < len(e) for e in map(eqs, o))
    scen = sum(bool(re.search(r"Scenario - [^\n]*=", x)) for x in o)
    broken = sum(bool(re.search(r"\(\d\s*→|→\s*→|\(\s*\(\d|\(\d \(\d", x)) for x in o)
    print(f"{name}: equation_repeat {rep}, scenario_has_equation {scen}, broken_numbering {broken}")
