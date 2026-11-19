"""Does τ come from the general example? Same checks for every condition (GSM8K Writer).
  python scripts/probes/grounding_metrics.py name=path.jsonl ..."""
import json, re, sys
def words(s): return set(re.findall(r"[a-z]{4,}", s.lower()))
GENERIC = re.compile(r"\b(apples?|bananas?|oranges?|fruits?|cookies?)\b", re.I)
print("| cond | n | scenario shares a word with general | 10-20-30 pattern | generic fruit/cookie | distinct scenario openings |")
print("|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    rs = [json.loads(l) for l in open(path)][:64]
    scen = [re.search(r"Scenario - ([^\n]*)", r["writer"]) for r in rs]
    scen = [s.group(1) if s else "" for s in scen]
    share = sum(len(words(s) & words(r["general"])) > 0 for s, r in zip(scen, rs))
    prog = sum(bool(re.search(r"\b10\b.*\b20\b.*\b30\b", r["writer"])) for r in rs)
    fruit = sum(bool(GENERIC.search(s)) and not GENERIC.search(r["general"]) for s, r in zip(scen, rs))
    print(f"| {name} | {len(rs)} | {share} | {prog} | {fruit} | {len({s[:40] for s in scen})} |")
