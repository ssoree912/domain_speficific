"""Rule-based checks on GSM8K Writer outputs (τ), same for every condition.
  python scripts/probes/writer_metrics.py name=path.jsonl ..."""
import json, re, sys
F = ["Scenario", "Symbols", "Equations", "Target", "Plan"]
SPLIT = re.compile(r"(?:^|\n)\s*-?\s*(Scenario|Symbols|Equations|Target|Plan)\s*[-:–]\s*")

def fields(x):
    parts = SPLIT.split("\n" + x)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

def check(x):
    f = fields(x)
    vals = [f.get(k, "") for k in F]
    return {
        "copy_template": bool(re.search(r"<A = value|<one kid|\(1\) \.\.\. = \.\.\.", x)),
        "missing_field": any(not v for v in vals),
        "dup_fields": len({v for v in vals if v}) < len([v for v in vals if v]),
        "echo_instructions": bool(re.search(r"kid-friendly|meaningful numbers|simple equations|Hard Restrictions", x)),
        "filler": bool(re.search(r"(\d+,\s*){6,}|([,.]\s*){8,}|N/A|Not applicable", x)),
        "symbols_has_number": bool(re.search(r"\d", f.get("Symbols", ""))),
        "target_is_T": bool(re.match(r"<?T\s*=", f.get("Target", ""))),
        "n_equations": len(re.findall(r"=", f.get("Equations", ""))),
    }

rows = []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    res = [check(json.loads(l)["writer"]) for l in open(path)][:64]
    n = len(res)
    row = {"cond": name, "n": n}
    for k in res[0]:
        if k == "n_equations":
            row["4-5 eq"] = f"{sum(4 <= r[k] <= 6 for r in res)}"
        else:
            row[k] = sum(r[k] for r in res)
    rows.append(row)
keys = list(rows[0])
print("| " + " | ".join(keys) + " |")
for r in rows: print("| " + " | ".join(str(r[k]) for k in keys) + " |")
