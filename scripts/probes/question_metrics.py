"""Checks on generated GSM8K questions, same for every condition.
  python scripts/probes/question_metrics.py name=path.jsonl[:field] ...   (field default "projection")"""
import json, re, statistics as st, sys
def words(s): return set(re.findall(r"[a-z]{5,}", s.lower()))
def ngrams(s, n=3):
    w = re.findall(r"\w+", s.lower()); return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
FRUIT = re.compile(r"\b(apples?|bananas?|oranges?|fruits?|cookies?)\b", re.I)
print("| cond | n | empty | shares word w/ general | fruit/cookie not in general | 'A group of' start | distinct 3-gram | ends '?' | 2-4 sentences | words (median) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    path, field = (path.split(":") + ["projection"])[:2]
    rs = [json.loads(l) for l in open(path)][:64]
    q = [r[field].strip() for r in rs]
    ok = [(r, x) for r, x in zip(rs, q) if x]
    g = [t for _, x in ok for t in ngrams(x)]
    sents = [len(re.findall(r"[.?!](?:\s|$)", x)) for _, x in ok]
    print(f"| {name} | {len(rs)} | {len(rs) - len(ok)} | {sum(len(words(x) & words(r['general'])) > 0 for r, x in ok)} | "
          f"{sum(bool(FRUIT.search(x)) and not FRUIT.search(r['general']) for r, x in ok)} | "
          f"{sum(x.startswith('A group of') for _, x in ok)} | {len(set(g)) / max(1, len(g)):.2f} | "
          f"{sum(x.endswith('?') for _, x in ok)} | {sum(2 <= s <= 4 for s in sents)} | "
          f"{st.median(len(x.split()) for _, x in ok) if ok else 0} |")
