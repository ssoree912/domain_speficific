"""Surface diversity of a full train.jsonl: n-gram distinctness, most common opening, question length.
  python scripts/probes/diversity_1k.py name=train.jsonl ..."""
import json, re, sys, statistics as st, collections as C
def ng(s,n=3):
    w=re.findall(r"\w+",s.lower()); return [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
print("| cond | n | q words med | nums/q | distinct 3-gram | distinct 4-gram(all) | top start (3w) | 'A group of' | 'total' q | sym leak (A, / A is) | ans words med |")
for a in sys.argv[1:]:
    name,p=a.split("=",1)
    rs=[json.loads(l) for l in open(p)]
    q=[r["instruction"].strip() for r in rs]; o=[r["output"] for r in rs]
    g3=[t for x in q for t in ng(x)]; g4=[t for x in q for t in ng(x,4)]
    nums=st.median(len(re.findall(r'\d+(?:\.\d+)?',x)) for x in q)
    starts=C.Counter(" ".join(x.split()[:3]) for x in q).most_common(1)[0]
    leak=sum(bool(re.search(r"\b[A-E](,| is| =)\s*\d",x)) or bool(re.search(r"\d[^.?]* - [a-z]",x)) for x in q)
    print(f"| {name} | {len(q)} | {st.median(len(x.split()) for x in q)} | {nums} | "
          f"{len(set(g3))/len(g3):.2f} | {len(set(g4))/len(g4):.2f} | \"{starts[0]}\" {starts[1]} | {sum(x.startswith('A group of') for x in q)} | "
          f"{sum('total' in x.lower() for x in q)} | {leak} | {st.median(len(x.split()) for x in o)} |")
