import json, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
from dream_chat import DreamChat
ROOT = "/home/M2026107/domain"
g = DreamChat(f"{ROOT}/model/Dream-v0-Instruct-7B", batch_size=8)
PRE = {"medqa": "Clinical Building Blocks:", "finqa": "Financial Scenario:"}
for d in ["medqa", "finqa"]:
    tpl = open(f"{ROOT}/pipeline/prompts/{d}_writer.txt").read()
    rows = [json.loads(l) for l in open(f"{ROOT}/data/synth/plan/{d}_n200.jsonl")][:8]
    convs = [[{"role": "user", "content": tpl.format(general=r["general"])}] for r in rows]
    for name, kw in [("dream-default T0.2 p0.95 steps=L", dict(temperature=0.2, top_p=0.95, top_k=None, steps=256)),
                     ("no prefix, pangea sampling", dict(temperature=0.9, top_p=0.7, top_k=10, steps=128, nopre=True))]:
        pre = "" if kw.pop("nopre", False) else PRE[d]
        outs = g.chat(convs, gen_length=256, assistant_prefix=pre, **kw)
        print("#####", d, name)
        for o in outs: print(repr(o[:250]))
