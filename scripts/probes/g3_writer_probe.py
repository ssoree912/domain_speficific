"""G3 Writer probe: a dLLM with anchors fixed on the canvas.
  python scripts/probes/g3_writer_probe.py <domain> <heads|rich> <n> [dream|llada] [low|pangea|greedy] [block]
(a block >= the canvas length is LLaDA without semi-AR blocks: one block, any order)
dream defaults to its low-temperature setting (T 0.2, top-p 0.95, steps = L); llada uses
LLaDA's semi-AR block decoding (block 32 by default) with PANGEA's T 0.9 / top-p 0.7 / top-k 10."""
import json, os, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchors_spec import SPECS
ROOT = "/home/M2026107/domain"
SAMPLING = {"low": dict(temperature=0.2, top_p=0.95, top_k=None),
            "pangea": dict(temperature=0.9, top_p=0.7, top_k=10),
            "greedy": dict(temperature=0.0, top_p=None, top_k=None)}
d, spec, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
engine = sys.argv[4] if len(sys.argv) > 4 else "dream"
sampling = sys.argv[5] if len(sys.argv) > 5 else ("low" if engine == "dream" else "pangea")
block = int(sys.argv[6]) if len(sys.argv) > 6 else 32
prefix, anchors = SPECS[spec][d]
tpl = open(f"{ROOT}/pipeline/prompts/{d}_writer.txt").read()
rows = [json.loads(l) for l in open(f"{ROOT}/data/synth/plan/{d}_n200.jsonl")][:n]
if engine == "dream":
    from dream_chat import DreamChat
    g = DreamChat(f"{ROOT}/model/Dream-v0-Instruct-7B", batch_size=4)
else:
    from llada_chat import LLaDAChat
    g = LLaDAChat(f"{ROOT}/model/LLaDA-8B-Instruct", batch_size=4)
anch, off = {}, 0
for text, span in anchors:
    anch[off] = text
    off += len(g.anchor_ids(text)) + span
convs = [[{"role": "user", "content": tpl.format(general=r["general"])}] for r in rows]
kw = dict(block_length=block) if engine == "llada" else {}
outs = g.chat(convs, gen_length=off, steps=off, assistant_prefix=prefix, anchors=[anch] * n,
              **SAMPLING[sampling], **kw)
tag = "" if engine == "dream" and sampling == "low" else f"_{engine}_{sampling}" + (f"_b{block}" if engine == "llada" else "")
out = f"{ROOT}/data/synth/probe_g3_writer/{d}_{spec}{tag}.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    for r, o in zip(rows, outs):
        fh.write(json.dumps({**r, "writer": o, "L": off}, ensure_ascii=False) + "\n")
print(out, "L", off)
