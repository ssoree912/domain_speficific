import json, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
from dream_chat import DreamChat
ROOT = "/home/M2026107/domain"
tpl = open(f"{ROOT}/pipeline/prompts/medqa_writer.txt").read()
rows = [json.loads(l) for l in open(f"{ROOT}/data/synth/plan/medqa_n200.jsonl")][:8]
convs = [[{"role": "user", "content": tpl.format(general=r["general"])}] for r in rows]
heads = ["Demographics", "Timeline", "Key clinical events", "Lab / imaging patterns", "Pathophysiologic principles"]
span = int(sys.argv[1]) if len(sys.argv) > 1 else 40
anch, off = {}, 0
g = DreamChat(f"{ROOT}/model/Dream-v0-Instruct-7B", batch_size=8)
for h in heads:
    t = ("\n" if off == 0 else "\n") + f"• {h} – "
    anch[off] = t
    off += len(g.anchor_ids(t)) + span
L = off
for alg in ["entropy", "maskgit_plus"]:
    outs = g.chat(convs, gen_length=L, steps=L // 2, assistant_prefix="Clinical Building Blocks:",
                  anchors=[anch] * len(convs), temperature=0.9, top_p=0.7, top_k=10, alg=alg)
    print("#####", alg, "L", L)
    for r, o in zip(rows, outs):
        print("GEN:", r["general"][:120].replace("\n", " "))
        print(o[:600]); print("---")
