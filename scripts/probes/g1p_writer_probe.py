"""G1' Writer probe: Qwen AR with the same anchors as the G3 probe, inserted span by span.
Per anchor: append it, sample up to its span cap (PANGEA sampling), cut the span at the first
newline or where the model starts writing the next anchor itself, then append the next anchor.
  python scripts/probes/g1p_writer_probe.py <domain> <heads|rich> <n>"""
import json, os, re, sys
sys.path.insert(0, "/home/M2026107/domain/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchors_spec import SPECS
from llm import HFChat
ROOT = "/home/M2026107/domain"
d, spec, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prefix, anchors = SPECS[spec][d]
tpl = open(f"{ROOT}/pipeline/prompts/{d}_writer.txt").read()
rows = [json.loads(l) for l in open(f"{ROOT}/data/synth/plan/{d}_n200.jsonl")][:n]
g = HFChat(f"{ROOT}/model/Qwen2.5-7B-Instruct", batch_size=16)
convs = [[{"role": "user", "content": tpl.format(general=r["general"])}] for r in rows]
texts = [""] * n
for k, (text, span) in enumerate(anchors):
    texts = [t + text for t in texts]
    outs = g.chat(convs, max_new_tokens=span, assistant_prefix=[prefix + t for t in texts],
                  do_sample=True, temperature=0.9, top_p=0.7, top_k=10)
    # the next anchor if it continues this line, e.g. " (2) " or " → (2) "
    nxt = anchors[k + 1][0] if k + 1 < len(anchors) and not anchors[k + 1][0].startswith("\n") else None
    cut = []
    for o in outs:
        o = o.split("\n")[0]
        if nxt:  # stop where the model starts the next item itself, even partially ("(2", "→")
            m = re.search(r"→|\(\s*\d", o)
            o = o[:m.start()] if m else o
        cut.append(o.rstrip(" ,;(→"))  # a span cap can also end on a bare "("
    texts = [t + o for t, o in zip(texts, cut)]
out = f"{ROOT}/data/synth/probe_g1p_writer/{d}_{spec}.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    for r, t in zip(rows, texts):
        fh.write(json.dumps({**r, "writer": t.strip()}, ensure_ascii=False) + "\n")
print(out)
