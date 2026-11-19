"""Independent greedy re-solve of synthetic questions, to check the labeler's answers.

  python scripts/probes/label_resolve.py out.jsonl name=train.jsonl ...

The labeler sampled one solution at T 0.9; here Qwen2.5-7B-Instruct solves each question
again greedily with a plain prompt. Agreement is a proxy for label correctness, not ground truth.
Resumable: rows already in out.jsonl are skipped."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm import load_llm

PROMPT = ("Solve the following math word problem step by step. "
          "End with a final line of the form \"The answer is: <number>\".\n\nProblem: {q}")

out_path, specs = sys.argv[1], sys.argv[2:]
done = set()
if os.path.exists(out_path):
    done = {(r["cond"], r["uid"]) for r in map(json.loads, open(out_path))}
todo = []
for spec in specs:
    name, path = spec.split("=", 1)
    todo += [(name, r) for r in map(json.loads, open(path)) if (name, r["uid"]) not in done]
print(f"{len(todo)} to solve", flush=True)
llm = load_llm("hf:model/Qwen2.5-7B-Instruct")
llm.batch_size = 32
with open(out_path, "a") as f:
    for s in range(0, len(todo), 128):
        chunk = todo[s:s + 128]
        outs = llm.chat([[{"role": "user", "content": PROMPT.format(q=r["instruction"])}] for _, r in chunk],
                        max_new_tokens=768)
        for (name, r), o in zip(chunk, outs):
            f.write(json.dumps({"cond": name, "uid": r["uid"], "resolve": o}) + "\n")
        f.flush()
        print(f"{s + len(chunk)}/{len(todo)}", flush=True)
