"""Checks on GSM8K Projection probe outputs, plus a labeler check: Qwen (PANGEA annotation
prompt and sampling) answers each question twice; we parse the final number of each answer.
  python scripts/probes/projection_metrics.py g1 g1p g2 g3"""
import json, re, sys, statistics as st
sys.path.insert(0, "/home/M2026107/domain/scripts")
ROOT = "/home/M2026107/domain"
NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def final_number(text):
    m = re.search(r"answer is:?\s*\**\s*\\?\(?\$?\\?(?:boxed\{)?\s*(-?[\d,]*\.?\d+)", text, re.I)
    if m:
        v = m.group(1)
    else:
        nums = NUM.findall(text)
        if not nums:
            return None
        v = nums[-1]
    try:
        return round(float(v.replace(",", "").replace("$", "")), 4)
    except ValueError:
        return None


def ngrams(s, n=5):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    from llm import HFChat
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/model/Qwen2.5-7B-Instruct")
    labeler = HFChat(f"{ROOT}/model/Qwen2.5-7B-Instruct", batch_size=16)
    tpl = open(f"{ROOT}/pipeline/prompts/gsm8k_annotation.txt").read()
    table = []
    for cond in sys.argv[1:]:
        rows = [json.loads(l) for l in open(f"{ROOT}/data/synth/probe_proj/gsm8k_{cond}.jsonl")]
        qs = [r["projection"].strip() for r in rows]
        ok = [i for i, q in enumerate(qs) if q]
        convs = [[{"role": "user", "content": tpl.format(question=qs[i])}] for i in ok]
        ans = [labeler.chat(convs, max_new_tokens=1024, assistant_prefix="- **Answer**: ", do_sample=True,
                            temperature=0.9, top_p=0.7, top_k=10) for _ in range(2)]
        a1 = [final_number(a) for a in ans[0]]
        a2 = [final_number(a) for a in ans[1]]
        lens = [len(tok(qs[i])["input_ids"]) for i in ok]
        sents = [len(re.findall(r"[.?!](?:\s|$)", qs[i])) for i in ok]
        seed_overlap = sum(len(ngrams(qs[i]) & ngrams(rows[i]["seed"])) > 0 for i in ok)
        repeat_num = sum(len(set(NUM.findall(qs[i]))) < len(NUM.findall(qs[i])) for i in ok)
        table.append({
            "cond": cond, "n": len(rows), "nonempty": len(ok),
            "ends_with_?": sum(qs[i].rstrip().endswith("?") for i in ok),
            "2-4 sentences": sum(2 <= s <= 4 for s in sents),
            "len_median": st.median(lens) if lens else 0,
            "repeated_number": repeat_num,
            "seed_5gram_overlap": seed_overlap,
            "labeler_parsed": sum(x is not None for x in a1),
            "labeler_agree": sum(x is not None and x == y for x, y in zip(a1, a2)),
        })
        with open(f"{ROOT}/data/synth/probe_proj/gsm8k_{cond}_labeled.jsonl", "w") as fh:
            for j, i in enumerate(ok):
                fh.write(json.dumps({"uid": rows[i]["uid"], "question": qs[i], "answer1": ans[0][j],
                                     "answer2": ans[1][j], "num1": a1[j], "num2": a2[j]},
                                    ensure_ascii=False) + "\n")
    keys = list(table[0])
    print("| " + " | ".join(keys) + " |")
    for r in table:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
