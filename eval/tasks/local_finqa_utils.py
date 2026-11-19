"""FinQA: prompt formatting and scoring.

The parquet keeps FinQA's nested fields as JSON strings (scripts/download_data.py).
format_question / format_answer are also used by scripts/make_seeds.py.

Scoring compares against the executed program value ``exe_ans`` and the annotated answer
text, since about 8% of FinQA test items have the two disagree. exe_ans writes percentages
as fractions (answer "14%" has exe_ans 0.14464), so a prediction p counts if p, or p / 100
when it is written with "%" or the gold is a fraction, is within 1% relative (0.005
absolute near zero) of a gold value. yes/no questions compare the word.
"""
import json
import re


def _j(x):
    return json.loads(x) if isinstance(x, str) and x[:1] in "[{" else x


def format_table(table):
    return "\n".join(" | ".join(c.strip() for c in row) for row in table)


def format_question(doc):
    pre = " ".join(_j(doc["pre_text"]))
    post = " ".join(_j(doc["post_text"]))
    table = format_table(_j(doc["table"]))
    q = _j(doc["qa"])["question"]
    return f"{pre}\n\n{table}\n\n{post}\n\n{q}".strip()


def gold_value(doc):
    return _j(doc["qa"])["exe_ans"]


def answer_text(doc):
    qa = _j(doc["qa"])
    if qa.get("answer"):
        return qa["answer"]
    v = qa["exe_ans"]
    return v if isinstance(v, str) else f"{v:g}"


def format_answer(doc):
    """Seed answer: evidence sentences, the calculation program, and the answer."""
    qa = _j(doc["qa"])
    evidence = "\n".join(qa.get("gold_inds", {}).values())
    steps = "\n".join(f"{s['op'].split('-')[0].rstrip('0123456789')}({s['arg1']}, {s['arg2']}) = {s['res']}"
                      for s in qa.get("steps", []))
    return f"Evidence:\n{evidence}\nCalculation:\n{steps}\nAnswer: {answer_text(doc)}"


def doc_to_text(doc):
    return f"{format_question(doc)}\nAnswer:"


def doc_to_target(doc):
    return str(answer_text(doc))


_NUM = re.compile(r"-?\$?\s?(?:\d[\d,]*\.?\d*|\.\d+)\s?%?")


def _to_float(s):
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    pct = s.endswith("%")
    return float(s.rstrip("%")), pct


def _close(a, b):
    return abs(a - b) <= max(0.005, 0.01 * abs(b))


def extract(text):
    """The first number after the last 'answer', else the last number in the text."""
    m = list(re.finditer(r"answer\s*(?:is)?\s*:?", text, flags=re.I))
    if m:
        nums = _NUM.findall(text[m[-1].end():])
        if nums:
            return nums[0]
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def process_results(doc, results):
    gold = gold_value(doc)
    text = results[0]
    if isinstance(gold, str):  # yes / no
        m = re.search(r"\b(yes|no)\b", text, flags=re.I)
        pred = m.group(1).lower() if m else None
        return {"acc": float(pred == gold.lower()), "extracted": float(pred is not None)}
    raw = extract(text)
    if raw is None:
        return {"acc": 0.0, "extracted": 0.0}
    try:
        v, pct = _to_float(raw)
    except ValueError:
        return {"acc": 0.0, "extracted": 0.0}
    golds = [gold]
    ans = _j(doc["qa"]).get("answer", "")
    if _NUM.fullmatch(ans.strip()):
        g, g_pct = _to_float(ans.strip())
        golds.append(g / 100 if g_pct else g)
    ok = any(_close(c, g) for g in golds
             for c in ([v, v / 100] if (pct or abs(g) < 1) else [v]))
    return {"acc": float(ok), "extracted": 1.0}
