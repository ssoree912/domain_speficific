"""MedQA (USMLE, 4 options): prompt formatting and scoring.

format_question / format_answer are also used by scripts/make_seeds.py, so the seed
questions and the benchmark prompts have the same shape.
"""
import re

LETTERS = "ABCD"


def format_question(doc):
    opts = "\n".join(f"{k}. {doc['options'][k]}" for k in LETTERS if doc["options"].get(k))
    return f"{doc['question'].strip()}\n{opts}"


def format_answer(doc):
    return f"{doc['answer_idx']}. {doc['answer']}"


def doc_to_text(doc):
    return f"Question: {format_question(doc)}\nAnswer:"


def doc_to_target(doc):
    return doc["answer_idx"]


# Ordered from most to least explicit; the first pattern that matches decides.
_PATTERNS = [
    r"answer is\s*:?\s*\(?([A-D])\b",
    r"[Aa]nswer\s*:\s*\(?([A-D])\b",
    r"^\s*\(?([A-D])[\.\):]",
    r"\b([A-D])\.\s",
    r"\(([A-D])\)",
]


def extract_choice(text, doc):
    for pat in _PATTERNS:
        m = re.search(pat, text, flags=re.M)
        if m:
            return m.group(1)
    # Fall back to the option whose text appears in the generation (first occurrence wins).
    low = text.lower()
    hits = [(low.find(doc["options"][k].lower()), k) for k in LETTERS
            if doc["options"].get(k) and doc["options"][k].lower() in low]
    return min(hits)[1] if hits else None


def process_results(doc, results):
    pred = extract_choice(results[0], doc)
    return {"acc": float(pred == doc["answer_idx"]), "extracted": float(pred is not None)}
