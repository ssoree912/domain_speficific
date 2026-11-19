"""PANGEA (NeurIPS 2025) test-bench protocol for GSM8K / MedQA / FinQA.

Ported from third_party/pangea/code/test_bench/eval_{gsm8k,medqa,finqa}.py: the Alpaca
prompts, the ground-truth parsing and the rule-based answer extraction / comparison those
scripts apply to the raw generation. That rule is what ``regex_acc`` reports; PANGEA's
headline numbers instead come from the LLM answer parser (scripts/pangea_parse.py), which
reuses ``ground_truth`` from here so both stages read the label the same way.
"""
import re
from fractions import Fraction

_ALPACA = ("Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n{instruction}\n\n### Response:")
PROMPTS = {
    "gsm8k": _ALPACA + " Let's think step by step.",
    "finqa": _ALPACA + " Let's think step by step.",
    "medqa": _ALPACA + "\n",
}


def ground_truth(dataset, doc):
    """The label exactly as each eval_*.py stores it in its GT column."""
    response = doc["response"]
    if dataset == "gsm8k":
        return int(response.split("#### ")[1].replace(",", ""))
    if dataset == "medqa":
        return response.split("#### ")[1].split(".")[0].strip()
    if isinstance(response, (int, float)) or (
            isinstance(response, str) and response.replace(".", "", 1).replace("-", "", 1).isdigit()):
        return float(response)
    return response


def _first_number(text, rounding):
    match = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    if not match:
        return None
    num_str = match.group()
    try:
        value = float(Fraction(num_str)) if "/" in num_str else float(num_str)
    except Exception:
        return None
    return round(value) if rounding else value


def extract_gsm8k(completion):
    if "The answer is:" in completion:
        text = completion.split("The answer is:")[-1].strip()
    elif "Final answer is:" in completion:
        text = completion.split("Final answer is:")[-1].strip()
    elif "answer is:" in completion:
        text = completion.split("answer is:")[-1].strip()
    else:
        text = completion.strip()
    return _first_number(text, rounding=True)


def extract_medqa(completion):
    text = completion.split("The answer is: ")
    if len(text) > 1:
        match = re.search(r"\b([A-Z])\.\s", text[-1].strip())
        if match:
            return match.group(1).strip()
    return None


def extract_finqa(completion):
    """Note PANGEA's substring test: any "no" in the text (e.g. "not") yields "no"."""
    if "The answer is:" in completion:
        text = completion.split("The answer is:")[-1].strip().lower()
    else:
        text = completion.strip().lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return _first_number(text, rounding=False)


def is_correct(dataset, pred, gt):
    if pred is None:
        return False
    if dataset == "gsm8k":
        return float(pred) == float(gt)
    if dataset == "medqa":
        return pred == gt
    if isinstance(pred, str) and isinstance(gt, str):
        return pred.lower() == gt.lower()
    if isinstance(pred, (int, float)) or isinstance(gt, (int, float)):
        try:
            return round(float(pred), 4) == round(float(gt), 4)
        except (ValueError, TypeError):
            return False
    return False


EXTRACT = {"gsm8k": extract_gsm8k, "medqa": extract_medqa, "finqa": extract_finqa}


def _results(dataset, doc, results):
    pred = EXTRACT[dataset](results[0])
    return {"regex_acc": float(is_correct(dataset, pred, ground_truth(dataset, doc))),
            "regex_extracted": float(pred is not None)}


def doc_to_text_gsm8k(doc):
    return PROMPTS["gsm8k"].format(instruction=doc["query"])


def doc_to_text_medqa(doc):
    return PROMPTS["medqa"].format(instruction=doc["query"])


def doc_to_text_finqa(doc):
    return PROMPTS["finqa"].format(instruction=doc["query"])


def doc_to_target_gsm8k(doc):
    return str(ground_truth("gsm8k", doc))


def doc_to_target_medqa(doc):
    return ground_truth("medqa", doc)


def doc_to_target_finqa(doc):
    return str(ground_truth("finqa", doc))


def process_results_gsm8k(doc, results):
    return _results("gsm8k", doc, results)


def process_results_medqa(doc, results):
    return _results("medqa", doc, results)


def process_results_finqa(doc, results):
    return _results("finqa", doc, results)
