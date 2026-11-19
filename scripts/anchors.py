"""G3 anchors: text fixed on Dream's response canvas before infilling (docs/PLAN.md).

Each plan is a list of (anchor text, span before it): the canvas is
  [span_1 masks] anchor_1 [span_2 masks] anchor_2 ...
so a span is the room the model has to write before the anchor. A span of 0 places the
anchor right at the start of the canvas (or right after the previous anchor).

  writer      PANGEA's output format with its fixed parts as anchors (settled on the GSM8K
              Writer probe, 2026-10-10: field names alone let Dream copy content across fields)
  projection  "numbers": the values in τ's Symbols, in order, then "?" (GSM8K: the spans
              share a budget close to the seed questions' median length of 60 tokens)
              "rich": τ's concrete content, so what came from the general example has to
              appear in the question: an entity of the Scenario (a proper noun, else a word
              the Scenario shares with the general example), each Symbols value with its unit
              ("3000 km"), the Target's description, then "?". Values τ computes (the
              answer, intermediate results; see derived()) are not anchored, since a fixed
              value always appears in the question.
"""

from __future__ import annotations

import re

# (anchor text, span cap of the text that follows it); converted to plan order below
WRITER = {
    "gsm8k": [
        ("\n- Scenario - ", 32),
        ("\n- Symbols - A = ", 40),
        ("\n- Equations - (1) ", 12), (" (2) ", 12), (" (3) ", 12), (" (4) ", 12),
        ("\n- Target - T = ", 16),
        ("\n- Plan - (1) ", 10), (" → (2) ", 10), (" → (3) ", 10), (" → (4) ", 10),
    ],
}

PROJECTION_BUDGET = {"gsm8k": 64}
MAX_NUMS = 6
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def writer_plan(domain: str) -> list[tuple[str, int]]:
    """Header-first spec -> plan: each anchor's span is the previous field's cap."""
    spec = WRITER[domain]
    return [(text, 0 if k == 0 else spec[k - 1][1]) for k, (text, _) in enumerate(spec)] + [("", spec[-1][1])]


def tau_numbers(tau: str) -> list[str]:
    """Values of τ's Symbols field in order ("A = 84 yards" -> "84"), deduplicated, at most MAX_NUMS."""
    m = re.search(r"Symbols\s*[-:–]\s*(.*)", tau)
    sym = m.group(1) if m else ""
    vals = [_NUM.match(v.strip()).group(0) for v in re.findall(r"=\s*([^,;]*)", sym)
            if _NUM.match(v.strip())] or _NUM.findall(sym)
    out = []
    for v in vals:
        v = v.rstrip(",")
        if v not in out:
            out.append(v)
    return out[:MAX_NUMS]


def _field(tau, name):
    m = re.search(rf"{name}\s*[-:–]\s*([^\n]*)", tau)
    return m.group(1).strip() if m else ""


_STOP = {"You", "Your", "The", "This", "That", "These", "They", "There", "When", "After", "Before",
         "If", "In", "On", "At", "For", "During", "Each", "Every", "Some", "One", "Two", "Three"}


def tau_entity(tau, general=""):
    """A proper noun of the Scenario, else its longest word that the general example also has."""
    scen = re.sub(r"[\[\]]", "", _field(tau, "Scenario"))
    for m in re.finditer(r"\b[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)*", scen):
        words = [w for w in m.group(0).split() if w not in _STOP]
        sentence_start = m.start() == 0 or re.search(r"[.!?]\s*$", scen[:m.start()])
        if words and (len(words) > 1 or not sentence_start):  # a lone capitalised first word is no name
            return " ".join(words)
    gen = set(re.findall(r"[a-z]{5,}", general.lower()))
    shared = [w for w in re.findall(r"[A-Za-z]{5,}", scen) if w.lower() in gen]
    return max(shared, key=len) if shared else ""


_OP = re.compile(r"[+\-×x*/÷]")


def _num(v):
    m = re.match(r"\$?(\d[\d,]*(?:\.\d+)?)", v.strip())
    return float(m.group(1).replace(",", "")) if m else None


def derived(tau):
    """Letters and values τ computes rather than gives: the Target's letter or value, letters
    defined by an expression ("C = A - B"), and numbers that are an expression's result
    ("B - A = 24 yards"). Anchoring them would put the answer or an intermediate result in
    the question."""
    letters, values = set(), set()
    tgt = _field(tau, "Target")
    m = re.match(r"([A-Z])\s*=\s*(.*)", tgt)
    if m:
        if m.group(1) != "T":
            letters.add(m.group(1))
        rhs = m.group(2).strip()
        if re.fullmatch(r"[A-Z]", rhs.split(" ")[0]):
            letters.add(rhs.split(" ")[0])
        elif _num(rhs) is not None:
            values.add(_num(rhs))
    for eq in re.split(r"\(\d\)", _field(tau, "Equations")):
        if "=" not in eq:
            continue
        lhs, rhs = (x.strip() for x in eq.split("=", 1))
        if re.fullmatch(r"[A-Z]", lhs) and _OP.search(rhs) and re.search(r"\b[A-Z]\b|\d", rhs):
            letters.add(lhs)
        if _OP.search(lhs):
            if re.fullmatch(r"[A-Z]", rhs.split(" ")[0]):
                letters.add(rhs.split(" ")[0])
            elif _num(rhs) is not None:
                values.add(_num(rhs))
    return letters, values


def tau_quantities(tau, given_only=False):
    """Symbols values with their unit: "A = 3000 km - distance" -> "3000 km" (at most MAX_NUMS).
    given_only drops what derived() finds; if that leaves nothing (τ whose equations define
    every symbol from another), only the values equal to the answer are dropped."""
    sym = re.sub(r"[\[\]]", "", _field(tau, "Symbols"))
    items = []
    for letter, v in re.findall(r"\b([A-Z])\s*=\s*([^,;]+?)(?=\s+[-–]\s+|,|;|$)", sym):
        m = re.match(r"(\$?\d[\d,]*(?:\.\d+)?%?(?:\s+[A-Za-z/°%$]+){0,2})", v.strip())
        if m and m.group(1).strip() not in [q for _, q in items]:
            items.append((letter, m.group(1).strip()))
    if not items:
        return [] if given_only else tau_numbers(tau)
    if given_only:
        letters, values = derived(tau)
        kept = [(l, q) for l, q in items if l not in letters and _num(q) not in values]
        if not kept:
            answer = derived_answer(tau)
            kept = [(l, q) for l, q in items if answer is None or _num(q) != answer]
        items = kept
    return [q for _, q in items][:MAX_NUMS]


def derived_answer(tau):
    """The Target's numeric value, if it states one ("T = 2000 km - ..." -> 2000)."""
    m = re.match(r"[A-Z]\s*=\s*(.*)", _field(tau, "Target"))
    return _num(m.group(1)) if m else None


def tau_target(tau):
    """The Target's description ("T = total number of pilots" -> "total number of pilots")."""
    t = _field(tau, "Target")
    t = re.sub(r"^[A-Z]\s*=\s*", "", t)
    t = re.sub(r"^\$?[\d.,]+\s*\S*\s+[-–]\s+", "", t)  # "2000 km - remaining distance" -> description
    t = re.sub(r"[\[\]?.]", "", t).strip()
    words = t.split()
    if (len(words) < 2 or len(words) > 10 or re.match(r"[\d$]", t) or re.search(r"[=+×÷]", t)
            or re.search(r"\b[A-Z]\b", t) or "numeric goal" in t):
        return ""
    return t


def projection_plan(domain: str, tau: str, general: str = "", style: str = "numbers") -> list[tuple[str, int]]:
    if domain != "gsm8k":
        raise NotImplementedError(f"G3 projection anchors for {domain} are not designed yet")
    if style == "numbers":
        nums = tau_numbers(tau)
        span = max(8, PROJECTION_BUDGET[domain] // (len(nums) + 1))
        return [(" " + v, span) for v in nums] + [("?", span)]
    ent, qty, tgt = tau_entity(tau, general), tau_quantities(tau, given_only=True), tau_target(tau)
    texts = ([" " + ent] if ent else []) + [" " + q for q in qty] + ([" " + tgt] if tgt else [])
    span = max(8, PROJECTION_BUDGET[domain] // (len(texts) + 1))
    plan = [(t, span) for t in texts]
    if ent:  # the entity opens the story: little room before it
        plan[0] = (plan[0][0], 4)
    return plan + [("?", 6 if tgt else span)]


def canvas(plan, n_tokens) -> tuple[dict[int, str], int]:
    """Plan -> ({offset: anchor text}, canvas length L); ``n_tokens(text)`` counts anchor tokens."""
    anchors, off = {}, 0
    for text, span in plan:
        off += span
        if text:
            anchors[off] = text
            off += n_tokens(text)
    return anchors, off


def clean(text: str) -> str:
    """The model sometimes writes its own '?' right before the anchored one."""
    return re.sub(r"\?(\s*\?)+", "?", text).strip()
