"""The answer skeleton Φ = (a_1, ..., a_K) and how it is written into text.

Anchor k is the line "<name>:" on its own line; every anchor after the first starts on a
new line. One rendering is shared by the synthetic answers r̃ (step 4), the template
masks (step 5) and the inference canvas (step 6), so the anchor tokens are identical in
all three.

    r̃ = a_1 c_1 a_2 c_2 ... a_K c_K
    r̃ = c_0 a_1 c_1                       (lead=True: free text before the first anchor)

lead is used by the answer_only ablation, where the only anchor is the final answer and
the reasoning goes in c_0.
"""

from __future__ import annotations

# Ablation skeletons that do not come from the seeds (H5).
GENERIC = ["Step 1", "Step 2", "Answer"]
ANSWER_ONLY = ["Answer"]


def anchor_texts(names: list[str], lead: bool = False) -> list[str]:
    return [f"{n}:\n" if k == 0 and not lead else f"\n{n}:\n" for k, n in enumerate(names)]


def render(names: list[str], sections: list[str], lead: bool = False) -> str:
    """sections holds K content spans, or K + 1 with lead (c_0 first)."""
    assert len(sections) == len(names) + lead
    head = sections[0].strip() if lead else ""
    body = sections[1:] if lead else sections
    return head + "".join(a + c.strip() for a, c in zip(anchor_texts(names, lead), body))


def split(names: list[str], text: str, lead: bool = False) -> list[str] | None:
    """Inverse of render: the content spans, or None if the anchors are not all in order."""
    pos = 0
    anchors = anchor_texts(names, lead)
    starts = []
    for a in anchors:
        i = text.find(a, pos)
        if i < 0:
            return None
        starts.append(i)
        pos = i + len(a)
    out = [text[:starts[0]].strip()] if lead else []
    for k, (i, a) in enumerate(zip(starts, anchors)):
        end = starts[k + 1] if k + 1 < len(starts) else len(text)
        out.append(text[i + len(a):end].strip())
    return out
