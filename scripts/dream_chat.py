"""Dream-v0-Instruct as a chat generator, with optional anchors on the response canvas.

The reply is a fixed canvas of L masked tokens after the chat-templated prompt (plus an
optional assistant prefix). Anchors are {offset in the canvas: text}; they are written into
the canvas before the first step and, since Dream only ever updates mask positions, stay
fixed while the rest is infilled. Without anchors this is plain Dream decoding (G2); with
them it is the anchored infilling of G3. The canvas end may be filled with EOS, which is
cut from the returned text.

Batching uses Dream's own diffusion_generate (left padding + attention mask). Its per-step
unmask count is the batch mean of the mask counts, which is only valid when every row has
the same number of masks, so rows are grouped by mask count before batching.

    gen = DreamChat("model/Dream-v0-Instruct-7B")
    outs = gen.chat(convs, gen_length=256, steps=256, anchors=[{0: "Question: "}, ...])
"""

from __future__ import annotations

from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer

# Tokenizing an anchor alone treats its leading whitespace as sequence-initial; a throwaway
# prefix is prepended and its tokens dropped (same approach as Irrelevant_cfg's templates).
_ANCHOR_PREFIXES = (".", "\n", " ", "a")


class DreamChat:
    def __init__(self, path: str, batch_size: int = 8):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.padding_side = "left"
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to("cuda").eval()
        self.mask_id = self.model.config.mask_token_id
        self.batch_size = batch_size
        self.name = f"dream:{path}"

    def anchor_ids(self, text: str) -> list[int]:
        for prefix in _ANCHOR_PREFIXES:
            n = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
            ids = self.tok(prefix + text, add_special_tokens=False)["input_ids"][n:]
            if self.tok.decode(ids) == text:
                return ids
        raise ValueError(f"cannot tokenize anchor {text!r} losslessly")

    def _canvas_anchors(self, anchors: dict[int, str], gen_length: int) -> list[tuple[int, list[int]]]:
        spans, end = [], 0
        for off, text in sorted(anchors.items()):
            ids = self.anchor_ids(text)
            if off < end or off + len(ids) > gen_length:
                raise ValueError(f"anchor {text!r} at {off} overlaps or overflows L={gen_length}")
            spans.append((off, ids))
            end = off + len(ids)
        return spans

    def banned_ids(self, substrings) -> list[int]:
        """Vocabulary ids whose text contains any of ``substrings`` (e.g. "=")."""
        key = tuple(substrings)
        if getattr(self, "_banned_key", None) != key:
            vocab = range(len(self.tok))
            self._banned = [i for i in vocab if any(s in self.tok.decode([i]) for s in substrings)]
            self._banned_key = key
        return self._banned

    def _span_tokens(self, prompt: str, text: str) -> list[int]:
        """Token indices (in the unpadded prompt) that overlap the first occurrence of ``text``."""
        start = prompt.find(text)
        if start < 0:
            raise ValueError(f"CFG text not found in prompt: {text[:60]!r}")
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        full, pre, upto = enc(prompt), enc(prompt[:start]), enc(prompt[:start + len(text)])
        lcp = lambda a, b: next((k for k, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
        # Dream's tokenizer is a slow one (no offset mapping): the span is where the full prompt
        # stops agreeing with the text before τ, up to where it stops agreeing with the text
        # through τ (plus one token if a merge straddles τ's end)
        a, b = lcp(full, pre), lcp(full, upto)
        if b < len(upto):
            b += 1
        return list(range(a, min(b, len(full))))

    def chat(self, conversations, gen_length=256, steps=None, assistant_prefix="", anchors=None,
             temperature=0.9, top_p=0.7, top_k=None, alg="entropy", ban=None,
             cfg_w=0.0, cfg_texts=None, cfg_late=1.0):
        """``ban``: substrings the model may not write anywhere in the canvas (anchors excepted).

        Contrastive CFG (``cfg_w`` > 0): the negative branch is the same sequence with the tokens
        of ``cfg_texts[i]`` (a span of row i's prompt, e.g. τ) replaced by [MASK], so canvas
        positions line up. Guided logits l_u + (1 + w)(l_c - l_u) are used once the canvas's
        remaining mask fraction is at most ``cfg_late`` (1.0 = every step); this costs one extra
        forward per guided step."""
        steps = steps or gen_length
        banned = torch.tensor(self.banned_ids(ban), device="cuda") if ban else None
        anchors = anchors or [{}] * len(conversations)
        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                   + assistant_prefix for c in conversations]
        cfg_tok = ([self._span_tokens(p, t) for p, t in zip(prompts, cfg_texts)]
                   if cfg_w and cfg_texts else None)
        spans = [self._canvas_anchors(a, gen_length) for a in anchors]
        groups = defaultdict(list)  # masks per row -> row indices
        for i, sp in enumerate(spans):
            groups[gen_length - sum(len(ids) for _, ids in sp)].append(i)

        out = [None] * len(prompts)
        for rows in groups.values():
            rows.sort(key=lambda i: len(prompts[i]))
            for s in range(0, len(rows), self.batch_size):
                idx = rows[s:s + self.batch_size]
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
                plen = enc.input_ids.shape[1]

                def place(step, x, logits, idx=idx, plen=plen):
                    if step is None:
                        for r, i in enumerate(idx):
                            for off, ids in spans[i]:
                                x[r, plen + off:plen + off + len(ids)] = torch.tensor(ids, device=x.device)
                    return x

                neg_pos = None
                if cfg_tok is not None:
                    # Dream's own attention setup (generation_utils._sample), for the extra forward
                    am = enc.attention_mask
                    if torch.any(am == 0):
                        am = torch.nn.functional.pad(am, (0, gen_length), value=1)
                        tok_idx = am.long().cumsum(-1) - 1
                        tok_idx.masked_fill_(am == 0, 1)
                        attn4 = torch.logical_and(am.unsqueeze(1).unsqueeze(-2), am.unsqueeze(1).unsqueeze(-1))
                    else:
                        tok_idx, attn4 = None, "full"
                    neg_pos = [(r, (plen - int(enc.attention_mask[r].sum())) + k)
                               for r, i in enumerate(idx) for k in cfg_tok[i]]
                    rows_t = torch.tensor([r for r, _ in neg_pos], device="cuda")
                    cols_t = torch.tensor([c for _, c in neg_pos], device="cuda")
                    n_canvas_masks = gen_length - min(sum(len(ids) for _, ids in spans[i]) for i in idx)

                def no_banned(step, x, logits, plen=plen):
                    if neg_pos is not None:
                        frac = (x[:, plen:] == self.mask_id).sum(1).max().item() / max(1, n_canvas_masks)
                        if frac <= cfg_late:
                            xu = x.clone()
                            xu[rows_t, cols_t] = self.mask_id
                            lu = self.model(xu, attn4, tok_idx).logits
                            lu = torch.cat([lu[:, :1], lu[:, :-1]], dim=1)
                            logits = lu + (1 + cfg_w) * (logits - lu)
                    if banned is not None:
                        logits[:, :, banned] = -float("inf")
                    return logits

                with torch.inference_mode():
                    seq = self.model.diffusion_generate(
                        enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=gen_length,
                        steps=steps, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg,
                        output_history=False, return_dict_in_generate=False,
                        generation_tokens_hook_func=place, generation_logits_hook_func=no_banned)
                for r, i in enumerate(idx):
                    out[i] = self.decode(seq[r, plen:].tolist())
        return out

    def decode(self, ids: list[int]) -> str:
        """Drop EOS / pad tokens anywhere in the canvas (EOS may sit between anchored sections)."""
        drop = {self.tok.eos_token_id, self.tok.pad_token_id, self.mask_id}
        return self.tok.decode([t for t in ids if t not in drop], skip_special_tokens=True).strip()
