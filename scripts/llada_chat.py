"""LLaDA-8B-Instruct as a chat generator, with optional anchors on the response canvas.

Decoding follows LLaDA's reference generate.py: the canvas is split into blocks of
``block_length`` decoded left to right (semi-autoregressive); inside a block each step
commits the masked positions with the highest confidence (softmax probability of the
sampled token, "low_confidence" remasking), the step count split evenly over blocks.
Sampling is Gumbel-max at ``temperature`` on logits optionally filtered by top-k / top-p
(PANGEA's sampling settings). Anchors are written into the canvas before decoding; only
mask positions are ever updated, so they stay fixed and every block sees all of them.

Batches are left-padded with an attention mask; per-row commit counts come from each row's
own mask count per block, so rows with different anchors can share a batch.

    gen = LLaDAChat("model/LLaDA-8B-Instruct")
    outs = gen.chat(convs, gen_length=256, anchors=[{0: "\\n- Scenario - "}, ...])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

_ANCHOR_PREFIXES = (".", "\n", " ", "a")


def _filter(logits, top_k=None, top_p=None):
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(torch.zeros_like(drop).scatter(-1, idx, drop), float("-inf"))
    return logits


def _gumbel(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits)
    return logits.exp() / (-torch.log(noise)) ** temperature


class LLaDAChat:
    def __init__(self, path: str, batch_size: int = 8):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.padding_side = "left"
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to("cuda").eval()
        self.mask_id = self.model.config.mask_token_id
        self.batch_size = batch_size
        self.name = f"llada:{path}"

    def anchor_ids(self, text: str) -> list[int]:
        for prefix in _ANCHOR_PREFIXES:
            n = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
            ids = self.tok(prefix + text, add_special_tokens=False)["input_ids"][n:]
            if self.tok.decode(ids) == text:
                return ids
        raise ValueError(f"cannot tokenize anchor {text!r} losslessly")

    @torch.no_grad()
    def chat(self, conversations, gen_length=256, steps=None, block_length=32, assistant_prefix="",
             anchors=None, temperature=0.9, top_p=None, top_k=None):
        gen_length = -(-gen_length // block_length) * block_length  # whole blocks; extra room fills with EOS
        steps = steps or gen_length
        n_blocks = gen_length // block_length
        per_block = max(1, steps // n_blocks)
        anchors = anchors or [{}] * len(conversations)
        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                   + assistant_prefix for c in conversations]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            b, plen = enc.input_ids.shape
            x = torch.full((b, plen + gen_length), self.mask_id, dtype=torch.long, device="cuda")
            x[:, :plen] = enc.input_ids
            for r, i in enumerate(idx):
                for off, text in anchors[i].items():
                    ids = self.anchor_ids(text)
                    if off + len(ids) > gen_length:
                        raise ValueError(f"anchor {text!r} at {off} overflows L={gen_length}")
                    x[r, plen + off:plen + off + len(ids)] = torch.tensor(ids, device="cuda")
            attn = torch.cat([enc.attention_mask, torch.ones(b, gen_length, dtype=enc.attention_mask.dtype,
                                                             device="cuda")], dim=1)
            for k in range(n_blocks):
                lo, hi = plen + k * block_length, plen + (k + 1) * block_length
                n_mask = (x[:, lo:hi] == self.mask_id).sum(1)
                if n_mask.sum() == 0:
                    continue
                # commits per step for each row: its block masks split evenly, remainder first
                sched = (n_mask[:, None] // per_block).repeat(1, per_block)
                for r in range(b):
                    sched[r, : n_mask[r] % per_block] += 1
                for t in range(per_block):
                    blk = x[:, lo:hi]
                    masked = blk == self.mask_id
                    if not masked.any():
                        break
                    logits = self.model(x, attention_mask=attn).logits[:, lo:hi].float()
                    logits = _filter(logits, top_k, top_p)
                    x0 = torch.argmax(_gumbel(logits, temperature), dim=-1)
                    conf = torch.gather(F.softmax(logits, dim=-1), -1, x0[..., None]).squeeze(-1)
                    conf = torch.where(masked, conf, torch.full_like(conf, -float("inf")))
                    for r in range(b):
                        n = int(min(sched[r, t], masked[r].sum()))
                        if n:
                            pick = torch.topk(conf[r], n).indices
                            blk[r, pick] = x0[r, pick]
            for r, i in enumerate(idx):
                out[i] = self.decode(x[r, plen:].tolist())
        return out

    def decode(self, ids: list[int]) -> str:
        """Drop EOS / pad / mask tokens anywhere in the canvas."""
        drop = {self.tok.eos_token_id, self.tok.pad_token_id, self.mask_id}
        eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int):
            drop.add(eot)
        return self.tok.decode([t for t in ids if t not in drop], skip_special_tokens=True).strip()
