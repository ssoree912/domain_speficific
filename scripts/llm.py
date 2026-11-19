"""Minimal chat-LLM backends for the data pipeline.

  hf:<model dir>                        local transformers model, batched greedy decoding
                                        (or sampling, with an optional assistant prefix)
  openai:<base url>|<model name>        any OpenAI-compatible server (vLLM, API);
                                        key from OPENAI_API_KEY (may be empty for vLLM)

    llm = load_llm("hf:model/Qwen2.5-7B-Instruct")
    outs = llm.chat([[{"role": "user", "content": "..."}], ...], max_new_tokens=512)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class HFChat:
    def __init__(self, path: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.batch_size = batch_size
        self.name = f"hf:{path}"

    def render(self, conversation, assistant_prefix=""):
        """Chat-templated prompt; ``assistant_prefix`` pre-fills the start of the reply."""
        return self.tok.apply_chat_template(conversation, tokenize=False,
                                            add_generation_prompt=True) + assistant_prefix

    def chat(self, conversations, max_new_tokens=512, assistant_prefix="", **sampling):
        """Greedy unless ``sampling`` gives generate() kwargs such as do_sample / temperature.
        ``assistant_prefix`` is one string for all rows or a list with one per row."""
        if isinstance(assistant_prefix, str):
            assistant_prefix = [assistant_prefix] * len(conversations)
        prompts = [self.render(c, p) for c, p in zip(conversations, assistant_prefix)]
        decoding = sampling or dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, **decoding,
                                          pad_token_id=self.tok.pad_token_id)
            for i, g in zip(idx, gen[:, enc.input_ids.shape[1]:]):
                out[i] = self.tok.decode(g, skip_special_tokens=True)
        return out


class OpenAIChat:
    def __init__(self, spec: str, workers: int = 16):
        self.base, self.model = spec.split("|", 1)
        self.base = self.base.rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.workers = workers
        self.name = f"openai:{self.model}"

    def _one(self, messages, max_new_tokens):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": max_new_tokens}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def chat(self, conversations, max_new_tokens=512):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda c: self._one(c, max_new_tokens), conversations))


def load_llm(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "hf":
        return HFChat(rest)
    if kind == "openai":
        return OpenAIChat(rest)
    raise ValueError(f"unknown LLM spec {spec!r} (use hf:<dir> or openai:<url>|<model>)")


def parse_json(text: str):
    """The first parseable JSON object in a reply (tolerates ```json fences and prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        depth = 0
        for e in range(s, len(text)):
            depth += {"{": 1, "}": -1}.get(text[e], 0)
            if depth == 0:
                try:
                    return json.loads(text[s:e + 1])
                except json.JSONDecodeError:
                    break
    return None
