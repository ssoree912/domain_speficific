#!/usr/bin/env python
"""Download the datasets this project needs that are not already in Future_dLLM/data.

Layout (same as Future_dLLM/data):
  data/raw/<name>/            original files as published
  data/train/<name>/*.parquet seed pool (train split)
  data/eval/<name>/*.parquet  benchmark (test split)
  data/general/cot_collection/  general data D_g (unfiltered)

GSM8K is symlinked from Future_dLLM/data, not downloaded here.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent / "data"

FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"


def write_source(out: Path, source: str, rows: dict, **extra) -> None:
    meta = {"source_dataset": source, "rows": rows, "downloaded": date.today().isoformat(), **extra}
    (out / "SOURCE.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def medqa() -> None:
    repo = "GBaker/MedQA-USMLE-4-options"
    raw = ROOT / "raw" / "medqa"
    for split, fname in [("train", "phrases_no_exclude_train.jsonl"), ("test", "phrases_no_exclude_test.jsonl")]:
        path = hf_hub_download(repo, fname, repo_type="dataset", local_dir=raw)
        df = pd.read_json(path, lines=True)
        out = ROOT / ("train" if split == "train" else "eval") / "medqa"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{split}.parquet", index=False)
        write_source(out, repo, {f"{split}.parquet": len(df)}, file=fname)
        print(f"medqa {split}: {len(df)} rows")


def finqa() -> None:
    raw = ROOT / "raw" / "finqa"
    raw.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        dst = raw / f"{split}.json"
        if not dst.exists():
            urllib.request.urlretrieve(FINQA_URL.format(split=split), dst)
        records = json.loads(dst.read_text())
        # Nested fields (qa, table, annotation) are kept as JSON strings so parquet stays flat.
        df = pd.DataFrame([{k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                            for k, v in r.items()} for r in records])
        out = ROOT / ("eval" if split == "test" else "train") / "finqa"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{split}.parquet", index=False)
        print(f"finqa {split}: {len(df)} rows")
    for sub, splits in [("train", ["train", "dev"]), ("eval", ["test"])]:
        out = ROOT / sub / "finqa"
        rows = {f"{s}.parquet": len(pd.read_parquet(out / f"{s}.parquet")) for s in splits}
        write_source(out, "czyssrs/FinQA (GitHub)", rows, url=FINQA_URL,
                     note="qa/table/pre_text/post_text non-string fields stored as JSON strings")


def cot_collection() -> None:
    repo = "kaist-ai/CoT-Collection"
    raw = ROOT / "raw" / "cot_collection"
    path = hf_hub_download(repo, "data/CoT_collection_en.json", repo_type="dataset", local_dir=raw)
    data = json.loads(Path(path).read_text())
    # Released as {id: {source, target, rationale, task, config, prompt}}.
    df = pd.DataFrame([{"id": k, **v} for k, v in data.items()])
    out = ROOT / "general" / "cot_collection"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "en.parquet", index=False)
    write_source(out, repo, {"en.parquet": len(df)}, file="data/CoT_collection_en.json",
                 note="unfiltered; math/medical/finance removal happens in the pipeline")
    print(f"cot_collection: {len(df)} rows")


PARTS = {"medqa": medqa, "finqa": finqa, "cot_collection": cot_collection}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    for p in ap.parse_args().parts:
        PARTS[p]()
