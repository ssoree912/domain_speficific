#!/usr/bin/env bash
# 64-row probe of contrastive CFG (negative = projection prompt with τ masked) on the qtau setup:
# Qwen τ shared from G1, Dream Projection with rich anchors, PANGEA sampling, "=" banned.
# Baseline w=0 is the first 64 rows of data/synth/g3_qtau (same settings, no CFG).
#   nohup bash scripts/probes/run_cfg_probe.sh > logs/probe_cfg.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--domain gsm8k --n 1000 --limit 64 --chunk 64"
for spec in "w1_l05 1 0.5" "w2_l05 2 0.5" "w1_l10 1 1.0"; do
  read -r tag w late <<<"$spec"
  echo "[$(date '+%F %T')] cfg_$tag projection (w=$w late=$late)"
  $PY scripts/pangea_gen.py projection --cond g3 --tag "cfg_$tag" $COMMON --batch-size 4 \
    --writer-from g1 --projection-anchors rich --dream-sampling pangea --cfg-w "$w" --cfg-late "$late"
  echo "[$(date '+%F %T')] cfg_$tag annotation"
  $PY scripts/pangea_gen.py annotation --cond g3 --tag "cfg_$tag" $COMMON --batch-size 16
  $PY scripts/pangea_gen.py export --cond g3 --tag "cfg_$tag" --domain gsm8k --n 1000
done
echo "[$(date '+%F %T')] probe done"
