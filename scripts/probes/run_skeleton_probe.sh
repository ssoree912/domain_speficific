#!/usr/bin/env bash
# Skeleton-first 64-row probe (scripts/skeleton_gen.py): Qwen skeleton -> Qwen AR / Dream anchored text -> Qwen labels.
#   nohup bash scripts/probes/run_skeleton_probe.sh > logs/probe_skeleton.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[$(date '+%F %T')] $*"; }
step skeleton;        $PY scripts/skeleton_gen.py skeleton --n 64
step write qwen;      $PY scripts/skeleton_gen.py write --gen qwen
step write dream;     $PY scripts/skeleton_gen.py write --gen dream
step annotate qwen;   $PY scripts/skeleton_gen.py annotate --gen qwen
step annotate dream;  $PY scripts/skeleton_gen.py annotate --gen dream
step report;          $PY scripts/skeleton_gen.py report
step done
