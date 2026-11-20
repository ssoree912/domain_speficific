#!/usr/bin/env bash
# Skeleton-first probe v5: v4 + steps counted as arithmetic lines (one operation per line), k uniform
# over 2-6, stratified select. Same 384 candidates as v4. Output data/synth/skel_v5/gsm8k/n64/.
#   nohup bash scripts/probes/run_skeleton_probe_v5.sh > logs/probe_skeleton_v5.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--tag v5 --v5 --n 64"
step() { echo "[$(date '+%F %T')] $*"; }
step skeleton;        $PY scripts/skeleton_gen.py skeleton $A --cand 6 --repair 2 --batch-size 32
step write qwen;      $PY scripts/skeleton_gen.py write --gen qwen $A
step write dream;     $PY scripts/skeleton_gen.py write --gen dream $A --budget 36
step annotate qwen;   $PY scripts/skeleton_gen.py annotate --gen qwen $A
step annotate dream;  $PY scripts/skeleton_gen.py annotate --gen dream $A
step verify;          $PY scripts/skeleton_gen.py verify $A
step report;          $PY scripts/skeleton_gen.py report $A
step select;          $PY scripts/skeleton_gen.py select $A --size 64
step done
