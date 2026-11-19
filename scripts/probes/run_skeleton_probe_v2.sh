#!/usr/bin/env bash
# Skeleton-first probe v2: no hidden constants, question >= 4 words, setting grounded in the
# general example, Dream span budget 36 (v1: 64). Output data/synth/skel_v2/gsm8k/n64/.
#   nohup bash scripts/probes/run_skeleton_probe_v2.sh > logs/probe_skeleton_v2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--tag v2 --v2 --n 64"
step() { echo "[$(date '+%F %T')] $*"; }
step skeleton;        $PY scripts/skeleton_gen.py skeleton $A --cand 3
step write qwen;      $PY scripts/skeleton_gen.py write --gen qwen $A
step write dream;     $PY scripts/skeleton_gen.py write --gen dream $A --budget 36
step annotate qwen;   $PY scripts/skeleton_gen.py annotate --gen qwen $A
step annotate dream;  $PY scripts/skeleton_gen.py annotate --gen dream $A
step report;          $PY scripts/skeleton_gen.py report $A
step done
