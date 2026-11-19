#!/usr/bin/env bash
# Skeleton-first probe v3: v2 + no number literals in the program, numeric phrase check, skeletons
# kept only if a plain rendering solved greedily gives the program answer, per-writer verify. Output data/synth/skel_v3/gsm8k/n64/.
#   nohup bash scripts/probes/run_skeleton_probe_v3.sh > logs/probe_skeleton_v3.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--tag v3 --v3 --n 64"
step() { echo "[$(date '+%F %T')] $*"; }
step skeleton;        $PY scripts/skeleton_gen.py skeleton $A --cand 6
step write qwen;      $PY scripts/skeleton_gen.py write --gen qwen $A
step write dream;     $PY scripts/skeleton_gen.py write --gen dream $A --budget 36
step annotate qwen;   $PY scripts/skeleton_gen.py annotate --gen qwen $A
step annotate dream;  $PY scripts/skeleton_gen.py annotate --gen dream $A
step verify;          $PY scripts/skeleton_gen.py verify $A
step report;          $PY scripts/skeleton_gen.py report $A
step done
