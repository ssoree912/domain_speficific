#!/usr/bin/env bash
# Skeleton-first GSM8K 1k (v5): skeletons from a 20k plan (stop at 1600 kept) -> Qwen / Dream write -> label -> verify
# (labels resampled up to 3x against the program answer) -> select 1000 verified rows per writer with GSM8K's step distribution -> students, 3 seeds each.
# G1 (PANGEA Qwen AR) gets seeds 43 and 44 next to its existing seed-42 run, for the comparison.
# Resumable: the skeleton stage resumes by chunk; finished stages / checkpoints / evals are skipped.
# Checkpoints are deleted after their eval (disk).
#   nohup bash scripts/run_skeleton_1k.sh > logs/run_skeleton_1k.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--tag v5 --v5 --n 1000"
D=data/synth/skel_v5/gsm8k/n1000
SEEDS="${SEEDS:-42 43 44}"
step() { echo "[$(date '+%F %T')] $*"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

$PY scripts/pangea_gen.py plan --domain gsm8k --n 20000
step skeleton
$PY scripts/skeleton_gen.py skeleton $A --plan gsm8k_n20000.jsonl --cand 20 --repair 2 --keep-all --max-kept 1600 --batch-size 32
NSK=$(lines $D/skeleton.jsonl)
step "skeletons kept: $NSK"
for GEN in qwen dream; do
  if [ "$(lines $D/write_$GEN.jsonl)" != "$NSK" ]; then
    step "write $GEN"; EXTRA=""; [ $GEN = dream ] && EXTRA="--budget 36"
    $PY scripts/skeleton_gen.py write --gen $GEN $A $EXTRA
  fi
  if [ "$(lines $D/annotate_$GEN.jsonl)" != "$NSK" ]; then
    step "annotate $GEN"; $PY scripts/skeleton_gen.py annotate --gen $GEN $A --resample 3
  fi
done
if [ "$(lines $D/verify_qwen.jsonl)" != "$NSK" ] || [ "$(lines $D/verify_dream.jsonl)" != "$NSK" ]; then
  step verify; $PY scripts/skeleton_gen.py verify $A
fi
step report; $PY scripts/skeleton_gen.py report $A
step select; $PY scripts/skeleton_gen.py select $A --size 1000

train_eval() {  # name data seed
  local NAME=$1 DATA=$2 SEED=$3 CKPT="ckpt/$1"
  local OUT="results/student/$NAME/gsm8k/len1024/out"
  if ls "$OUT"/*/pangea_acc_* >/dev/null 2>&1; then step "$NAME already evaluated"; return; fi
  if [ ! -f "$CKPT/config.json" ]; then
    step "train $NAME"; $PY scripts/train_student.py --data "$DATA" --out "$CKPT" --seed "$SEED"
  fi
  step "eval $NAME"
  MODEL_PATH="$ROOT/$CKPT" MODEL_NAME="$NAME" bash scripts/run_student_eval.sh gsm8k
  $PY scripts/pangea_parse.py "$OUT"
  rm -rf "$CKPT"
}
for SEED in $SEEDS; do
  for GEN in qwen dream; do
    train_eval "skel5_${GEN}_gsm8k_n1000_s$SEED" "$D/train_$GEN.jsonl" "$SEED"
  done
  [ "$SEED" != 42 ] && train_eval "g1_gsm8k_n1000_s$SEED" data/synth/g1/gsm8k/n1000/train.jsonl "$SEED"
done
step done
