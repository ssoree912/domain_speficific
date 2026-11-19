#!/usr/bin/env bash
# GSM8K 1k, G1 (Qwen AR) vs G3 (Dream + anchors): generate -> label -> export -> train -> eval.
# Resumable: generation stages skip finished uids; a finished checkpoint or eval is not redone.
#   nohup bash scripts/run_gsm8k_1k.sh > logs/run_gsm8k_1k.log 2>&1 &
# G3 Projection only, on G1's τ, rich anchors, T 0.9 (data/synth/g3_qtau):
#   CONDS=g3 TAG=qtau STAGES=projection GEN_ARGS="--writer-from g1 --projection-anchors rich --dream-sampling pangea" \
#     nohup bash scripts/run_gsm8k_1k.sh > logs/run_gsm8k_1k_g3_qtau.log 2>&1 &
# G3 with PANGEA sampling (T 0.9) into data/synth/g3_t09:
#   CONDS=g3 TAG=t09 GEN_ARGS="--dream-sampling pangea" nohup bash scripts/run_gsm8k_1k.sh > logs/run_gsm8k_1k_g3_t09.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=~/.conda/envs/future-dllm/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=1000
CONDS="${CONDS:-g1 g3}"
TAG="${TAG:-}"
GEN_ARGS="${GEN_ARGS:-}"
STAGES="${STAGES:-writer projection}"  # "projection" alone with GEN_ARGS="--writer-from g1" shares G1's τ
TAGARG=(); [ -n "$TAG" ] && TAGARG=(--tag "$TAG")
step() { echo "[$(date '+%F %T')] $*"; }

$PY scripts/pangea_gen.py plan --domain gsm8k --n $N
for COND in $CONDS; do
  DIR="$COND${TAG:+_$TAG}"
  BS=16; [ $COND = g3 ] && BS=4
  for STAGE in $STAGES; do
    step "$COND $STAGE"
    $PY scripts/pangea_gen.py $STAGE --cond $COND --domain gsm8k --n $N --batch-size $BS --chunk 64 "${TAGARG[@]}" $GEN_ARGS
  done
  step "$COND annotation"
  $PY scripts/pangea_gen.py annotation --cond $COND --domain gsm8k --n $N --batch-size 16 --chunk 64 "${TAGARG[@]}"
  $PY scripts/pangea_gen.py export --cond $COND --domain gsm8k --n $N "${TAGARG[@]}"

  CKPT="ckpt/${DIR}_gsm8k_n$N"
  if [ ! -f "$CKPT/config.json" ]; then
    step "$COND train"
    $PY scripts/train_student.py --data "data/synth/$DIR/gsm8k/n$N/train.jsonl" --out "$CKPT"
  fi
  NAME="${DIR}_gsm8k_n$N"
  OUT="results/student/$NAME/gsm8k/len1024/out"
  if ! ls "$OUT"/*/pangea_acc_* >/dev/null 2>&1; then
    step "$COND eval"
    MODEL_PATH="$ROOT/$CKPT" MODEL_NAME="$NAME" bash scripts/run_student_eval.sh gsm8k
    $PY scripts/pangea_parse.py "$OUT"
  fi
done
step done
