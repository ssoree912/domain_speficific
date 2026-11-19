#!/usr/bin/env bash
# 0-shot lm-eval through the Irrelevant_cfg harness (its LLaDA_cfg model wrapper and decoder).
#
#   scripts/run_eval.sh <gsm8k|medqa|finqa> [vanilla]
#
# gsm8k uses the harness's own task (eval/tasks/gsm8k.yaml); medqa / finqa are this
# project's tasks (eval/tasks). Data come from <repo>/data. Decoding defaults follow
# the harness: LLaDA low_confidence / temperature 0, Dream entropy / 0.2 / 0.95.
#
# Env:
#   MODEL_PATH   default <repo>/model/LLaDA-8B-Instruct
#   GEN_LENGTH   response canvas L, steps = L, one block (default 256, the plan's L)
#   HARNESS      Irrelevant_cfg checkout (default ~/Notebooks/dlpch.345/Irrelevant_cfg)
#   PY           python with lm-eval 0.4.x (default: the future-dllm conda env)
#   LIMIT, SHARD ("i/n"), OUT_ROOT, MAX_SEQ_LEN, DECODING, DREAM_ALG/TEMPERATURE/TOP_P
#
# Generations go to <out>/generations.jsonl; rerunning the same command resumes.
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <gsm8k|medqa|finqa> [vanilla]}"
CONFIG="${2:-vanilla}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${HARNESS:-$HOME/Notebooks/dlpch.345/Irrelevant_cfg}"
PY="${PY:-$HOME/.conda/envs/future-dllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/model/LLaDA-8B-Instruct}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"
GEN_LENGTH="${GEN_LENGTH:-256}"
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data}" && pwd)"

case "$DATASET" in
  gsm8k|medqa|finqa) TASK="local_$DATASET" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

if grep -qiE '"model_type": *"dream"' "$MODEL_PATH/config.json" 2>/dev/null; then FAMILY=dream; else FAMILY=llada; fi
DECODING="${DECODING:-$FAMILY}"
case "$DECODING" in
  llada) DECODE_ARGS=",sampler=llada"; DECODE_TAG="" ;;
  dream) DREAM_ALG="${DREAM_ALG:-entropy}" DREAM_TEMPERATURE="${DREAM_TEMPERATURE:-0.2}"
         DREAM_TOP_P="${DREAM_TOP_P:-0.95}"
         DECODE_ARGS=",sampler=dream,alg=$DREAM_ALG,temperature=$DREAM_TEMPERATURE,top_p=$DREAM_TOP_P"
         DECODE_TAG="_dream_${DREAM_ALG}_t${DREAM_TEMPERATURE}_p${DREAM_TOP_P}" ;;
  *) echo "unknown decoding: $DECODING" >&2; exit 1 ;;
esac
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
[ "$MAX_SEQ_LEN" != 4096 ] && DECODE_ARGS="$DECODE_ARGS,max_seq_len=$MAX_SEQ_LEN"

OUT="$OUT_ROOT/$(basename "$MODEL_PATH")/$DATASET/${CONFIG}_len${GEN_LENGTH}${DECODE_TAG}${LIMIT:+_limit$LIMIT}"
TASKS_DIR="$OUT/tasks${SHARD:+_shard${SHARD%/*}}"
mkdir -p "$TASKS_DIR"
cp "$HARNESS"/eval/tasks/local_gsm8k_utils.py "$ROOT"/eval/tasks/local_*_utils.py "$TASKS_DIR/"
for y in "$HARNESS"/eval/tasks/gsm8k.yaml "$ROOT"/eval/tasks/*.yaml; do
  sed "s|DATA_DIR|$DATA_DIR|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

export LLADA_CFG_STORE="$OUT/generations.jsonl" LLADA_CFG_SHARD="${SHARD:-0/1}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
RUN_OUT="$OUT/out"
[ -n "${SHARD:-}" ] && RUN_OUT="$OUT/shard_out/${SHARD%/*}"
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "$(basename "$MODEL_PATH") $DATASET config=$CONFIG gen_length=$GEN_LENGTH decoding=$DECODING samples=${LIMIT:-all} shard=${SHARD:-none} -> $OUT"
cd "$HARNESS"
"$PY" eval/lm_eval_model.py \
  --model LLaDA_cfg \
  --model_args "pretrained=$MODEL_PATH,config=$CONFIG,gen_length=$GEN_LENGTH$DECODE_ARGS" \
  --tasks "$TASK" --num_fewshot 0 \
  --include_path "$TASKS_DIR" \
  --batch_size 1 \
  --log_samples \
  "${LIMIT_ARGS[@]}" \
  --output_path "$RUN_OUT"
