#!/usr/bin/env bash
# 0-shot PANGEA-protocol evaluation of an AR student model with lm-eval's hf backend.
#
#   scripts/run_student_eval.sh <gsm8k|medqa|finqa>
#
# Tasks are eval/pangea/pangea_<dataset>.yaml: PANGEA's Alpaca prompt, stop strings and
# greedy decoding on its test jsonl; they report PANGEA's rule-based regex_acc. The paper's
# metric needs the LLM answer parser afterwards: scripts/pangea_parse.py <out dir>.
#
# Env:
#   MODEL_PATH    default <repo>/model/Llama-3.2-1B
#   MODEL_NAME    results directory name (default basename of MODEL_PATH)
#   BATCH_SIZE    default 16 ("auto" probes with the model's 128k max length)
#   MAX_GEN_TOKS  overrides the task's 1024 (PANGEA used max_tokens=seq_len, 4096)
#   PANGEA_DATA   default <repo>/third_party/pangea/code/test_bench/data/test
#   PY            python with lm-eval 0.4.x (default: the future-dllm conda env)
#   LIMIT, OUT_ROOT
#
# Output: <OUT_ROOT>/student/<model>/<dataset>/len<MAX_GEN_TOKS>[_limit<LIMIT>]/out
set -euo pipefail

DATASET="${1:?usage: run_student_eval.sh <gsm8k|medqa|finqa>}"
case "$DATASET" in
  gsm8k|medqa|finqa) TASK="pangea_$DATASET" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/.conda/envs/future-dllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/model/Llama-3.2-1B}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
PANGEA_DATA="$(cd "${PANGEA_DATA:-$ROOT/third_party/pangea/code/test_bench/data/test}" && pwd)"

OUT="$OUT_ROOT/student/$MODEL_NAME/$DATASET/len${MAX_GEN_TOKS}${LIMIT:+_limit$LIMIT}"
TASKS_DIR="$OUT/tasks"
mkdir -p "$TASKS_DIR"
cp "$ROOT"/eval/pangea/pangea_utils.py "$TASKS_DIR/"
for y in "$ROOT"/eval/pangea/*.yaml; do
  sed "s|PANGEA_DATA|$PANGEA_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "$MODEL_NAME $DATASET max_gen_toks=$MAX_GEN_TOKS batch=$BATCH_SIZE samples=${LIMIT:-all} -> $OUT"
"$PY" -m lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL_PATH,dtype=bfloat16" \
  --tasks "$TASK" --num_fewshot 0 \
  --include_path "$TASKS_DIR" \
  --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  "${LIMIT_ARGS[@]}" \
  --output_path "$OUT/out"
