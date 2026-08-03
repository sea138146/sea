#!/usr/bin/env bash
set -euo pipefail

cd /unlearning/experment_data/open-unlearning

export PYTHONPATH="src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 仅使用本地模型和数据。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="Llama-3.2-1B-Instruct"
TRAINER="SampleStopNPO"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-3.2-1B-Instruct_full"
TOFU_DATA_DIR="/data/datasets/datasets-TOFU"
RETAIN_LOGS_PATH="/model/evals/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json"

FORGET_SPLIT="forget05"
HOLDOUT_SPLIT="holdout05"
RETAIN_SPLIT="retain95"

LEARNING_RATE="1e-5"
BATCH_SIZE=1
GRAD_ACC=8
MAX_STEPS=200
SEED=0

OUTPUT_DIR="./outputs/momentum_200steps_1b_seed0"
LOG_FILE="momentum_200steps_1b_seed0.log"
TASK_NAME="unlearn_tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_momentum_smoke"

# 检查本地依赖。
test -d "${PRETRAINED_PATH}" || {
  echo "[ERROR] Model path not found: ${PRETRAINED_PATH}"
  exit 1
}

test -f "${TOFU_DATA_DIR}/${FORGET_SPLIT}.json" || {
  echo "[ERROR] Missing ${TOFU_DATA_DIR}/${FORGET_SPLIT}.json"
  exit 1
}

test -f "${TOFU_DATA_DIR}/${HOLDOUT_SPLIT}.json" || {
  echo "[ERROR] Missing ${TOFU_DATA_DIR}/${HOLDOUT_SPLIT}.json"
  exit 1
}

test -f "${TOFU_DATA_DIR}/${RETAIN_SPLIT}.json" || {
  echo "[ERROR] Missing ${TOFU_DATA_DIR}/${RETAIN_SPLIT}.json"
  exit 1
}

test -f "${RETAIN_LOGS_PATH}" || {
  echo "[ERROR] Retain evaluation log not found: ${RETAIN_LOGS_PATH}"
  exit 1
}

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "[TASK]       ${TASK_NAME}"
echo "[MODEL]      ${PRETRAINED_PATH}"
echo "[TRAINER]    ${TRAINER}"
echo "[FORGET]     ${FORGET_SPLIT}"
echo "[RETAIN]     ${RETAIN_SPLIT}"
echo "[LR]         ${LEARNING_RATE}"
echo "[BATCH]      ${BATCH_SIZE}"
echo "[GRAD ACC]   ${GRAD_ACC}"
echo "[MAX STEPS]  ${MAX_STEPS}"
echo "[OUTPUT]     ${OUTPUT_DIR}"
echo "============================================================"

python src/train.py --config-name=unlearn.yaml \
  mode=unlearn \
  experiment=unlearn/tofu/default.yaml \
  trainer="${TRAINER}" \
  model="${MODEL}" \
  task_name="${TASK_NAME}" \
  forget_split="${FORGET_SPLIT}" \
  holdout_split="${HOLDOUT_SPLIT}" \
  retain_split="${RETAIN_SPLIT}" \
  retain_logs_path="${RETAIN_LOGS_PATH}" \
  model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.model_args.torch_dtype=float16 \
  model.model_args.attn_implementation=flash_attention_2 \
  trainer.method_args.beta=0.1 \
  trainer.method_args.alpha=1.0 \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.retain_loss_type=NLL \
  trainer.method_args.ema_alpha=0.8 \
  trainer.method_args.first_epsilon=0.10 \
  trainer.method_args.warmup_observations=4 \
  trainer.method_args.patience=2 \
  trainer.method_args.log_interval=1 \
  trainer.method_args.epsilon=1.0e-8 \
  ++trainer.args.output_dir="${OUTPUT_DIR}" \
  ++trainer.args.seed="${SEED}" \
  ++trainer.args.learning_rate="${LEARNING_RATE}" \
  ++trainer.args.per_device_train_batch_size="${BATCH_SIZE}" \
  ++trainer.args.gradient_accumulation_steps="${GRAD_ACC}" \
  ++trainer.args.max_steps="${MAX_STEPS}" \
  ++trainer.args.logging_steps=1 \
  ++trainer.args.report_to=none \
  ++trainer.args.eval_on_start=false \
  ++trainer.args.do_eval=false \
  ++trainer.args.eval_strategy=no \
  ++trainer.args.save_strategy=no \
  ++trainer.args.remove_unused_columns=false \
  ++trainer.args.ddp_find_unused_parameters=false \
  2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "[DONE] Momentum smoke test completed."
echo "[LOG]  ${LOG_FILE}"
echo "[OUT]  ${OUTPUT_DIR}"
echo "============================================================"

find "${OUTPUT_DIR}" \
  -maxdepth 2 \
  -type f \
  \( -name "*summary*.json" -o -name "*sample_stop*.json" \) \
  -print
