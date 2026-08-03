#!/usr/bin/env bash
set -euo pipefail

cd /unlearning/experment_data/open-unlearning
export PYTHONPATH=src:${PYTHONPATH:-}

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

REPORTTO="none"

MODEL="Llama-3.2-3B-Instruct"
TRAINER="SampleStopNPO"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-3.2-3B-Instruct_full"
TOFU_DATA_DIR="/data/datasets/datasets-TOFU"

forget_split="forget05"
holdout_split="holdout05"
retain_split="retain95"

lr="1e-5"
bsz="1"
grad_acc="8"
steps=20

SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_s${steps}"
TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="./outputs/abs_first_smoke_seed0"

CKPT_DIR="${OUTPUT_DIR}/checkpoint-${steps}"
EVAL_TASK_NAME="${TASK_NAME}_refeval_em"
EVAL_DIR="/unlearning/experment_data/open-unlearning-saves/eval/${EVAL_TASK_NAME}"
RETAIN_LOGS_PATH=""

test -d "${PRETRAINED_PATH}" || { echo "[ERROR] model path not found: ${PRETRAINED_PATH}"; exit 1; }
test -f "${TOFU_DATA_DIR}/${forget_split}.json" || { echo "[ERROR] missing ${TOFU_DATA_DIR}/${forget_split}.json"; exit 1; }
test -f "${TOFU_DATA_DIR}/${holdout_split}.json" || { echo "[ERROR] missing ${TOFU_DATA_DIR}/${holdout_split}.json"; exit 1; }
test -f "${TOFU_DATA_DIR}/${retain_split}.json" || { echo "[ERROR] missing ${TOFU_DATA_DIR}/${retain_split}.json"; exit 1; }

echo "============================================================"
echo "[TRAIN] ${TASK_NAME}"
echo "[OUT] ${OUTPUT_DIR}"
echo "============================================================"

python src/train.py --config-name=unlearn.yaml \
  mode=unlearn \
  experiment=unlearn/tofu/default.yaml \
  trainer="${TRAINER}" \
  model="${MODEL}" \
  task_name="${TASK_NAME}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_split="${retain_split}" \
  retain_logs_path=/model/evals/tofu_Llama-3.2-3B-Instruct_retain95/TOFU_EVAL.json
  model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.model_args.torch_dtype=float16 \
  model.model_args.attn_implementation=flash_attention_2 \
  trainer.args.output_dir="${OUTPUT_DIR}" \
  trainer.args.ddp_find_unused_parameters=false \
  trainer.args.report_to="${REPORTTO}" \
  trainer.args.logging_steps=1 \
  trainer.args.learning_rate="${lr}" \
  trainer.args.per_device_train_batch_size="${bsz}" \
  trainer.args.gradient_accumulation_steps="${grad_acc}" \
  trainer.args.use_cache=false \
  trainer.args.eval_on_start=false \
  trainer.args.do_eval=false \
  trainer.args.eval_strategy=no \
  trainer.args.save_strategy=steps \
  +trainer.args.save_steps="${steps}" \
  +trainer.args.save_total_limit=1 \
  +trainer.args.max_steps="${steps}" \
  trainer.args.remove_unused_columns=false

echo "============================================================"
echo "[EVAL] ${EVAL_TASK_NAME}"
echo "[CKPT] ${CKPT_DIR}"
echo "[RETAIN_LOGS] ${RETAIN_LOGS_PATH}"
echo "[EVAL_OUT] ${EVAL_DIR}"
echo "============================================================"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT_DIR}"
  exit 1
fi

python src/train.py --config-name=unlearn.yaml \
  mode=eval \
  experiment=eval/tofu/default.yaml \
  model="${MODEL}" \
  task_name="${EVAL_TASK_NAME}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_logs_path=/model/evals/tofu_Llama-3.2-3B-Instruct_retain95/TOFU_EVAL.json
  model.model_args.pretrained_model_name_or_path="${CKPT_DIR}" \
  model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.model_args.torch_dtype=float16 \
  model.model_args.attn_implementation=flash_attention_2 \
  trainer.args.output_dir="${EVAL_DIR}"

if [ -f "${EVAL_DIR}/TOFU_EVAL.json" ] || [ -f "${EVAL_DIR}/evals/TOFU_EVAL.json" ]; then
  echo "[OK] eval finished. Deleting checkpoint: ${CKPT_DIR}"
  rm -rf "${CKPT_DIR}"
else
  echo "[ERROR] eval result not found. Keep checkpoint: ${CKPT_DIR}"
  exit 1
fi

echo "[DONE] train -> eval -> delete finished."
