#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

cd /unlearning/experment_data/open-unlearning

: "${GRID_SPLIT:?GRID_SPLIT is required}"
: "${GRID_LR:?GRID_LR is required}"
: "${GRID_BSZ:?GRID_BSZ is required}"
: "${GRID_GA:?GRID_GA is required}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

MODEL="Llama-2-7b-chat-hf"

# 实际训练算法仍为标准 NPO
TRAINER="NPO"

# 独立保存目录，不覆盖已有 NPO、SatImp、SimNPO 等结果
RESULT_GROUP="NPO_MatchedGrid"

REPORTTO="none"

read -r FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT <<< "${GRID_SPLIT}"

LR="${GRID_LR}"
BSZ="${GRID_BSZ}"
GA="${GRID_GA}"

EPOCHS=10
SEED=0

BETA=0.1
ALPHA=1.0
GAMMA=1.0

EFFECTIVE_BATCH=$((BSZ * GA))

PRETRAINED_PATH="/model/finetune_models/tofu_${MODEL}_full"

RETAIN_LOGS="/model/evals/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

SUFFIX="lr${LR}_eb${EFFECTIVE_BATCH}"
SUFFIX+="_b${BSZ}_ga${GA}"
SUFFIX+="_e${EPOCHS}_s${SEED}"
SUFFIX+="_beta${BETA}"

TASK_NAME="unlearn_tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_${SUFFIX}"

OUTPUT_DIR="./saves/unlearn/tofu/${FORGET_SPLIT}"
OUTPUT_DIR+="/${MODEL}/${RESULT_GROUP}/${SUFFIX}"

EVAL_ROOT="./saves/eval_checkpoints/tofu/${FORGET_SPLIT}"
EVAL_ROOT+="/${MODEL}/${RESULT_GROUP}/${SUFFIX}"

test -d "${PRETRAINED_PATH}" || {
    echo "[ERROR] Model not found:"
    echo "        ${PRETRAINED_PATH}"
    exit 1
}

test -s "${RETAIN_LOGS}" || {
    echo "[ERROR] Retain log not found or empty:"
    echo "        ${RETAIN_LOGS}"
    exit 1
}

test -f "configs/trainer/NPO.yaml" || {
    echo "[ERROR] NPO configuration not found:"
    echo "        configs/trainer/NPO.yaml"
    exit 1
}

if [[ -f "${EVAL_ROOT}/NPO_DONE" ]]; then
    echo "[SKIP] Already completed:"
    echo "       ${EVAL_ROOT}"
    exit 0
fi

# 只清理 NPO_MatchedGrid 下对应的未完成实验。
# 不会修改原来的 .../${MODEL}/NPO/ 目录。
if [[ -d "${OUTPUT_DIR}" || -d "${EVAL_ROOT}" ]]; then
    echo "[CLEAN] Removing incomplete matched-grid run:"
    echo "        ${SUFFIX}"

    rm -rf "${OUTPUT_DIR}" "${EVAL_ROOT}"
fi

mkdir -p "${EVAL_ROOT}"

echo "============================================================"
echo "Trainer        : ${TRAINER}"
echo "Result group   : ${RESULT_GROUP}"
echo "Model          : ${MODEL}"
echo "Forget split   : ${FORGET_SPLIT}"
echo "Holdout split  : ${HOLDOUT_SPLIT}"
echo "Retain split   : ${RETAIN_SPLIT}"
echo "Learning rate  : ${LR}"
echo "Batch size     : ${BSZ}"
echo "Grad acc       : ${GA}"
echo "Effective BS   : ${EFFECTIVE_BATCH}"
echo "Epochs         : ${EPOCHS}"
echo "Seed           : ${SEED}"
echo "Beta           : ${BETA}"
echo "Alpha          : ${ALPHA}"
echo "Gamma          : ${GAMMA}"
echo "Retain loss    : NLL"
echo "Train output   : ${OUTPUT_DIR}"
echo "Eval output    : ${EVAL_ROOT}"
echo "============================================================"

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer="${TRAINER}" \
    trainer.method_args.beta="${BETA}" \
    trainer.method_args.alpha="${ALPHA}" \
    trainer.method_args.gamma="${GAMMA}" \
    trainer.method_args.retain_loss_type=NLL \
    model="${MODEL}" \
    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    model.model_args.torch_dtype=bfloat16 \
    model.model_args.attn_implementation=flash_attention_2 \
    ++model.model_args.use_cache=false \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    task_name="${TASK_NAME}" \
    paths.output_dir="${OUTPUT_DIR}" \
    ++do_save=true \
    eval.tofu.retain_logs_path="${RETAIN_LOGS}" \
    ++trainer.args.seed="${SEED}" \
    ++trainer.args.learning_rate="${LR}" \
    ++trainer.args.per_device_train_batch_size="${BSZ}" \
    ++trainer.args.gradient_accumulation_steps="${GA}" \
    ++trainer.args.num_train_epochs="${EPOCHS}" \
    ++trainer.args.gradient_checkpointing=true \
    ++trainer.args.bf16=true \
    ++trainer.args.fp16=false \
    ++trainer.args.ddp_find_unused_parameters=false \
    ++trainer.args.remove_unused_columns=false \
    ++trainer.args.report_to="${REPORTTO}" \
    ++trainer.args.run_name="${TASK_NAME}" \
    ++trainer.args.logging_steps=1 \
    ++trainer.args.eval_strategy=no \
    ++trainer.args.eval_on_start=false \
    ++trainer.args.do_eval=false \
    ++trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=1 \
    ++trainer.args.save_only_model=true

CHECKPOINT=$(
    find "${OUTPUT_DIR}" \
        -maxdepth 1 \
        -mindepth 1 \
        -type d \
        -name "checkpoint-*" \
        | sort -V \
        | tail -1
)

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] No checkpoint found:"
    echo "        ${OUTPUT_DIR}"
    exit 1
fi

STEP="${CHECKPOINT##*-}"

EVAL_DIR="${EVAL_ROOT}/checkpoint-${STEP}"

EVAL_TASK="${TASK_NAME}_checkpoint_${STEP}_eval"

rm -rf "${EVAL_DIR}"
mkdir -p "${EVAL_DIR}"

echo "============================================================"
echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Evaluation output    : ${EVAL_DIR}"
echo "============================================================"

python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model="${MODEL}" \
    model.model_args.pretrained_model_name_or_path="${CHECKPOINT}" \
    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_logs_path="${RETAIN_LOGS}" \
    task_name="${EVAL_TASK}" \
    paths.output_dir="${EVAL_DIR}" \
    eval.tofu.overwrite=true

EVAL_JSON=$(
    find "${EVAL_DIR}" \
        -type f \
        -name "TOFU_EVAL.json" \
        -size +0c \
        -print \
        -quit
)

SUMMARY_JSON=$(
    find "${EVAL_DIR}" \
        -type f \
        -name "TOFU_SUMMARY.json" \
        -size +0c \
        -print \
        -quit
)

if [[ -z "${EVAL_JSON}" ]]; then
    echo "[ERROR] TOFU_EVAL.json was not generated."
    echo "[INFO] Keeping checkpoint:"
    echo "       ${CHECKPOINT}"
    exit 1
fi

if [[ -z "${SUMMARY_JSON}" ]]; then
    echo "[ERROR] TOFU_SUMMARY.json was not generated."
    echo "[INFO] Keeping checkpoint:"
    echo "       ${CHECKPOINT}"
    exit 1
fi

# 暂存两个最终评估文件
TEMP_DIR=$(mktemp -d)

cleanup_temp() {
    rm -rf "${TEMP_DIR}"
}

trap cleanup_temp EXIT

cp "${EVAL_JSON}" \
    "${TEMP_DIR}/TOFU_EVAL.json"

cp "${SUMMARY_JSON}" \
    "${TEMP_DIR}/TOFU_SUMMARY.json"

# 删除评估过程产生的其他文件，仅保留最终两个 JSON
rm -rf "${EVAL_DIR}"
mkdir -p "${EVAL_DIR}"

mv "${TEMP_DIR}/TOFU_EVAL.json" \
    "${EVAL_DIR}/TOFU_EVAL.json"

mv "${TEMP_DIR}/TOFU_SUMMARY.json" \
    "${EVAL_DIR}/TOFU_SUMMARY.json"

rmdir "${TEMP_DIR}"

trap - EXIT

test -s "${EVAL_DIR}/TOFU_EVAL.json" || {
    echo "[ERROR] Final TOFU_EVAL.json verification failed."
    exit 1
}

test -s "${EVAL_DIR}/TOFU_SUMMARY.json" || {
    echo "[ERROR] Final TOFU_SUMMARY.json verification failed."
    exit 1
}

# 评估成功后，删除训练 checkpoint，减少磁盘占用
rm -rf "${OUTPUT_DIR}"

# 保存真实运行配置
cat > "${EVAL_ROOT}/RUN_CONFIG.txt" <<CONFIG_EOF
trainer=${TRAINER}
result_group=${RESULT_GROUP}
model=${MODEL}
forget_split=${FORGET_SPLIT}
holdout_split=${HOLDOUT_SPLIT}
retain_split=${RETAIN_SPLIT}
learning_rate=${LR}
per_device_train_batch_size=${BSZ}
gradient_accumulation_steps=${GA}
effective_batch_size=${EFFECTIVE_BATCH}
epochs=${EPOCHS}
seed=${SEED}
beta=${BETA}
alpha=${ALPHA}
gamma=${GAMMA}
retain_loss_type=NLL
checkpoint_step=${STEP}
CONFIG_EOF

touch "${EVAL_ROOT}/NPO_DONE"

echo "============================================================"
echo "[DONE] NPO matched-grid experiment completed"
echo "Result          : ${EVAL_ROOT}"
echo "Checkpoint step : ${STEP}"
echo "============================================================"
