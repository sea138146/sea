#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/unlearning/experment_data/open-unlearning"
SAVE_ROOT="/unlearning/experment_data/open-unlearning-saves"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

MODEL="Llama-3.2-1B-Instruct"
TRAINER="NPO"
SEED="0"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-3.2-1B-Instruct_full"
TOKENIZER_PATH="/model/Llama/Llama-3.2-1B-Instruct"
TOFU_DATA_DIR="/data/datasets/datasets-TOFU"

FORGET_SPLIT="forget05"
HOLDOUT_SPLIT="holdout05"
RETAIN_SPLIT="retain95"

LR="1e-5"
BETA="0.5"
ALPHA="1.0"
EPOCHS="10"

TRAIN_BSZ="1"
GRAD_ACC="8"

# forget05 共200条样本：
# ceil(200 / (1 * 8)) = 25 optimizer steps/epoch
STEPS_PER_EPOCH="25"

PROBE_BSZ="${PROBE_BSZ:-8}"
PROBE_DTYPE="${PROBE_DTYPE:-float16}"
PROBE_ATTN="${PROBE_ATTN:-flash_attention_2}"

SUFFIX="lr1e-5_beta0.5_alpha1_b1_ga8_epoch10_track_seed0"

TASK_NAME="unlearn_tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_${SUFFIX}"

OUTPUT_DIR="${SAVE_ROOT}/unlearn/tofu/${FORGET_SPLIT}/${MODEL}/${TRAINER}/${SUFFIX}"

LOG_FILE="${REPO_ROOT}/train_1b_forget05_NPO_track_seed0.log"

test -d "${PRETRAINED_PATH}" || {
    echo "[ERROR] 不存在参考模型: ${PRETRAINED_PATH}"
    exit 1
}

test -d "${TOKENIZER_PATH}" || {
    echo "[ERROR] 不存在 tokenizer: ${TOKENIZER_PATH}"
    exit 1
}

test -f "${TOFU_DATA_DIR}/${FORGET_SPLIT}.json" || {
    echo "[ERROR] 缺少 ${FORGET_SPLIT}.json"
    exit 1
}

test -f "${TOFU_DATA_DIR}/${HOLDOUT_SPLIT}.json" || {
    echo "[ERROR] 缺少 ${HOLDOUT_SPLIT}.json"
    exit 1
}

test -f "${TOFU_DATA_DIR}/${RETAIN_SPLIT}.json" || {
    echo "[ERROR] 缺少 ${RETAIN_SPLIT}.json"
    exit 1
}

test -f scripts/eval_npo_sample_trajectories.py || {
    echo "[ERROR] 缺少 scripts/eval_npo_sample_trajectories.py"
    exit 1
}

if [[ -e "${OUTPUT_DIR}" && "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo "[ERROR] 输出目录已存在："
    echo "${OUTPUT_DIR}"
    echo
    echo "为避免覆盖旧实验，请先检查目录。"
    echo "确认要删除后执行："
    echo "rm -rf '${OUTPUT_DIR}'"
    exit 1
fi

echo "============================================================"
echo "[PHASE 1] NPO逐样本遗忘轨迹"
echo "[MODEL]  ${MODEL}"
echo "[DATA]   ${FORGET_SPLIT}"
echo "[SEED]   ${SEED}"
echo "[OUTPUT] ${OUTPUT_DIR}"
echo "============================================================"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    mkdir -p "${OUTPUT_DIR}"

    TRAIN_START="$(date +%s)"

    python src/train.py --config-name=unlearn.yaml \
        mode=unlearn \
        experiment=unlearn/tofu/default.yaml \
        trainer="${TRAINER}" \
        model="${MODEL}" \
        task_name="${TASK_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        retain_logs_path=/model/evals/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json \
        model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
        model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
        model.model_args.torch_dtype=float16 \
        model.model_args.attn_implementation=flash_attention_2 \
        trainer.args.output_dir="${OUTPUT_DIR}" \
        trainer.args.ddp_find_unused_parameters=false \
        trainer.args.report_to=none \
        trainer.args.logging_steps=1 \
        trainer.args.learning_rate="${LR}" \
        trainer.args.per_device_train_batch_size="${TRAIN_BSZ}" \
        trainer.args.gradient_accumulation_steps="${GRAD_ACC}" \
        trainer.args.eval_on_start=true \
        trainer.args.do_eval=true \
        trainer.args.eval_strategy=epoch \
        trainer.args.save_strategy=epoch \
        ++trainer.args.save_only_model=true \
        ++trainer.args.save_total_limit=11 \
        ++trainer.args.save_safetensors=true \
        ++trainer.args.seed="${SEED}" \
        trainer.args.num_train_epochs="${EPOCHS}" \
        trainer.method_args.beta="${BETA}" \
        trainer.method_args.alpha="${ALPHA}" \
        trainer.args.remove_unused_columns=false \
        2>&1 | tee "${LOG_FILE}"

    TRAIN_END="$(date +%s)"
    TRAIN_SECONDS="$((TRAIN_END - TRAIN_START))"

    cat > "${OUTPUT_DIR}/run_timing.json" <<JSON
{
  "seed": ${SEED},
  "training_wall_seconds": ${TRAIN_SECONDS},
  "training_log": "${LOG_FILE}"
}
JSON
else
    echo "[INFO] SKIP_TRAIN=1，跳过训练，直接读取已有checkpoint。"
fi

echo
echo "[CHECK] 检查checkpoint模型权重"

mapfile -t CHECKPOINTS < <(
    find "${OUTPUT_DIR}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name "checkpoint-*" \
        | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "[ERROR] 未找到checkpoint目录：${OUTPUT_DIR}"
    exit 1
fi

VALID_CHECKPOINTS=()

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    STEP="${CHECKPOINT##*-}"

    if find "${CHECKPOINT}" \
        -maxdepth 1 \
        -type f \
        \( \
            -name "model*.safetensors" \
            -o -name "pytorch_model*.bin" \
        \) \
        | grep -q .; then

        VALID_CHECKPOINTS+=("${CHECKPOINT}")
    elif [[ "${STEP}" == "0" ]]; then
        echo "[INFO] 跳过checkpoint-0；初始权重使用PRETRAINED_PATH。"
    else
        echo "[WARN] 跳过无模型权重的checkpoint：${CHECKPOINT}"
    fi
done

if [[ "${#VALID_CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "[ERROR] 没有找到任何带模型权重的训练checkpoint。"
    exit 1
fi

echo "[CHECK] 有效模型checkpoint数量：${#VALID_CHECKPOINTS[@]}"
printf '%s\n' "${VALID_CHECKPOINTS[@]}"

echo "[CHECK] 找到 ${#CHECKPOINTS[@]} 个有效checkpoint："

printf '%s\n' "${CHECKPOINTS[@]}"

echo
echo "============================================================"
echo "[PROBE] 开始提取逐样本轨迹"
echo "============================================================"

python scripts/eval_npo_sample_trajectories.py \
    --repo-root "${REPO_ROOT}" \
    --checkpoint-root "${OUTPUT_DIR}" \
    --reference-model "${PRETRAINED_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --model-name "${MODEL}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --holdout-split "${HOLDOUT_SPLIT}" \
    --beta "${BETA}" \
    --seed "${SEED}" \
    --batch-size "${PROBE_BSZ}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --dtype "${PROBE_DTYPE}" \
    --attn-implementation "${PROBE_ATTN}" \
    --overwrite

echo
echo "============================================================"
echo "[DONE] 第一阶段轨迹提取完成"
echo
echo "${OUTPUT_DIR}/trajectories/"
echo "├── forget_samples_raw.jsonl"
echo "├── forget_samples_trajectory.csv"
echo "├── global_metrics.jsonl"
echo "├── reference_logps.jsonl"
echo "├── sample_metadata.jsonl"
echo "└── validation_report.json"
echo "============================================================"
