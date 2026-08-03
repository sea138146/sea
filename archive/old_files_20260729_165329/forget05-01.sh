#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate unlearning

cd /unlearning/experment_data/open-unlearning

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="BalDRO"

REPORTTO="none"

MODEL="Llama-2-7b-chat-hf"
TRAINER="NPO"

# ============================================================
# 数据划分
# ============================================================
FORGET_SPLIT="forget05"
HOLDOUT_SPLIT="holdout05"
RETAIN_SPLIT="retain95"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
TOKENIZER_PATH="/model/Llama/Llama-2-7b-chat-hf"

# ============================================================
# 超参数网格
# ============================================================
lr_set=(
  "2e-5"
  "4e-5"
  "5e-5"
  "6e-5"
  "7e-5"
)

# 格式：
#   micro_batch_size gradient_accumulation_steps
#
# 有效 batch size：
#   8 × 2 = 16
#   8 × 4 = 32
bz_set=(
  "8 2"
  "8 4"
)

epoch_set=(
  "10"
)

# ============================================================
# 输出目录
# ============================================================
OUTPUT_ROOT="./saves/unlearn/tofu/${FORGET_SPLIT}/${MODEL}/forget05-01"
LOG_DIR="./logs/forget05-01"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_DIR}"

# ============================================================
# 工具函数：检查 checkpoint 中的两个评估 JSON
# ============================================================
find_eval_jsons() {
  local checkpoint_dir="$1"

  local tofu_eval_file
  local tofu_summary_file

  tofu_eval_file=$(
    find "${checkpoint_dir}" \
      -type f \
      -name "TOFU_EVAL.json" \
      -print \
      -quit
  )

  tofu_summary_file=$(
    find "${checkpoint_dir}" \
      -type f \
      -name "TOFU_SUMMARY.json" \
      -print \
      -quit
  )

  printf '%s\n%s\n' "${tofu_eval_file}" "${tofu_summary_file}"
}

# ============================================================
# 工具函数：清理一个 checkpoint，只保留两个 JSON
# ============================================================
clean_checkpoint() {
  local checkpoint_dir="$1"
  local tofu_eval_file="$2"
  local tofu_summary_file="$3"

  local temp_dir
  local remaining_count

  temp_dir=$(mktemp -d)

  # 临时保存两个 JSON
  cp -f \
    "${tofu_eval_file}" \
    "${temp_dir}/TOFU_EVAL.json"

  cp -f \
    "${tofu_summary_file}" \
    "${temp_dir}/TOFU_SUMMARY.json"

  # 清空 checkpoint 内所有内容
  find "${checkpoint_dir}" \
    -mindepth 1 \
    -maxdepth 1 \
    -exec rm -rf -- {} +

  # 把两个 JSON 放回 checkpoint 根目录
  cp -f \
    "${temp_dir}/TOFU_EVAL.json" \
    "${checkpoint_dir}/TOFU_EVAL.json"

  cp -f \
    "${temp_dir}/TOFU_SUMMARY.json" \
    "${checkpoint_dir}/TOFU_SUMMARY.json"

  rm -rf "${temp_dir}"

  # 最终检查
  if [[ ! -f "${checkpoint_dir}/TOFU_EVAL.json" ]]; then
    echo "[ERROR] TOFU_EVAL.json was not preserved:"
    echo "        ${checkpoint_dir}"
    return 1
  fi

  if [[ ! -f "${checkpoint_dir}/TOFU_SUMMARY.json" ]]; then
    echo "[ERROR] TOFU_SUMMARY.json was not preserved:"
    echo "        ${checkpoint_dir}"
    return 1
  fi

  remaining_count=$(
    find "${checkpoint_dir}" \
      -mindepth 1 \
      -maxdepth 1 \
      | wc -l
  )

  if [[ "${remaining_count}" -ne 2 ]]; then
    echo "[ERROR] Checkpoint does not contain exactly two files:"
    echo "        ${checkpoint_dir}"
    echo "[FOUND] ${remaining_count} entries"

    find "${checkpoint_dir}" \
      -mindepth 1 \
      -maxdepth 1 \
      -printf '        %f\n'

    return 1
  fi

  echo "[CLEANED CHECKPOINT] ${checkpoint_dir}"
  echo "  kept: TOFU_EVAL.json"
  echo "  kept: TOFU_SUMMARY.json"
}

# ============================================================
# 主循环
# ============================================================
for lr in "${lr_set[@]}"; do
  for bz in "${bz_set[@]}"; do
    for epochs in "${epoch_set[@]}"; do

      bsz=$(echo "${bz}" | cut -d' ' -f1)
      grad_acc=$(echo "${bz}" | cut -d' ' -f2)
      eff_bsz=$((bsz * grad_acc))

      SUFFIX="lr${lr}_b${eff_bsz}_mb${bsz}_ga${grad_acc}_e${epochs}"

      TASK_NAME="unlearn_tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_${SUFFIX}"

      OUTPUT_DIR="${OUTPUT_ROOT}/${SUFFIX}"

      RETAIN_LOGS_PATH="/model/evals/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

      LOG_FILE="${LOG_DIR}/${FORGET_SPLIT}_${SUFFIX}.log"

      echo "============================================================"
      echo "[FORGET]       ${FORGET_SPLIT}"
      echo "[HOLDOUT]      ${HOLDOUT_SPLIT}"
      echo "[RETAIN]       ${RETAIN_SPLIT}"
      echo "[LR]           ${lr}"
      echo "[MICRO_BSZ]    ${bsz}"
      echo "[GRAD_ACC]     ${grad_acc}"
      echo "[EFF_BSZ]      ${eff_bsz}"
      echo "[EPOCHS]       ${epochs}"
      echo "[TASK]         ${TASK_NAME}"
      echo "[OUTPUT]       ${OUTPUT_DIR}"
      echo "[LOG]          ${LOG_FILE}"
      echo "[RETAIN_LOG]   ${RETAIN_LOGS_PATH}"
      echo "============================================================"

      # --------------------------------------------------------
      # 防止覆盖已有结果
      # --------------------------------------------------------
      if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "[ERROR] Output directory already exists:"
        echo "        ${OUTPUT_DIR}"
        echo
        echo "如需重跑当前参数组合，请先手动删除该目录。"
        exit 1
      fi

      mkdir -p "${OUTPUT_DIR}"

      # --------------------------------------------------------
      # 训练与逐 epoch 评估
      #
      # save_strategy=epoch：
      #   每个 epoch 保存一个 checkpoint
      #
      # eval_strategy=epoch：
      #   每个 epoch 结束后进行一次评估
      #
      # 注意：
      #   训练期间不删除任何 checkpoint。
      #   当前超参数组合全部完成后才统一清理。
      # --------------------------------------------------------
      python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="${TRAINER}" \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
        model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        task_name="${TASK_NAME}" \
        paths.output_dir="${OUTPUT_DIR}" \
        eval.tofu.retain_logs_path="${RETAIN_LOGS_PATH}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.report_to="${REPORTTO}" \
        trainer.args.logging_steps=1 \
        trainer.args.learning_rate="${lr}" \
        trainer.args.per_device_train_batch_size="${bsz}" \
        trainer.args.gradient_accumulation_steps="${grad_acc}" \
        trainer.args.num_train_epochs="${epochs}" \
        trainer.args.eval_strategy=epoch \
        trainer.args.eval_on_start=False \
        trainer.args.save_strategy=epoch \
        2>&1 | tee "${LOG_FILE}"

      echo
      echo "[TRAIN+EVAL DONE] ${FORGET_SPLIT} ${SUFFIX}"
      echo

      # ========================================================
      # 第一阶段：
      # 检查每个 checkpoint 是否同时存在两个评估 JSON
      #
      # 只要有一个 checkpoint 不完整，就停止清理。
      # 此时所有模型文件仍然保留，避免误删。
      # ========================================================
      checkpoint_count=0
      valid_checkpoint_count=0

      declare -a checkpoint_dirs=()
      declare -a tofu_eval_files=()
      declare -a tofu_summary_files=()

      while IFS= read -r -d '' checkpoint_dir; do
        checkpoint_count=$((checkpoint_count + 1))

        mapfile -t eval_paths < <(
          find_eval_jsons "${checkpoint_dir}"
        )

        tofu_eval_file="${eval_paths[0]:-}"
        tofu_summary_file="${eval_paths[1]:-}"

        checkpoint_dirs+=("${checkpoint_dir}")
        tofu_eval_files+=("${tofu_eval_file}")
        tofu_summary_files+=("${tofu_summary_file}")

        echo "------------------------------------------------------------"
        echo "[CHECKPOINT] ${checkpoint_dir}"

        if [[ -n "${tofu_eval_file}" ]]; then
          echo "[FOUND] TOFU_EVAL.json"
          echo "        ${tofu_eval_file}"
        else
          echo "[MISSING] TOFU_EVAL.json"
        fi

        if [[ -n "${tofu_summary_file}" ]]; then
          echo "[FOUND] TOFU_SUMMARY.json"
          echo "        ${tofu_summary_file}"
        else
          echo "[MISSING] TOFU_SUMMARY.json"
        fi

        if [[ -n "${tofu_eval_file}" && -n "${tofu_summary_file}" ]]; then
          valid_checkpoint_count=$((valid_checkpoint_count + 1))
        fi

      done < <(
        find "${OUTPUT_DIR}" \
          -type d \
          -name "checkpoint-*" \
          -print0 \
          | sort -zV
      )

      echo "============================================================"
      echo "[CHECKPOINT COUNT]       ${checkpoint_count}"
      echo "[VALID CHECKPOINT COUNT] ${valid_checkpoint_count}"
      echo "============================================================"

      if [[ "${checkpoint_count}" -eq 0 ]]; then
        echo "[ERROR] No checkpoint directories were found:"
        echo "        ${OUTPUT_DIR}"
        echo
        echo "[SAFE EXIT] No model files were deleted."
        exit 1
      fi

      if [[ "${valid_checkpoint_count}" -ne "${checkpoint_count}" ]]; then
        echo "[ERROR] Some checkpoints do not contain both JSON files."
        echo
        echo "[SAFE EXIT] No checkpoint files were deleted."
        echo "[SAFE EXIT] Please inspect:"
        echo "            ${OUTPUT_DIR}"
        exit 1
      fi

      # ========================================================
      # 第二阶段：
      # 每个 checkpoint 只保留两个 JSON
      # ========================================================
      for index in "${!checkpoint_dirs[@]}"; do
        clean_checkpoint \
          "${checkpoint_dirs[$index]}" \
          "${tofu_eval_files[$index]}" \
          "${tofu_summary_files[$index]}"
      done

      # ========================================================
      # 第三阶段：
      # 清理当前超参数组合根目录
      #
      # 只保留 checkpoint-* 目录。
      # 删除根目录下的最终模型、配置、tokenizer、状态文件等。
      # ========================================================
      find "${OUTPUT_DIR}" \
        -mindepth 1 \
        -maxdepth 1 \
        ! -name "checkpoint-*" \
        -exec rm -rf -- {} +

      # ========================================================
      # 第四阶段：
      # 最终严格检查
      # ========================================================
      final_checkpoint_count=0
      final_error_count=0

      while IFS= read -r -d '' checkpoint_dir; do
        final_checkpoint_count=$((final_checkpoint_count + 1))

        entry_count=$(
          find "${checkpoint_dir}" \
            -mindepth 1 \
            -maxdepth 1 \
            | wc -l
        )

        if [[ "${entry_count}" -ne 2 ]]; then
          echo "[FINAL CHECK ERROR] ${checkpoint_dir}"
          echo "Expected 2 entries, found ${entry_count}"

          find "${checkpoint_dir}" \
            -mindepth 1 \
            -maxdepth 1 \
            -printf '        %f\n'

          final_error_count=$((final_error_count + 1))
          continue
        fi

        if [[ ! -f "${checkpoint_dir}/TOFU_EVAL.json" ]]; then
          echo "[FINAL CHECK ERROR] Missing TOFU_EVAL.json:"
          echo "                    ${checkpoint_dir}"
          final_error_count=$((final_error_count + 1))
        fi

        if [[ ! -f "${checkpoint_dir}/TOFU_SUMMARY.json" ]]; then
          echo "[FINAL CHECK ERROR] Missing TOFU_SUMMARY.json:"
          echo "                    ${checkpoint_dir}"
          final_error_count=$((final_error_count + 1))
        fi

      done < <(
        find "${OUTPUT_DIR}" \
          -mindepth 1 \
          -maxdepth 1 \
          -type d \
          -name "checkpoint-*" \
          -print0 \
          | sort -zV
      )

      if [[ "${final_error_count}" -ne 0 ]]; then
        echo "[ERROR] Final checkpoint validation failed."
        exit 1
      fi

      echo
      echo "============================================================"
      echo "[RUN COMPLETE] ${SUFFIX}"
      echo "[CHECKPOINTS]  ${final_checkpoint_count}"
      echo "[FINAL STATE]  Each checkpoint contains only:"
      echo "               TOFU_EVAL.json"
      echo "               TOFU_SUMMARY.json"
      echo "[OUTPUT]       ${OUTPUT_DIR}"
      echo "============================================================"
      echo

      # 清理数组，避免下一组继承
      unset checkpoint_dirs
      unset tofu_eval_files
      unset tofu_summary_files
      unset eval_paths

    done
  done
done

echo "============================================================"
echo "[ALL DONE] NPO forget05 grid completed"
echo "[OUTPUT ROOT] ${OUTPUT_ROOT}"
echo "[LOG ROOT]    ${LOG_DIR}"
echo "[FINAL STATE] Every checkpoint keeps only two evaluation JSON files"
echo "============================================================"
