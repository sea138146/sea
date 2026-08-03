#!/usr/bin/env bash
set -euo pipefail

ROOT="./saves/unlearn/tofu/forget05/Llama-2-7b-chat-hf/SimNPO"

if [[ ! -d "${ROOT}" ]]; then
  echo "[ERROR] SimNPO forget05 directory not found:"
  echo "        ${ROOT}"
  exit 1
fi

run_count=0
cleaned_count=0
skipped_count=0

# 只枚举 SimNPO 根目录下的参数组合目录，
# 不会把 checkpoint 或 evals 误当成参数组合。
while IFS= read -r -d '' run_dir; do
  run_count=$((run_count + 1))

  echo
  echo "============================================================"
  echo "[SIMNPO RUN] ${run_dir}"
  echo "============================================================"

  checkpoint_count=0
  valid_checkpoint_count=0
  declare -a checkpoint_dirs=()

  # 第一阶段：检查当前参数组合的全部 checkpoint
  while IFS= read -r -d '' checkpoint_dir; do
    checkpoint_count=$((checkpoint_count + 1))
    checkpoint_dirs+=("${checkpoint_dir}")

    eval_dir="${checkpoint_dir}/evals"

    if [[ ! -d "${eval_dir}" ]]; then
      echo "[INVALID] Missing evals/: ${checkpoint_dir}"
      continue
    fi

    eval_file_count=$(find "${eval_dir}" -type f | wc -l)

    if [[ "${eval_file_count}" -eq 0 ]]; then
      echo "[INVALID] Empty evals/: ${eval_dir}"
      continue
    fi

    valid_checkpoint_count=$((valid_checkpoint_count + 1))

    echo "[VALID] $(basename "${checkpoint_dir}") eval_files=${eval_file_count}"

  done < <(
    find "${run_dir}" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name 'checkpoint-*' \
      -print0 |
    sort -zV
  )

  echo "[CHECKPOINTS] total=${checkpoint_count}, valid=${valid_checkpoint_count}"

  if [[ "${checkpoint_count}" -eq 0 ]]; then
    echo "[SKIP] No checkpoint directories."
    skipped_count=$((skipped_count + 1))
    unset checkpoint_dirs
    continue
  fi

  if [[ "${valid_checkpoint_count}" -ne "${checkpoint_count}" ]]; then
    echo "[SAFE SKIP] Some checkpoints are missing eval results."
    echo "[SAFE SKIP] No files were deleted from this parameter run."
    skipped_count=$((skipped_count + 1))
    unset checkpoint_dirs
    continue
  fi

  # 第二阶段：所有 checkpoint 检查通过后才执行删除
  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    find "${checkpoint_dir}" \
      -mindepth 1 \
      -maxdepth 1 \
      ! -name 'evals' \
      -exec rm -rf -- {} +

    echo "[CLEANED] ${checkpoint_dir}"
  done

  # 参数组合根目录只保留 checkpoint-*
  find "${run_dir}" \
    -mindepth 1 \
    -maxdepth 1 \
    ! -name 'checkpoint-*' \
    -exec rm -rf -- {} +

  # 第三阶段：最终验证
  validation_errors=0

  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    entry_count=$(
      find "${checkpoint_dir}" \
        -mindepth 1 \
        -maxdepth 1 |
      wc -l
    )

    if [[ "${entry_count}" -ne 1 || ! -d "${checkpoint_dir}/evals" ]]; then
      echo "[VERIFY ERROR] ${checkpoint_dir}"

      find "${checkpoint_dir}" \
        -mindepth 1 \
        -maxdepth 1 \
        -printf '  %f\n'

      validation_errors=$((validation_errors + 1))
    fi
  done

  if [[ "${validation_errors}" -ne 0 ]]; then
    echo "[ERROR] Validation failed for:"
    echo "        ${run_dir}"
    exit 1
  fi

  cleaned_count=$((cleaned_count + 1))
  echo "[RUN CLEANED] ${run_dir}"

  unset checkpoint_dirs

done < <(
  find "${ROOT}" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -print0 |
  sort -zV
)

echo
echo "============================================================"
echo "[ALL DONE]"
echo "SimNPO forget05 runs found: ${run_count}"
echo "Cleaned:                   ${cleaned_count}"
echo "Safely skipped:            ${skipped_count}"
echo "============================================================"
