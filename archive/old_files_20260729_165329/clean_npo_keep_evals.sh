#!/usr/bin/env bash
set -euo pipefail

ROOT="./saves/unlearn/tofu"
MODEL="Llama-2-7b-chat-hf"
TRAINER="NPO"

run_count=0
cleaned_run_count=0
skipped_run_count=0

while IFS= read -r -d '' run_dir; do
  run_count=$((run_count + 1))

  echo
  echo "============================================================"
  echo "[NPO RUN] ${run_dir}"
  echo "============================================================"

  checkpoint_count=0
  valid_checkpoint_count=0
  declare -a checkpoint_dirs=()

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
    skipped_run_count=$((skipped_run_count + 1))
    unset checkpoint_dirs
    continue
  fi

  if [[ "${valid_checkpoint_count}" -ne "${checkpoint_count}" ]]; then
    echo "[SAFE SKIP] Some checkpoints do not have valid evals/."
    echo "[SAFE SKIP] No files were deleted from this run."
    skipped_run_count=$((skipped_run_count + 1))
    unset checkpoint_dirs
    continue
  fi

  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    find "${checkpoint_dir}" \
      -mindepth 1 \
      -maxdepth 1 \
      ! -name 'evals' \
      -exec rm -rf -- {} +

    echo "[CLEANED] ${checkpoint_dir}"
  done

  find "${run_dir}" \
    -mindepth 1 \
    -maxdepth 1 \
    ! -name 'checkpoint-*' \
    -exec rm -rf -- {} +

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
    echo "[ERROR] Validation failed for ${run_dir}"
    exit 1
  fi

  cleaned_run_count=$((cleaned_run_count + 1))
  echo "[RUN CLEANED] ${run_dir}"

  unset checkpoint_dirs

done < <(
  find "${ROOT}" \
    -type d \
    -path "*/${MODEL}/${TRAINER}/*" \
    -mindepth 1 \
    -maxdepth 20 \
    -print0 |
  sort -zV
)

echo
echo "============================================================"
echo "[ALL DONE]"
echo "NPO parameter runs found: ${run_count}"
echo "Cleaned:                  ${cleaned_run_count}"
echo "Safely skipped:           ${skipped_run_count}"
echo "============================================================"
