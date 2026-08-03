#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

cd /unlearning/experment_data/open-unlearning

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="Llama-2-7b-chat-hf"
TRAINER="NPO"
RESULT_GROUP="NPO_MatchedGrid"

WORKER="./run_npo_grid_worker.sh"

SPLITS=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

EFFECTIVE_BATCHES=(
    8
    16
)

LRS=(
    "2e-5"
    "4e-5"
    "5e-5"
    "6e-5"
)

MICRO_BATCH=4
EPOCHS=10
SEED=0
BETA=0.1

LOG_ROOT="./logs/npo_grid24"

BASE_EVAL="./saves/eval_checkpoints/tofu"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${WORKER}" ]]; then
    echo "[ERROR] Worker does not exist or is not executable:"
    echo "        ${WORKER}"
    exit 1
fi

# 启动前检查三个 retain 日志
for spec in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${spec}"

    retain_log="/model/evals/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    if [[ ! -s "${retain_log}" ]]; then
        echo "[ERROR] Missing retain log:"
        echo "        ${retain_log}"
        exit 1
    fi
done

total=$((
    ${#SPLITS[@]}
    * ${#EFFECTIVE_BATCHES[@]}
    * ${#LRS[@]}
))

current=0

for spec in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${spec}"

    for effective_batch in "${EFFECTIVE_BATCHES[@]}"; do
        if (( effective_batch % MICRO_BATCH != 0 )); then
            echo "[ERROR] Effective batch ${effective_batch} cannot be divided by micro batch ${MICRO_BATCH}"
            exit 1
        fi

        grad_acc=$((effective_batch / MICRO_BATCH))

        for lr in "${LRS[@]}"; do
            current=$((current + 1))

            suffix="lr${lr}_eb${effective_batch}"
            suffix+="_b${MICRO_BATCH}_ga${grad_acc}"
            suffix+="_e${EPOCHS}_s${SEED}_beta${BETA}"

            result_dir="${BASE_EVAL}/${forget_split}"
            result_dir+="/${MODEL}/${RESULT_GROUP}/${suffix}"

            log_file="${LOG_ROOT}/${forget_split}"
            log_file+="_eb${effective_batch}_lr${lr}.log"

            echo
            echo "============================================================"
            echo "[${current}/${total}] ${TRAINER} matched grid"
            echo "split           = ${forget_split}"
            echo "holdout         = ${holdout_split}"
            echo "retain          = ${retain_split}"
            echo "effective batch = ${effective_batch}"
            echo "micro batch     = ${MICRO_BATCH}"
            echo "gradient acc    = ${grad_acc}"
            echo "learning rate   = ${lr}"
            echo "epochs          = ${EPOCHS}"
            echo "seed            = ${SEED}"
            echo "beta            = ${BETA}"
            echo "result          = ${result_dir}"
            echo "============================================================"

            if [[ -f "${result_dir}/NPO_DONE" ]]; then
                echo "[SKIP] Completed run found."
                continue
            fi

            set -o pipefail

            GRID_SPLIT="${spec}" \
            GRID_LR="${lr}" \
            GRID_BSZ="${MICRO_BATCH}" \
            GRID_GA="${grad_acc}" \
            bash "${WORKER}" 2>&1 \
                | tee "${log_file}"

            if [[ ! -f "${result_dir}/NPO_DONE" ]]; then
                echo "[ERROR] Completion marker was not generated:"
                echo "        ${result_dir}"
                exit 1
            fi

            echo "[OK] Completed ${current}/${total}"
        done
    done
done

echo
echo "============================================================"
echo "[DONE] All 24 NPO matched-grid experiments completed"
echo "============================================================"
