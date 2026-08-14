#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

MODEL_CONFIG="Llama-2-7b-hf"
MODEL_TAG="Llama-2-7b-hf"
TRAINER="NPO"
TOKENIZER_PATH="/model/Llama/Llama-2-7b-chat-hf"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints/muse"

splits=(
    "News"
    "Books"
)

lr_set=("1e-5" "2e-5" "5e-5")
bz_set=("4 8") # batch_size, gradient_accumulation_steps
alpha_set=("1") # standard NPO retain weight
epoch_set=("10")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

test -f "${TOKENIZER_PATH}/tokenizer.json" || {
    echo "Missing local Llama-2 tokenizer: ${TOKENIZER_PATH}"
    exit 1
}

for DATA_SPLIT in "${splits[@]}"; do
    DATA_SPLIT_LOWER="${DATA_SPLIT,,}"
    TARGET_MODEL="/model/finetune_models/MUSE-${DATA_SPLIT_LOWER}_target"
    RETAIN_LOGS="/model/evals/muse_Llama-2-7b-hf_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
    DATA_FILE="/data/datasets/MUSE/MUSE-${DATA_SPLIT}/raw/forget-00000-of-00001.parquet"

    case "${DATA_SPLIT}" in
        News) NUM_FORGET_CHUNKS=407 ;;
        Books) NUM_FORGET_CHUNKS=553 ;;
        *)
            echo "Unknown MUSE split: ${DATA_SPLIT}"
            exit 1
            ;;
    esac
    test -f "${TARGET_MODEL}/config.json" || {
        echo "Missing target model: ${TARGET_MODEL}"
        exit 1
    }
    test -f "${RETAIN_LOGS}" || {
        echo "Missing MUSE retrain evaluation: ${RETAIN_LOGS}"
        exit 1
    }
    test -f "${DATA_FILE}" || {
        echo "Missing local MUSE dataset: ${DATA_FILE}"
        exit 1
    }

for LR in "${lr_set[@]}"; do
for BZ in "${bz_set[@]}"; do
    read -r BATCH_SIZE GRAD_ACC <<< "${BZ}"
for ALPHA in "${alpha_set[@]}"; do
for EPOCHS in "${epoch_set[@]}"; do
    FORGET_BATCHES_PER_EPOCH=$(((NUM_FORGET_CHUNKS + BATCH_SIZE - 1) / BATCH_SIZE))
    OPTIMIZER_STEPS_PER_EPOCH=$(((FORGET_BATCHES_PER_EPOCH + GRAD_ACC - 1) / GRAD_ACC))
    MAX_STEPS=$((OPTIMIZER_STEPS_PER_EPOCH * EPOCHS))

SUFFIX="lr${LR}_b${BATCH_SIZE}_ga${GRAD_ACC}_alpha${ALPHA}_e${EPOCHS}_nlltrace"
TASK_NAME="muse_${MODEL_TAG}_${DATA_SPLIT}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${DATA_SPLIT}/${MODEL_TAG}/${TRAINER}/${SUFFIX}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"
LOG_FILE="${OUTPUT_DIR}/${TRAINER}.log"

if [[ -f "${EVAL_OUTPUT_DIR}/MUSE_SUMMARY.json" ]]; then
    echo "Completed; skipping: ${OUTPUT_DIR}"
    continue
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "MUSE ${DATA_SPLIT} NPO baseline" | tee -a "${LOG_FILE}"
echo "target_model=${TARGET_MODEL}" | tee -a "${LOG_FILE}"
echo "tokenizer=${TOKENIZER_PATH}" | tee -a "${LOG_FILE}"
echo "lr=${LR} batch=${BATCH_SIZE} grad_acc=${GRAD_ACC}" | tee -a "${LOG_FILE}"
echo "optimizer_steps_per_epoch=${OPTIMIZER_STEPS_PER_EPOCH} max_steps=${MAX_STEPS}" | tee -a "${LOG_FILE}"
echo "effective_batch_size=$((BATCH_SIZE * GRAD_ACC))" | tee -a "${LOG_FILE}"
echo "epochs=${EPOCHS} alpha=${ALPHA}" | tee -a "${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "training_eval=disabled final_eval=enabled" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# Training does not run the expensive MUSE evaluator at every epoch. The NPO
# trainer still writes one forward-only normalized-NLL snapshot for every
# forget chunk at initialization and after each epoch.
python -u src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default \
    trainer="${TRAINER}" \
    collator=DataCollatorForSupervisedDatasetwithIndex \
    model="${MODEL_CONFIG}" \
    data_split="${DATA_SPLIT}" \
    task_name="${TASK_NAME}" \
    paths.output_dir="${OUTPUT_DIR}" \
    retain_logs_path="${RETAIN_LOGS}" \
    model.model_args.pretrained_model_name_or_path="${TARGET_MODEL}" \
    +model.model_args.local_files_only=true \
    model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
    +model.tokenizer_args.local_files_only=true \
    trainer.args.do_train=true \
    trainer.args.do_eval=false \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=false \
    trainer.args.report_to=none \
    trainer.args.logging_steps=1 \
    trainer.args.save_strategy=no \
    trainer.args.gradient_checkpointing=true \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.learning_rate="${LR}" \
    trainer.args.per_device_train_batch_size="${BATCH_SIZE}" \
    trainer.args.gradient_accumulation_steps="${GRAD_ACC}" \
    trainer.args.num_train_epochs="${EPOCHS}" \
    trainer.method_args.alpha="${ALPHA}" \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.retain_loss_type=NLL \
    trainer.method_args.log_per_sample_normalized_nll=true \
    2>&1 | tee -a "${LOG_FILE}"

test -f "${OUTPUT_DIR}/config.json" || {
    echo "Training finished without a saved model: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
    exit 1
}

mkdir -p "${EVAL_OUTPUT_DIR}"
echo "Starting final MUSE evaluation" | tee -a "${LOG_FILE}"

python -u src/eval.py --config-name=eval.yaml \
    experiment=eval/muse/default \
    model="${MODEL_CONFIG}" \
    data_split="${DATA_SPLIT}" \
    task_name="${TASK_NAME}_final_eval" \
    paths.output_dir="${EVAL_OUTPUT_DIR}" \
    retain_logs_path="${RETAIN_LOGS}" \
    model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
    +model.model_args.local_files_only=true \
    model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
    +model.tokenizer_args.local_files_only=true \
    eval.muse.overwrite=true \
    2>&1 | tee -a "${LOG_FILE}"

test -f "${EVAL_OUTPUT_DIR}/MUSE_SUMMARY.json" || {
    echo "Final evaluation did not produce MUSE_SUMMARY.json" | tee -a "${LOG_FILE}"
    exit 1
}

echo "Completed: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Summary: ${EVAL_OUTPUT_DIR}/MUSE_SUMMARY.json" | tee -a "${LOG_FILE}"

done
done
done
done
done
