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
TRAINER="SampleEarlyStopWGAIrreversible"
TOKENIZER_PATH="/model/Llama/Llama-2-7b-chat-hf"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints/muse"
INITIAL_NLL_CACHE_ROOT="${CHECKPOINT_ROOT}/initial_nll/${MODEL_TAG}/WGA"

# split, learning rate, and three cumulative normalized-NLL-gain thresholds.
# WGA moves much faster than NPO on Books, so warm_up=1 is required to avoid
# waiting until the model is already over-unlearned.
splits=(
    "News 2e-5 0.2 0.5 1.0"
    "Books 1e-5 0.2 0.5 1.0"
)

bz_set=("4 8")
alpha_set=("1")
beta_set=("1.0")
warm_up_set=("1")
patience_set=("1")
rebound_delta_set=("0.1")
reactivation_patience_set=("2")
epoch_set=("10")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${INITIAL_NLL_CACHE_ROOT}"

test -f "${TOKENIZER_PATH}/tokenizer.json" || {
    echo "Missing local Llama-2 tokenizer: ${TOKENIZER_PATH}"
    exit 1
}

for SPLIT_CONFIG in "${splits[@]}"; do
    read -r DATA_SPLIT SPLIT_LR THRESHOLD_A THRESHOLD_B THRESHOLD_C <<< "${SPLIT_CONFIG}"
    DATA_SPLIT_LOWER="${DATA_SPLIT,,}"
    TARGET_MODEL="/model/finetune_models/MUSE-${DATA_SPLIT_LOWER}_target"
    RETAIN_LOGS="/model/evals/muse_Llama-2-7b-hf_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
    DATA_FILE="/data/datasets/MUSE/MUSE-${DATA_SPLIT}/raw/forget-00000-of-00001.parquet"
    INITIAL_NLL_CACHE="${INITIAL_NLL_CACHE_ROOT}/${DATA_SPLIT}.json"

    case "${DATA_SPLIT}" in
        News) NUM_FORGET_CHUNKS=407 ;;
        Books) NUM_FORGET_CHUNKS=553 ;;
        *) echo "Unknown MUSE split: ${DATA_SPLIT}"; exit 1 ;;
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

for WARM_UP in "${warm_up_set[@]}"; do
for PATIENCE in "${patience_set[@]}"; do
for GAIN_THRESHOLD in "${THRESHOLD_A}" "${THRESHOLD_B}" "${THRESHOLD_C}"; do
for REBOUND_DELTA in "${rebound_delta_set[@]}"; do
for REACTIVATION_PATIENCE in "${reactivation_patience_set[@]}"; do
for LR in "${SPLIT_LR}"; do
for BZ in "${bz_set[@]}"; do
    read -r BATCH_SIZE GRAD_ACC <<< "${BZ}"
for ALPHA in "${alpha_set[@]}"; do
for BETA in "${beta_set[@]}"; do
for EPOCHS in "${epoch_set[@]}"; do
    FORGET_BATCHES_PER_EPOCH=$(((NUM_FORGET_CHUNKS + BATCH_SIZE - 1) / BATCH_SIZE))
    OPTIMIZER_STEPS_PER_EPOCH=$(((FORGET_BATCHES_PER_EPOCH + GRAD_ACC - 1) / GRAD_ACC))
    MAX_STEPS=$((OPTIMIZER_STEPS_PER_EPOCH * EPOCHS))

SUFFIX="lr${LR}_b${BATCH_SIZE}_ga${GRAD_ACC}_beta${BETA}_alpha${ALPHA}_e${EPOCHS}_ies_wga_normnll_gain_ge${GAIN_THRESHOLD}_warm${WARM_UP}_patience${PATIENCE}_rebounddelta${REBOUND_DELTA}_rpat${REACTIVATION_PATIENCE}_samplingbaseline_masked"
TASK_NAME="muse_${MODEL_TAG}_${DATA_SPLIT}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${DATA_SPLIT}/${MODEL_TAG}/${TRAINER}/${SUFFIX}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"
LOG_FILE="${OUTPUT_DIR}/${TRAINER}.log"

if [[ -f "${EVAL_OUTPUT_DIR}/MUSE_SUMMARY.json" ]]; then
    echo "Completed; skipping: ${OUTPUT_DIR}"
    continue
fi
mkdir -p "${OUTPUT_DIR}"

{
    echo "============================================================"
    echo "MUSE ${DATA_SPLIT} WGA sample early stop with rebound"
    echo "target_model=${TARGET_MODEL}"
    echo "tokenizer=${TOKENIZER_PATH}"
    echo "lr=${LR} batch=${BATCH_SIZE} grad_acc=${GRAD_ACC} beta=${BETA}"
    echo "optimizer_steps_per_epoch=${OPTIMIZER_STEPS_PER_EPOCH} max_steps=${MAX_STEPS}"
    echo "effective_batch_size=$((BATCH_SIZE * GRAD_ACC))"
    echo "epochs=${EPOCHS} alpha=${ALPHA}"
    echo "warm_up=${WARM_UP} patience=${PATIENCE} gain_threshold=${GAIN_THRESHOLD}"
    echo "rebound_delta=${REBOUND_DELTA} rebound_threshold=${GAIN_THRESHOLD}-${REBOUND_DELTA}"
    echo "reactivation_patience=${REACTIVATION_PATIENCE}"
    echo "initial_nll_cache=${INITIAL_NLL_CACHE}"
    echo "sampling=baseline_forget_anchor_with_masked_forget_loss"
    echo "stopped_forget_gradient=zero retain_batch=full forget_denominator=original_valid_tokens"
    echo "output_dir=${OUTPUT_DIR}"
    echo "training_eval=disabled final_eval=enabled"
    echo "============================================================"
} | tee -a "${LOG_FILE}"

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
    trainer.args.dataloader_drop_last=false \
    +trainer.args.dataloader_num_workers=0 \
    trainer.args.num_train_epochs="${EPOCHS}" \
    trainer.method_args.alpha="${ALPHA}" \
    trainer.method_args.beta="${BETA}" \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.retain_loss_type=NLL \
    trainer.method_args.warm_up="${WARM_UP}" \
    trainer.method_args.gain_threshold="${GAIN_THRESHOLD}" \
    trainer.method_args.patience="${PATIENCE}" \
    trainer.method_args.rebound_delta="${REBOUND_DELTA}" \
    trainer.method_args.reactivation_patience="${REACTIVATION_PATIENCE}" \
    trainer.method_args.initial_nll_cache_path="${INITIAL_NLL_CACHE}" \
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

done
done
done
done
done
done
done
done
done
done
done
