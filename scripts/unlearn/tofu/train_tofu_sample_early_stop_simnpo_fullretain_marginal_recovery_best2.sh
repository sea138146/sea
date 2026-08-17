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

MODEL_CONFIG="Llama-2-7b-chat-hf"
MODEL_TAG="Llama-2-7b-chat-hf"
TRAINER="SampleEarlyStopSimNPOMarginalRatio"
VARIANT_SUFFIX="marginalrecovery_fullretain_fixed_forget_denom_best2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

splits=(
 #   "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"
INITIAL_NLL_CACHE_ROOT="${CHECKPOINT_ROOT}/tofu/initial_nll/${MODEL_TAG}"

# Controller grid. The first experiment uses the proposed defaults.
moving_average_window_set=("2")
stop_ratio_threshold_set=("0.1" "0.15" "0.20")
rebound_ratio_threshold_set=("0.1")
ratio_epsilon="1e-8"

lr_set=("2e-5" "5e-5")
bz_set=("4 2" "8 2")
beta_set=("3.5")
delta_set=("1")
gamma_set=("0.25")
epoch_set=("10")

mkdir -p "${CHECKPOINT_ROOT}" "${EVAL_ROOT}" "${INITIAL_NLL_CACHE_ROOT}"

if [[ ! -e "${PRETRAINED_PATH}" ]]; then
    echo "错误：未找到预训练模型：${PRETRAINED_PATH}"
    exit 1
fi

for moving_average_window in "${moving_average_window_set[@]}"; do
for stop_ratio_threshold in "${stop_ratio_threshold_set[@]}"; do
for rebound_ratio_threshold in "${rebound_ratio_threshold_set[@]}"; do
for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"
    initial_nll_cache_file="${INITIAL_NLL_CACHE_ROOT}/${forget_split}.json"
    retain_eval_file="${EVAL_ROOT}/tofu_${MODEL_TAG}_${retain_split}/TOFU_EVAL.json"

    if [[ ! -f "${retain_eval_file}" ]]; then
        echo "错误：未找到 retain 评估文件：${retain_eval_file}"
        exit 1
    fi

    for lr in "${lr_set[@]}"; do
    for bz in "${bz_set[@]}"; do
        read -r bsz grad_acc <<< "${bz}"
        case "${forget_split}:${lr}:${bsz}:${grad_acc}:${stop_ratio_threshold}" in
            "forget05:5e-5:8:2:0.15"|"forget10:2e-5:4:2:0.1") ;;
            *) continue ;;
        esac
        for beta in "${beta_set[@]}"; do
        for delta in "${delta_set[@]}"; do
        for gamma in "${gamma_set[@]}"; do
        for epochs in "${epoch_set[@]}"; do
            SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_beta${beta}_delta${delta}_gamma${gamma}_e${epochs}_marginalratio_ma${moving_average_window}_stop${stop_ratio_threshold}_rebound${rebound_ratio_threshold}_${VARIANT_SUFFIX}"
            TASK_NAME="unlearn_tofu_${MODEL_TAG}_${forget_split}_${TRAINER}_${SUFFIX}"
            OUTPUT_DIR="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL_TAG}/${TRAINER}/${SUFFIX}"
            EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

            if [[ -f "${EVAL_OUTPUT_DIR}/TOFU_SUMMARY.json" ]]; then
                echo "跳过已完成实验：${TASK_NAME}"
                continue
            fi

            mkdir -p "${OUTPUT_DIR}"

            echo
            echo "============================================================"
            echo "开始 SimNPO marginal-recovery FullRetain best2 训练：${TASK_NAME}"
            echo "forget_split=${forget_split} retain_split=${retain_split}"
            echo "learning_rate=${lr} batch_size=${bsz} grad_acc=${grad_acc}"
            echo "epochs=${epochs} moving_average_window=${moving_average_window}"
            echo "beta=${beta} delta=${delta} gamma=${gamma}"
            echo "stop_ratio_threshold=${stop_ratio_threshold}"
            echo "rebound_ratio_threshold=${rebound_ratio_threshold}"
            echo "sampling=baseline_forget_anchor_with_stopped_forget_masked"
            echo "normalization=forget_original_batch_retain_full_baseline_mean"
            echo "output_dir=${OUTPUT_DIR}"
            echo "============================================================"

            python -u src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default \
                trainer="${TRAINER}" \
                collator=DataCollatorForSupervisedDatasetwithIndex \
                model="${MODEL_CONFIG}" \
                model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                +model.model_args.local_files_only=true \
                model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                +model.tokenizer_args.local_files_only=true \
                forget_split="${forget_split}" \
                holdout_split="${holdout_split}" \
                retain_split="${retain_split}" \
                task_name="${TASK_NAME}" \
                paths.output_dir="${OUTPUT_DIR}" \
                eval.tofu.retain_logs_path="${retain_eval_file}" \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.report_to=none \
                trainer.args.do_train=true \
                trainer.args.do_eval=false \
                trainer.args.logging_steps=10 \
                +trainer.args.disable_tqdm=true \
                trainer.args.save_strategy=no \
                trainer.args.learning_rate="${lr}" \
                trainer.args.per_device_train_batch_size="${bsz}" \
                trainer.args.gradient_accumulation_steps="${grad_acc}" \
                trainer.args.dataloader_drop_last=false \
                trainer.args.num_train_epochs="${epochs}" \
                trainer.args.eval_strategy=no \
                trainer.args.eval_on_start=false \
                trainer.method_args.moving_average_window="${moving_average_window}" \
                trainer.method_args.stop_ratio_threshold="${stop_ratio_threshold}" \
                trainer.method_args.rebound_ratio_threshold="${rebound_ratio_threshold}" \
                trainer.method_args.ratio_epsilon="${ratio_epsilon}" \
                trainer.method_args.initial_nll_cache_path="${initial_nll_cache_file}" \
                trainer.method_args.beta="${beta}" \
                trainer.method_args.delta="${delta}" \
                trainer.method_args.gamma="${gamma}" \
                trainer.method_args.alpha=1.0 \
                trainer.method_args.retain_loss_type=NLL \
                trainer.method_args.log_per_sample_normalized_nll=false

            if [[ ! -f "${OUTPUT_DIR}/config.json" ]]; then
                echo "错误：训练结束后没有找到最终模型：${OUTPUT_DIR}/config.json"
                exit 1
            fi

            EVAL_TASK_NAME="${TASK_NAME}_final_eval"
            rm -rf "${EVAL_OUTPUT_DIR}"
            mkdir -p "${EVAL_OUTPUT_DIR}"

            python -u src/eval.py --config-name=eval.yaml \
                experiment=eval/tofu/default \
                model="${MODEL_CONFIG}" \
                model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
                +model.model_args.local_files_only=true \
                model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                +model.tokenizer_args.local_files_only=true \
                forget_split="${forget_split}" \
                holdout_split="${holdout_split}" \
                retain_logs_path="${retain_eval_file}" \
                task_name="${EVAL_TASK_NAME}" \
                paths.output_dir="${EVAL_OUTPUT_DIR}"

            echo "完成：${TASK_NAME}"
            find "${EVAL_OUTPUT_DIR}" -maxdepth 2 -type f \
                \( -name "TOFU_EVAL.json" -o -name "TOFU_SUMMARY.json" \) \
                -print
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
