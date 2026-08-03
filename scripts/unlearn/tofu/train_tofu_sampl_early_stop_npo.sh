#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL="Llama-3.2-1B-Instruct"
TRAINER="SampleEarlyStopNPO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

splits=(
    "forget01 holdout01 retain99"
#    "forget05 holdout05 retain95"
#    "forget10 holdout10 retain90"
)

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-3.2-1B-Instruct_full"

CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"

threshold_set=(
    "1e-2"
    "3e-2"
    "5e-2"
)

for threshold in "${threshold_set[@]}"; do
moving_average_rate="2"
k="1"

lr_set=(
    "2e-5"
)

bz_set=(
    "4 2"
)

epoch_set=(
    10
)

mkdir -p "${CHECKPOINT_ROOT}"
mkdir -p "${EVAL_ROOT}"

if [[ ! -e "${PRETRAINED_PATH}" ]]; then
    echo "错误：未找到预训练模型：${PRETRAINED_PATH}"
    exit 1
fi

for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"

    retain_eval_file="${EVAL_ROOT}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    if [[ ! -f "${retain_eval_file}" ]]; then
        echo "错误：未找到 retain 评估文件：${retain_eval_file}"
        exit 1
    fi

    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            read -r bsz grad_acc <<< "${bz}"

            for epochs in "${epoch_set[@]}"; do
                SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_e${epochs}_ies_tokmean_tau${threshold}_ma${moving_average_rate}_k${k}"

                TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"

                OUTPUT_DIR="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

                mkdir -p "${OUTPUT_DIR}"

                echo
                echo "============================================================"
                echo "开始训练：${TASK_NAME}"
                echo "model=${MODEL}"
                echo "pretrained_path=${PRETRAINED_PATH}"
                echo "forget_split=${forget_split}"
                echo "retain_split=${retain_split}"
                echo "learning_rate=${lr}"
                echo "batch_size=${bsz}"
                echo "gradient_accumulation_steps=${grad_acc}"
                echo "effective_batch_size=$((bsz * grad_acc))"
                echo "epochs=${epochs}"
                echo "threshold=${threshold}"
                echo "moving_average_rate=${moving_average_rate}"
                echo "k=${k}"
                echo "output_dir=${OUTPUT_DIR}"
                echo "============================================================"

                python src/train.py --config-name=unlearn.yaml \
                    experiment=unlearn/tofu/default \
                    trainer="${TRAINER}" \
                    collator=DataCollatorForSupervisedDatasetwithIndex \
                    model="${MODEL}" \
                    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
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
                    trainer.args.logging_steps=1 \
                    trainer.args.learning_rate="${lr}" \
                    trainer.args.per_device_train_batch_size="${bsz}" \
                    trainer.args.gradient_accumulation_steps="${grad_acc}" \
                    trainer.args.num_train_epochs="${epochs}" \
                    trainer.args.dataloader_drop_last=false \
                    trainer.args.eval_strategy=no \
                    trainer.args.eval_on_start=false \
                    trainer.method_args.threshold="${threshold}" \
                    trainer.method_args.moving_average_rate="${moving_average_rate}" \
                    trainer.method_args.k="${k}"

                # ============================================================
                # Evaluate the final locally saved model after training.
                # ============================================================
                if [[ ! -f "${OUTPUT_DIR}/config.json" ]]; then
                    echo "错误：训练结束后没有找到最终模型：${OUTPUT_DIR}/config.json"
                    exit 1
                fi

                EVAL_TASK_NAME="${TASK_NAME}_final_eval"
                EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

                # 防止同名旧评估结果导致 evaluator 因 overwrite=false 而跳过。
                rm -rf "${EVAL_OUTPUT_DIR}"
                mkdir -p "${EVAL_OUTPUT_DIR}"

                echo
                echo "============================================================"
                echo "开始评估最终模型"
                echo "eval_model_path=${OUTPUT_DIR}"
                echo "tokenizer_path=${PRETRAINED_PATH}"
                echo "forget_split=${forget_split}"
                echo "holdout_split=${holdout_split}"
                echo "retain_logs_path=${retain_eval_file}"
                echo "eval_output_dir=${EVAL_OUTPUT_DIR}"
                echo "============================================================"

                python -u src/eval.py --config-name=eval.yaml \
                    experiment=eval/tofu/default \
                    model="${MODEL}" \
                    model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    forget_split="${forget_split}" \
                    holdout_split="${holdout_split}" \
                    retain_logs_path="${retain_eval_file}" \
                    task_name="${EVAL_TASK_NAME}" \
                    paths.output_dir="${EVAL_OUTPUT_DIR}"

                echo
                echo "============================================================"
                echo "训练和评估全部完成"
                echo "model_dir=${OUTPUT_DIR}"
                echo "eval_dir=${EVAL_OUTPUT_DIR}"
                echo "============================================================"

                find "${EVAL_OUTPUT_DIR}" -maxdepth 2 -type f \
                    \( -name "TOFU_EVAL.json" -o -name "TOFU_SUMMARY.json" \) \
                    -print
            done
        done
    done
done

done
