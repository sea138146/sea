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

MODEL="Llama-2-7b-chat-hf"
TRAINER="NPO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"

splits=(
#    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
#    "forget10 holdout10 retain90"
)

lr_set=(
    "2e-5"
    "5e-5"
)

bz_set=(
    "4 2"
)

epoch_set=(
    "10"
)

test -f "${PRETRAINED_PATH}/config.json" || {
    echo "找不到本地模型：${PRETRAINED_PATH}"
    exit 1
}

for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"

    RETAIN_LOGS="${EVAL_ROOT}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    test -f "${RETAIN_LOGS}" || {
        echo "找不到 retain 评估文件：${RETAIN_LOGS}"
        exit 1
    }

    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            read -r bsz grad_acc <<< "${bz}"

            for epochs in "${epoch_set[@]}"; do
                effective_batch=$((bsz * grad_acc))

                SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_e${epochs}"
                TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"

                OUTPUT_DIR="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"
                EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

                if [[ -f "${EVAL_OUTPUT_DIR}/TOFU_SUMMARY.json" ]]; then
                    echo
                    echo "已完成，跳过：${OUTPUT_DIR}"
                    continue
                fi

                rm -rf "${OUTPUT_DIR}"
                mkdir -p "${OUTPUT_DIR}"

                echo
                echo "============================================================"
                echo "标准 NPO 基线训练"
                echo "model=${MODEL}"
                echo "forget_split=${forget_split}"
                echo "holdout_split=${holdout_split}"
                echo "retain_split=${retain_split}"
                echo "learning_rate=${lr}"
                echo "batch_size=${bsz}"
                echo "gradient_accumulation_steps=${grad_acc}"
                echo "effective_batch_size=${effective_batch}"
                echo "epochs=${epochs}"
                echo "output_dir=${OUTPUT_DIR}"
                echo "============================================================"

                python -u src/train.py --config-name=unlearn.yaml \
                    experiment=unlearn/tofu/default \
                    trainer="${TRAINER}" \
                    collator=DataCollatorForSupervisedDataset \
                    model="${MODEL}" \
                    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    +model.model_args.local_files_only=true \
                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    +model.tokenizer_args.local_files_only=true \
                    forget_split="${forget_split}" \
                    holdout_split="${holdout_split}" \
                    retain_split="${retain_split}" \
                    task_name="${TASK_NAME}" \
                    paths.output_dir="${OUTPUT_DIR}" \
                    trainer.args.do_train=true \
                    trainer.args.do_eval=false \
                    trainer.args.eval_strategy=no \
                    trainer.args.eval_on_start=false \
                    trainer.args.report_to=none \
                    trainer.args.logging_steps=1 \
                    trainer.args.save_strategy=no \
                    trainer.args.gradient_checkpointing=true \
                    trainer.args.ddp_find_unused_parameters=true \
                    trainer.args.learning_rate="${lr}" \
                    trainer.args.per_device_train_batch_size="${bsz}" \
                    trainer.args.gradient_accumulation_steps="${grad_acc}" \
                    trainer.args.num_train_epochs="${epochs}" \
                    trainer.method_args.beta=0.1 \
                    trainer.method_args.alpha=1.0 \
                    trainer.method_args.gamma=1.0 \
                    trainer.method_args.retain_loss_type=NLL \
                    trainer.method_args.log_per_sample_normalized_nll=true

                test -f "${OUTPUT_DIR}/config.json" || {
                    echo "训练完成后未找到模型：${OUTPUT_DIR}/config.json"
                    exit 1
                }

                rm -rf "${EVAL_OUTPUT_DIR}"
                mkdir -p "${EVAL_OUTPUT_DIR}"

                echo
                echo "============================================================"
                echo "评估标准 NPO 最终模型"
                echo "model_path=${OUTPUT_DIR}"
                echo "eval_dir=${EVAL_OUTPUT_DIR}"
                echo "============================================================"

                python -u src/eval.py --config-name=eval.yaml \
                    experiment=eval/tofu/default \
                    model="${MODEL}" \
                    model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
                    +model.model_args.local_files_only=true \
                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    +model.tokenizer_args.local_files_only=true \
                    forget_split="${forget_split}" \
                    holdout_split="${holdout_split}" \
                    retain_logs_path="${RETAIN_LOGS}" \
                    task_name="${TASK_NAME}_final_eval" \
                    paths.output_dir="${EVAL_OUTPUT_DIR}"

                echo
                echo "完成：${forget_split}/${SUFFIX}"
                echo "评估结果：${EVAL_OUTPUT_DIR}/TOFU_SUMMARY.json"
            done
        done
    done
done

echo
echo "================ NPO 基线网格汇总 ================"

python - <<'PY'
import json
from pathlib import Path

base = Path(
    "/unlearning/experment_data/checkpoints/tofu"
)

for forget_split in ("forget01", "forget05", "forget10"):
    root = (
        base
        / forget_split
        / "Llama-2-7b-chat-hf"
        / "NPO"
    )

    if not root.exists():
        continue

    for run_dir in sorted(root.glob("lr*_b4_ga2_e10")):
        summary_path = (
            run_dir
            / "eval"
            / "TOFU_SUMMARY.json"
        )

        if not summary_path.is_file():
            continue

        data = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        print("=" * 100)
        print(f"{forget_split}/{run_dir.name}")
        print("forget_quality:", data.get("forget_quality"))
        print("model_utility:", data.get("model_utility"))
        print("forget_Q_A_Prob:", data.get("forget_Q_A_Prob"))
        print("forget_Q_A_ROUGE:", data.get("forget_Q_A_ROUGE"))
        print("privleak:", data.get("privleak"))
        print(
            "extraction_strength:",
            data.get("extraction_strength"),
        )
        print(
            "exact_memorization:",
            data.get("exact_memorization"),
        )
PY
