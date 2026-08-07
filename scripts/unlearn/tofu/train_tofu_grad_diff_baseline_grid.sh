#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate unlearning

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Keep these training settings aligned with
# train_tofu_sample_early_stop_grad_diff_irreversible.sh.  GradDiff has no
# early-stopping threshold, so this baseline produces one run per LR/batch/epoch
# grid point rather than repeating the three threshold values.
MODEL_CONFIG="Llama-3.2-1B-Instruct"
MODEL_TAG="Llama-3.2-1B-Instruct"
TRAINER="GradDiff"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-3.2-1B-Instruct_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

splits=("forget01 holdout01 retain99")
lr_set=("9e-6")

# per-device batch size=4, gradient accumulation=2
bz_set=("4 2")
epoch_set=("10")

test -f "${PRETRAINED_PATH}/config.json" || {
    echo "Missing local model: ${PRETRAINED_PATH}"
    exit 1
}

for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"

    retain_logs="${EVAL_ROOT}/tofu_${MODEL_TAG}_${retain_split}/TOFU_EVAL.json"
    test -f "${retain_logs}" || {
        echo "Missing retain evaluation file: ${retain_logs}"
        exit 1
    }

    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            read -r bsz grad_acc <<< "${bz}"

            for epochs in "${epoch_set[@]}"; do
                suffix="lr${lr}_b${bsz}_ga${grad_acc}_e${epochs}"
                task_name="unlearn_tofu_${MODEL_TAG}_${forget_split}_${TRAINER}_${suffix}"
                output_dir="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL_TAG}/${TRAINER}/${suffix}"
                eval_output_dir="${output_dir}/eval"

                if [[ -f "${eval_output_dir}/TOFU_SUMMARY.json" ]]; then
                    echo "Completed; skipping: ${output_dir}"
                    continue
                fi

                echo
                echo "============================================================"
                echo "GradDiff baseline training"
                echo "Model:       ${MODEL_TAG}"
                echo "Split:       ${forget_split}"
                echo "LR:          ${lr}"
                echo "Batch size:  ${bsz}"
                echo "Grad accum:  ${grad_acc}"
                echo "Epochs:      ${epochs}"
                echo "Output:      ${output_dir}"
                echo "============================================================"

                rm -rf "${output_dir}"
                mkdir -p "${output_dir}"

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
                    task_name="${task_name}" \
                    paths.output_dir="${output_dir}" \
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
                    trainer.method_args.alpha=1.0 \
                    trainer.method_args.gamma=1.0 \
                    trainer.method_args.retain_loss_type=NLL

                test -f "${output_dir}/config.json" || {
                    echo "Training did not save config.json: ${output_dir}/config.json"
                    exit 1
                }

                rm -rf "${eval_output_dir}"
                mkdir -p "${eval_output_dir}"

                python -u src/eval.py --config-name=eval.yaml \
                    experiment=eval/tofu/default \
                    model="${MODEL_CONFIG}" \
                    model.model_args.pretrained_model_name_or_path="${output_dir}" \
                    +model.model_args.local_files_only=true \
                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                    +model.tokenizer_args.local_files_only=true \
                    forget_split="${forget_split}" \
                    holdout_split="${holdout_split}" \
                    retain_logs_path="${retain_logs}" \
                    task_name="${task_name}_final_eval" \
                    paths.output_dir="${eval_output_dir}"
            done
        done
    done
done
