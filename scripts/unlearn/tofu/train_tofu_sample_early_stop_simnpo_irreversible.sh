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
TRAINER="SampleEarlyStopSimNPOIrreversible"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"
INITIAL_NLL_CACHE_ROOT="${CHECKPOINT_ROOT}/tofu/initial_nll/${MODEL_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

splits=(
    # "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)
gain_threshold_set=("2.5" "3.0")
early_stop_set=(
    "2 1 0.2 2" # warm_up patience rebound_delta reactivation_patience
)
lr_set=("2e-5" "5e-5")
bz_set=("8 2" "8 4")
beta_set=("3.5")
delta_set=("1")
gamma_set=("0.125" "0.25")
epoch_set=("10")

mkdir -p "${INITIAL_NLL_CACHE_ROOT}"

test -f "${PRETRAINED_PATH}/config.json" || {
    echo "Missing local model: ${PRETRAINED_PATH}"
    exit 1
}

for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"
    initial_nll_cache_file="${INITIAL_NLL_CACHE_ROOT}/${forget_split}.json"
    retain_logs="${EVAL_ROOT}/tofu_${MODEL_TAG}_${retain_split}/TOFU_EVAL.json"

    test -f "${retain_logs}" || {
        echo "Missing retain evaluation file: ${retain_logs}"
        exit 1
    }

    for early_stop_config in "${early_stop_set[@]}"; do
        read -r warm_up patience rebound_delta reactivation_patience <<< "${early_stop_config}"
        for gain_threshold in "${gain_threshold_set[@]}"; do
        for lr in "${lr_set[@]}"; do
            for bz in "${bz_set[@]}"; do
                read -r bsz grad_acc <<< "${bz}"

                for beta in "${beta_set[@]}"; do
                    for delta in "${delta_set[@]}"; do
                        for gamma in "${gamma_set[@]}"; do
                            for epochs in "${epoch_set[@]}"; do
                    suffix="lr${lr}_b${bsz}_ga${grad_acc}_beta${beta}_delta${delta}_gamma${gamma}_e${epochs}_ies_simnpo_normnll_gain_ge${gain_threshold}_warm${warm_up}_patience${patience}_rebounddelta${rebound_delta}_rpat${reactivation_patience}"
                    task_name="unlearn_tofu_${MODEL_TAG}_${forget_split}_${TRAINER}_${suffix}"
                    output_dir="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL_TAG}/${TRAINER}/${suffix}"
                    eval_output_dir="${output_dir}/eval"

                    if [[ -f "${eval_output_dir}/TOFU_SUMMARY.json" ]]; then
                        echo "Completed; skipping: ${output_dir}"
                        continue
                    fi

                    echo
                    echo "============================================================"
                    echo "SIMNPO reversible sample early stopping"
                    echo "Model:       ${MODEL_TAG}"
                    echo "Split:       ${forget_split}"
                    echo "Gain thres:  ${gain_threshold}"
                    echo "Warm up:     ${warm_up}"
                    echo "Patience:    ${patience}"
                    echo "Rebound:     ${rebound_delta}"
                    echo "React pat:   ${reactivation_patience}"
                    echo "LR:          ${lr}"
                    echo "Batch size:  ${bsz}"
                    echo "Grad accum:  ${grad_acc}"
                    echo "Beta:        ${beta}"
                    echo "Delta:       ${delta}"
                    echo "Gamma:       ${gamma}"
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
                        trainer.method_args.warm_up="${warm_up}" \
                        trainer.method_args.gain_threshold="${gain_threshold}" \
                        trainer.method_args.patience="${patience}" \
                        trainer.method_args.rebound_delta="${rebound_delta}" \
                        trainer.method_args.reactivation_patience="${reactivation_patience}" \
                        trainer.method_args.initial_nll_cache_path="${initial_nll_cache_file}" \
                        trainer.method_args.delta="${delta}" \
                        trainer.method_args.beta="${beta}" \
                        trainer.method_args.alpha=1.0 \
                        trainer.method_args.gamma="${gamma}" \
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
            done
        done
        done
    done
done
