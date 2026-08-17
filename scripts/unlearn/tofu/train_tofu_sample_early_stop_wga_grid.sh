#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false

MODEL_CONFIG="Llama-2-7b-chat-hf"
MODEL_TAG="Llama-2-7b-chat-hf"
TRAINER="SampleEarlyStopWGAMarginalRatio"
PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
EVAL_ROOT="/model/evals"
INITIAL_NLL_CACHE_ROOT="${CHECKPOINT_ROOT}/tofu/initial_nll/${MODEL_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

splits=("forget05 holdout05 retain95" "forget10 holdout10 retain90")
moving_average_window_set=("2")
stop_ratio_threshold_set=("0.1" "0.2")
rebound_ratio_threshold_set=("0.1")
ratio_epsilon="1e-8"
lr_set=("2e-5")
bz_set=("8 2")
beta_set=("1.0")
epoch_set=("10")

mkdir -p "${CHECKPOINT_ROOT}" "${EVAL_ROOT}" "${INITIAL_NLL_CACHE_ROOT}"
test -f "${PRETRAINED_PATH}/config.json" || { echo "Missing model: ${PRETRAINED_PATH}"; exit 1; }

for ma in "${moving_average_window_set[@]}"; do
for stop in "${stop_ratio_threshold_set[@]}"; do
for rebound in "${rebound_ratio_threshold_set[@]}"; do
for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"
    cache="${INITIAL_NLL_CACHE_ROOT}/${forget_split}.json"
    retain_logs="${EVAL_ROOT}/tofu_${MODEL_TAG}_${retain_split}/TOFU_EVAL.json"
    test -f "${retain_logs}" || { echo "Missing retain eval: ${retain_logs}"; exit 1; }
    for lr in "${lr_set[@]}"; do
    for bz in "${bz_set[@]}"; do
    read -r bsz grad_acc <<< "${bz}"
    for beta in "${beta_set[@]}"; do
    for epochs in "${epoch_set[@]}"; do
        suffix="lr${lr}_b${bsz}_ga${grad_acc}_beta${beta}_e${epochs}_marginalratio_ma${ma}_stop${stop}_rebound${rebound}_fullretain_fixed_forget_denom"
        task="unlearn_tofu_${MODEL_TAG}_${forget_split}_${TRAINER}_${suffix}"
        out="${CHECKPOINT_ROOT}/tofu/${forget_split}/${MODEL_TAG}/${TRAINER}/${suffix}"
        eval_out="${out}/eval"
        if [[ -f "${eval_out}/TOFU_SUMMARY.json" ]]; then echo "Completed; skipping: ${out}"; continue; fi
        mkdir -p "${out}"
        echo "START ${task}"
        python -u src/train.py --config-name=unlearn.yaml \
          experiment=unlearn/tofu/default trainer="${TRAINER}" \
          collator=DataCollatorForSupervisedDatasetwithIndex model="${MODEL_CONFIG}" \
          model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
          +model.model_args.local_files_only=true \
          model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
          +model.tokenizer_args.local_files_only=true \
          forget_split="${forget_split}" holdout_split="${holdout_split}" retain_split="${retain_split}" \
          task_name="${task}" paths.output_dir="${out}" eval.tofu.retain_logs_path="${retain_logs}" \
          trainer.args.do_train=true trainer.args.do_eval=false trainer.args.eval_strategy=no \
          trainer.args.eval_on_start=false trainer.args.report_to=none trainer.args.logging_steps=1 \
          trainer.args.save_strategy=no trainer.args.gradient_checkpointing=true \
          trainer.args.ddp_find_unused_parameters=true trainer.args.dataloader_num_workers=0 \
          trainer.args.dataloader_drop_last=false trainer.args.learning_rate="${lr}" \
          trainer.args.per_device_train_batch_size="${bsz}" \
          trainer.args.gradient_accumulation_steps="${grad_acc}" trainer.args.num_train_epochs="${epochs}" \
          trainer.method_args.beta="${beta}" trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 \
          trainer.method_args.retain_loss_type=NLL trainer.method_args.moving_average_window="${ma}" \
          trainer.method_args.stop_ratio_threshold="${stop}" \
          trainer.method_args.rebound_ratio_threshold="${rebound}" \
          trainer.method_args.ratio_epsilon="${ratio_epsilon}" \
          trainer.method_args.initial_nll_cache_path="${cache}"
        test -f "${out}/config.json" || { echo "Training failed: ${out}"; exit 1; }
        rm -rf "${eval_out}"; mkdir -p "${eval_out}"
        python -u src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
          model="${MODEL_CONFIG}" model.model_args.pretrained_model_name_or_path="${out}" \
          +model.model_args.local_files_only=true \
          model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
          +model.tokenizer_args.local_files_only=true forget_split="${forget_split}" \
          holdout_split="${holdout_split}" retain_logs_path="${retain_logs}" \
          task_name="${task}_final_eval" paths.output_dir="${eval_out}"
    done; done; done; done
done; done; done; done
