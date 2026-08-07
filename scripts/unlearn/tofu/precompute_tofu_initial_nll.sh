#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_CONFIG="Llama-2-7b-chat-hf"
MODEL_TAG="Llama-2-7b-chat-hf"
TRAINER="SampleEarlyStopNPOLossIrreversible"
PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
CHECKPOINT_ROOT="/unlearning/experment_data/checkpoints"
CACHE_ROOT="${CHECKPOINT_ROOT}/tofu/initial_nll/${MODEL_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

mkdir -p "${CACHE_ROOT}"

for split in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"
    cache_file="${CACHE_ROOT}/${forget_split}.json"
    output_dir="${CACHE_ROOT}/precompute_state/${forget_split}"

    if [[ -f "${cache_file}" ]]; then
        echo "跳过已有缓存：${cache_file}"
        continue
    fi

    echo "预计算 ${forget_split} initial normalized NLL -> ${cache_file}"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="${TRAINER}" \
        collator=DataCollatorForSupervisedDatasetwithIndex \
        model="${MODEL_CONFIG}" \
        model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
        model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        retain_split="${retain_split}" \
        task_name="precompute_${MODEL_TAG}_${forget_split}_initial_nll" \
        paths.output_dir="${output_dir}" \
        ~eval \
        ~eval.tofu \
        trainer.args.do_train=false \
        trainer.args.do_eval=false \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=false \
        trainer.args.report_to=none \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.dataloader_drop_last=false \
        trainer.method_args.initial_nll_cache_path="${cache_file}"
done

echo "全部 initial NLL 缓存已就绪：${CACHE_ROOT}"
