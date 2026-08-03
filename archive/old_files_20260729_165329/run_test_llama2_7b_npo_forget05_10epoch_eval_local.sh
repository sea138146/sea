#!/usr/bin/env bash
set -euo pipefail

cd /unlearning/experment_data/open-unlearning

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

REPORTTO="none"
WANDB_PROJECT="BalDRO"

MODEL="Llama-2-7b-chat-hf"
TRAINER="NPO"

forget_split="forget05"
holdout_split="holdout05"
retain_split="retain95"

PRETRAINED_PATH="/model/finetune_models/tofu_Llama-2-7b-chat-hf_full"
TOKENIZER_PATH="/model/Llama/Llama-2-7b-chat-hf"
RETAIN_LOGS_PATH="/model/evals/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

lr="1e-5"
bsz="8"
grad_acc="2"
EPOCHS="1"
DO_SAVE=true

SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_e${EPOCHS}"
TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

test -d "${PRETRAINED_PATH}" || { echo "[ERROR] missing PRETRAINED_PATH: ${PRETRAINED_PATH}"; exit 1; }
test -d "${TOKENIZER_PATH}" || { echo "[ERROR] missing TOKENIZER_PATH: ${TOKENIZER_PATH}"; exit 1; }
test -f "${RETAIN_LOGS_PATH}" || { echo "[ERROR] missing RETAIN_LOGS_PATH: ${RETAIN_LOGS_PATH}"; exit 1; }

echo "[INFO] python: $(which python)"
python -V

echo "============================================================"
echo "[TRAIN + EVAL TEST]"
echo "[TASK]      ${TASK_NAME}"
echo "[OUTPUT]    ${OUTPUT_DIR}"
echo "[MODEL]     ${PRETRAINED_PATH}"
echo "[TOKENIZER] ${TOKENIZER_PATH}"
echo "[RETAIN]    ${RETAIN_LOGS_PATH}"
echo "[EPOCHS]    ${EPOCHS}"
echo "============================================================"

rm -rf "${OUTPUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT}"

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer="${TRAINER}" \
  model="${MODEL}" \
  model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_split="${retain_split}" \
  task_name="${TASK_NAME}" \
  paths.output_dir="${OUTPUT_DIR}" \
  eval.tofu.retain_logs_path="${RETAIN_LOGS_PATH}" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.report_to="${REPORTTO}" \
  trainer.args.logging_steps=1 \
  trainer.args.learning_rate="${lr}" \
  trainer.args.per_device_train_batch_size="${bsz}" \
  trainer.args.gradient_accumulation_steps="${grad_acc}" \
  trainer.args.num_train_epochs="${EPOCHS}" \
  trainer.args.eval_strategy=no \
  trainer.args.save_strategy=epoch \
  trainer.args.eval_on_start=false

echo "============================================================"
echo "[DONE] training and epoch eval finished"
echo "[OUTPUT] ${OUTPUT_DIR}"
echo "============================================================"

echo "[INFO] eval files:"
find "${OUTPUT_DIR}" -type f \( -name "TOFU_EVAL.json" -o -name "TOFU_SUMMARY.json" \) -print

echo "[INFO] quick summary:"
python - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
files = list(out.glob("checkpoint-*/evals/TOFU_SUMMARY.json"))

if not files:
    print("[WARN] no TOFU_SUMMARY.json found")
    raise SystemExit(0)

def ckpt_step(p):
    name = p.parents[1].name
    try:
        return int(name.split("-")[-1])
    except Exception:
        return -1

summary_path = sorted(files, key=ckpt_step)[-1]
eval_path = summary_path.parent / "TOFU_EVAL.json"

summary = json.loads(summary_path.read_text(encoding="utf-8"))
evals = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}

def s(*keys):
    for k in keys:
        if k in summary:
            return summary[k]
    return ""

def a(*keys):
    for k in keys:
        v = evals.get(k)
        if isinstance(v, dict):
            return v.get("agg_value", v.get("mean", ""))
        if v is not None:
            return v
    return ""

row = {
    "Method": out.name,
    "FQ": s("forget_quality", "FQ"),
    "MU": s("model_utility", "MU"),
    "Fluency": s("fluency", "Fluency", "fluency_score"),
    "EM": s("exact_memorization", "EM"),
    "ES": s("extraction_strength", "ES"),
    "F-TR": s("forget_truth_ratio", "F-TR", "forget_TR"),
    "Ra-TR": a("ra_Truth_Ratio", "Ra_Truth_Ratio"),
    "R-TR": a("retain_Truth_Ratio", "R_Truth_Ratio"),
    "Rw-TR": a("rw_Truth_Ratio", "wf_Truth_Ratio", "Rw_Truth_Ratio"),
}

print("[SUMMARY_PATH]", summary_path)
for k, v in row.items():
    print(f"{k}: {v}")
PY
