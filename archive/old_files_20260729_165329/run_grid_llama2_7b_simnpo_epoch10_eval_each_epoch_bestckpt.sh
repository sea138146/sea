#!/usr/bin/env bash
set -euo pipefail

cd /unlearning/experment_data/open-unlearning

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")

# 用法：
# bash run_grid_llama2_7b_simnpo_epoch10_eval_each_epoch_bestckpt.sh

export CUDA_VISIBLE_DEVICES=0

MODEL="Llama-2-7b-chat-hf"
REPORTTO="none"

TRAINER="SimNPO"
PRETRAINED_PATH="/model/finetune_models/tofu_${MODEL}_full"
TOKENIZER_PATH="/model/Llama/Llama-2-7b-chat-hf"

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="./logs/unlearn/tofu/${MODEL}/${TRAINER}/day${DATE}_time${TIME}"
mkdir -p "${LOG_DIR}"

# =========================
# Grid
# 按别人 SimNPO 脚本补全 alpha / beta / delta / gamma
# =========================

splits=(
    "forget05 holdout05 retain95"
    "forget01 holdout01 retain99"
    "forget10 holdout10 retain90"
)

lr_set=("2e-5" "5e-5" "6e-5")

# 每项是：micro_batch gradient_accumulation
# "8 2" -> Eff-B=16
# "8 4" -> Eff-B=32
bz_set=("8 2" "8 4")

alpha=1.0
beta_set=(3.5 4.5)
delta_set=(0 1)
gamma_set=(0.125 0.25)

epoch_set=("10")

calc_eff_bsz() {
    local bsz="$1"
    local grad_acc="$2"
    echo $(( bsz * grad_acc ))
}

select_best_and_clean_checkpoints() {
    local output_dir="$1"

    if [ ! -d "${output_dir}" ]; then
        echo "[CLEAN] output_dir not found: ${output_dir}"
        return 0
    fi

    python - "${output_dir}" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def get_any(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

def metric(data, keys):
    v = get_any(data, keys, None)
    if v is not None:
        return v

    for prefix in ["metrics", "summary", "aggregate", "aggregated", "eval_log", "TOFU"]:
        cur = data.get(prefix) if isinstance(data, dict) else None
        if isinstance(cur, dict):
            v = get_any(cur, keys, None)
            if v is not None:
                return v
    return None

rows = []

for summary_path in output_dir.glob("checkpoint-*/evals/TOFU_SUMMARY.json"):
    ckpt_dir = summary_path.parents[1]
    m = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
    ckpt = int(m.group(1)) if m else -1

    try:
        data = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"[CLEAN] bad json: {summary_path} {e}")
        continue

    fq = metric(data, ["forget_quality", "Forget Quality", "FQ", "fq"])
    mu = metric(data, ["model_utility", "Model Utility", "MU", "mu"])
    em = metric(data, ["exact_memorization", "Exact Memorization", "EM", "em"])
    es = metric(data, ["extraction_strength", "Extraction Strength", "ES", "es"])
    score = metric(data, ["score", "Score"])

    score_f = to_float(score)

    # 没有显式 Score 时，用近似排序：FQ + MU - EM - ES
    if score_f is None:
        vals = [to_float(x) for x in [fq, mu, em, es]]
        if all(v is not None for v in vals):
            score_f = vals[0] + vals[1] - vals[2] - vals[3]
        else:
            score_f = float("-inf")

    rows.append({
        "ckpt": ckpt,
        "ckpt_dir": ckpt_dir,
        "score": score_f,
        "fq": to_float(fq),
        "mu": to_float(mu),
        "em": to_float(em),
        "es": to_float(es),
    })

if not rows:
    print(f"[CLEAN] no TOFU_SUMMARY.json found under {output_dir}")
    sys.exit(0)

rows.sort(key=lambda r: (r["score"], r["ckpt"]), reverse=True)
best = rows[0]
best_dir = best["ckpt_dir"]

print("[BEST]", best_dir)
print(
    "[BEST_METRICS]",
    f"ckpt={best['ckpt']}",
    f"score={best['score']:.6f}",
    f"FQ={best['fq']}",
    f"MU={best['mu']}",
    f"EM={best['em']}",
    f"ES={best['es']}",
)

best_json = output_dir / "best_checkpoint.json"
best_json.write_text(json.dumps({
    "best_checkpoint": best_dir.name,
    "best_checkpoint_path": str(best_dir),
    "score": best["score"],
    "FQ": best["fq"],
    "MU": best["mu"],
    "EM": best["em"],
    "ES": best["es"],
}, indent=2, ensure_ascii=False))

# 非最佳 checkpoint 只保留 evals/
for r in rows:
    ckpt_dir = r["ckpt_dir"]

    if ckpt_dir == best_dir:
        print(f"[KEEP_FULL] {ckpt_dir}")
        continue

    if not ckpt_dir.exists():
        continue

    for item in ckpt_dir.iterdir():
        if item.name == "evals":
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"[CLEAN_WARN] failed to remove {item}: {e}")

    print(f"[KEEP_EVALS_ONLY] {ckpt_dir}")

print(f"[CLEAN_DONE] best={best_dir.name}")
PY
}

echo "============================================================"
echo "[START] ${TRAINER} grid"
echo "[MODEL] ${MODEL}"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES}"
echo "[REPORTTO] ${REPORTTO}"
echo "[LOG_DIR] ${LOG_DIR}"
echo "============================================================"

for split in "${splits[@]}"; do
    forget_split=$(echo "${split}" | cut -d' ' -f1)
    holdout_split=$(echo "${split}" | cut -d' ' -f2)
    retain_split=$(echo "${split}" | cut -d' ' -f3)

    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            bsz=$(echo "${bz}" | cut -d' ' -f1)
            grad_acc=$(echo "${bz}" | cut -d' ' -f2)
            eff_bsz=$(calc_eff_bsz "${bsz}" "${grad_acc}")

            for epochs in "${epoch_set[@]}"; do
                for beta in "${beta_set[@]}"; do
                    for delta in "${delta_set[@]}"; do
                        for gamma in "${gamma_set[@]}"; do

                            SUFFIX="lr${lr}_b${eff_bsz}_mb${bsz}_ga${grad_acc}_a${alpha}_beta${beta}_delta${delta}_gamma${gamma}_e${epochs}"
                            TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}_day${DATE}_time${TIME}"
                            OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"
                            RETAIN_LOGS_PATH="/model/evals/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

                            echo
                            echo "============================================================"
                            echo "[EXPERIMENT]"
                            echo "[SPLIT]      ${forget_split} / ${holdout_split} / ${retain_split}"
                            echo "[TRAINER]    ${TRAINER}"
                            echo "[MODEL]      ${MODEL}"
                            echo "[LR]         ${lr}"
                            echo "[BSZ]        ${bsz}"
                            echo "[GA]         ${grad_acc}"
                            echo "[EFF_BSZ]    ${eff_bsz}"
                            echo "[EPOCHS]     ${epochs}"
                            echo "[ALPHA]      ${alpha}"
                            echo "[BETA]       ${beta}"
                            echo "[DELTA]      ${delta}"
                            echo "[GAMMA]      ${gamma}"
                            echo "[OUTPUT]     ${OUTPUT_DIR}"
                            echo "[RETAINLOG]  ${RETAIN_LOGS_PATH}"
                            echo "============================================================"

                            rm -rf "${OUTPUT_DIR}"

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
                                trainer.args.num_train_epochs="${epochs}" \
                                trainer.args.eval_strategy=epoch \
                                trainer.args.eval_on_start=False \
                                trainer.args.save_strategy=epoch \
                                trainer.method_args.gamma="${gamma}" \
                                trainer.method_args.alpha="${alpha}" \
                                trainer.method_args.retain_loss_type=NLL \
                                trainer.method_args.beta="${beta}" \
                                trainer.method_args.delta="${delta}" \
                                2>&1 | tee "${LOG_DIR}/${forget_split}_${SUFFIX}.log"

                            echo "[TRAIN+EVAL DONE] ${TRAINER} ${forget_split} ${SUFFIX}"

                            select_best_and_clean_checkpoints "${OUTPUT_DIR}"

                            echo "[EXPERIMENT DONE] ${TRAINER} ${forget_split} ${SUFFIX}"

                        done
                    done
                done
            done
        done
    done
done

echo
echo "============================================================"
echo "[ALL DONE] ${TRAINER} grid finished."
echo "[LOG_DIR] ${LOG_DIR}"
echo "============================================================"
