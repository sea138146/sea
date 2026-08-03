#!/usr/bin/env bash
set -euo pipefail

cd /unlearning/experment_data/open-unlearning

source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="Llama-2-7b-chat-hf"
TRAINER="SampleStopMarginalNPO"
WORKER="./run_marginal_grid_worker.sh"

SPLITS=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# Fixed effective batch size:
# micro batch 4 x gradient accumulation 2 = effective batch 8.
EFFECTIVE_BATCH=8
MICRO_BATCH=4

LRS=("2e-5" "4e-5" "5e-5" "6e-5")
FORGETTING_RHOS=("0.80" "0.90")

WINDOW_SIZE=3
REACTIVATION_K=3
EPOCHS=10
SEED=0

LOG_ROOT="./logs/progress_rebound_grid24_bs8"
BASE_EVAL="./saves/eval_checkpoints/tofu"
GRID_SUMMARY_CSV="${LOG_ROOT}/grid_efficiency_summary.csv"
GRID_SUMMARY_JSONL="${LOG_ROOT}/grid_efficiency_metrics.jsonl"
GRID_RUN_SUMMARY="${LOG_ROOT}/grid_run_summary.json"

mkdir -p "${LOG_ROOT}"

# This grid intentionally reruns every combination, so start fresh summaries.
rm -f "${GRID_SUMMARY_CSV}" "${GRID_SUMMARY_JSONL}" "${GRID_RUN_SUMMARY}"

now_ns() {
    python - <<'PY'
import time
print(time.time_ns())
PY
}

elapsed_sec() {
    python - "$1" "$2" <<'PY'
import sys
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
print(f"{(end_ns - start_ns) / 1_000_000_000:.6f}")
PY
}

if [[ ! -f "${WORKER}" ]]; then
    echo "[ERROR] worker 不存在：${WORKER}"
    exit 1
fi

if (( EFFECTIVE_BATCH % MICRO_BATCH != 0 )); then
    echo "[ERROR] effective batch 无法被 micro batch 整除"
    echo "        EFFECTIVE_BATCH=${EFFECTIVE_BATCH}"
    echo "        MICRO_BATCH=${MICRO_BATCH}"
    exit 1
fi

GRAD_ACC=$((EFFECTIVE_BATCH / MICRO_BATCH))

if (( MICRO_BATCH * GRAD_ACC != EFFECTIVE_BATCH )); then
    echo "[ERROR] batch size 配置不一致"
    exit 1
fi

# Check all retain-reference evaluation files before launching the grid.
for spec in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${spec}"

    retain_log="/model/evals/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    if [[ ! -s "${retain_log}" ]]; then
        echo "[ERROR] 缺少 retain 基准日志：${retain_log}"
        exit 1
    fi
done

total=$((${#SPLITS[@]} * ${#LRS[@]} * ${#FORGETTING_RHOS[@]}))
current=0
GRID_START_NS=$(now_ns)

for spec in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${spec}"

    for lr in "${LRS[@]}"; do
        for forgetting_rho in "${FORGETTING_RHOS[@]}"; do
            current=$((current + 1))

            suffix="lr${lr}_eb${EFFECTIVE_BATCH}"
            suffix+="_b${MICRO_BATCH}_ga${GRAD_ACC}"
            suffix+="_e${EPOCHS}_s${SEED}"
            suffix+="_rho${forgetting_rho}_w${WINDOW_SIZE}_rk${REACTIVATION_K}"

            result_dir="${BASE_EVAL}/${forget_split}"
            result_dir+="/${MODEL}/${TRAINER}/${suffix}"

            log_file="${LOG_ROOT}/${forget_split}"
            log_file+="_eb${EFFECTIVE_BATCH}"
            log_file+="_lr${lr}_rho${forgetting_rho}_rk${REACTIVATION_K}.log"

            echo
            echo "============================================================"
            echo "[${current}/${total}] SampleStopMarginalNPO"
            echo "split           = ${forget_split}"
            echo "holdout         = ${holdout_split}"
            echo "retain          = ${retain_split}"
            echo "effective batch = ${EFFECTIVE_BATCH}"
            echo "micro batch     = ${MICRO_BATCH}"
            echo "gradient acc    = ${GRAD_ACC}"
            echo "learning rate   = ${lr}"
            echo "progress rho    = ${forgetting_rho}"
            echo "progress window = ${WINDOW_SIZE}"
            echo "reactivation k  = ${REACTIVATION_K}"
            echo "max epochs      = ${EPOCHS}"
            echo "seed            = ${SEED}"
            echo "result          = ${result_dir}"
            echo "============================================================"

            FORGETTING_RHO="${forgetting_rho}" \
            WINDOW_SIZE="${WINDOW_SIZE}" \
            REACTIVATION_K="${REACTIVATION_K}" \
            GRID_SPLIT="${spec}" \
            GRID_LR="${lr}" \
            GRID_BSZ="${MICRO_BATCH}" \
            GRID_GA="${GRAD_ACC}" \
            GRID_EFFECTIVE_BATCH="${EFFECTIVE_BATCH}" \
            GRID_EPOCHS="${EPOCHS}" \
            GRID_SEED="${SEED}" \
            bash "${WORKER}" 2>&1 | tee "${log_file}"

            if [[ ! -f "${result_dir}/GRID_DONE" ]]; then
                echo "[ERROR] 该组没有生成完成标记："
                echo "        ${result_dir}"
                exit 1
            fi

            metrics_file="${result_dir}/efficiency_metrics.json"
            if [[ ! -s "${metrics_file}" ]]; then
                echo "[ERROR] 缺少效率指标文件：${metrics_file}"
                exit 1
            fi

            # Append a compact CSV row and the complete JSON object.
            python - \
                "${metrics_file}" \
                "${GRID_SUMMARY_CSV}" \
                "${GRID_SUMMARY_JSONL}" \
                "${result_dir}" <<'PY'
import csv
import json
import pathlib
import sys

metrics_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
jsonl_path = pathlib.Path(sys.argv[3])
result_dir = sys.argv[4]

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics["result_dir"] = result_dir

splits = metrics.get("splits", {})
hparams = metrics.get("hyperparameters", {})
timing = metrics.get("timing", {})
compute = metrics.get("compute", {})

row = {
    "task_name": metrics.get("task_name"),
    "forget_split": splits.get("forget"),
    "holdout_split": splits.get("holdout"),
    "retain_split": splits.get("retain"),
    "learning_rate": hparams.get("learning_rate"),
    "forgetting_progress_threshold": hparams.get("forgetting_progress_threshold"),
    "window_size": hparams.get("window_size"),
    "reactivation_enabled": hparams.get("reactivation_enabled"),
    "reactivation_patience": hparams.get("reactivation_patience"),
    "micro_batch_size": hparams.get("micro_batch_size"),
    "gradient_accumulation_steps": hparams.get("gradient_accumulation_steps"),
    "effective_batch_size": hparams.get("effective_batch_size"),
    "configured_max_epochs": compute.get("configured_max_epochs"),
    "completed_epochs": compute.get("completed_epochs"),
    "optimizer_steps": compute.get("optimizer_steps"),
    "train_process_wall_time_sec": timing.get("train_process_wall_time_sec"),
    "trainer_reported_train_runtime_sec": timing.get("trainer_reported_train_runtime_sec"),
    "evaluation_phase_wall_time_sec": timing.get("evaluation_phase_wall_time_sec"),
    "total_combo_wall_time_sec": timing.get("total_combo_wall_time_sec"),
    "single_gpu_train_process_hours": timing.get("single_gpu_train_process_hours"),
    "result_dir": result_dir,
}

csv_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not csv_path.exists()
with csv_path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    if write_header:
        writer.writeheader()
    writer.writerow(row)

with jsonl_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
PY

            echo "[OK] 完成 ${current}/${total}"
            echo "[SUMMARY] ${GRID_SUMMARY_CSV}"
        done
    done
done

GRID_END_NS=$(now_ns)
GRID_TOTAL_WALL_SEC=$(elapsed_sec "${GRID_START_NS}" "${GRID_END_NS}")

python - \
    "${GRID_RUN_SUMMARY}" \
    "${GRID_TOTAL_WALL_SEC}" \
    "${total}" \
    "${GRID_SUMMARY_CSV}" \
    "${GRID_SUMMARY_JSONL}" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
payload = {
    "grid_total_wall_time_sec": float(sys.argv[2]),
    "grid_total_wall_time_hours": float(sys.argv[2]) / 3600.0,
    "completed_combinations": int(sys.argv[3]),
    "summary_csv": sys.argv[4],
    "metrics_jsonl": sys.argv[5],
    "note": "Grid total time includes training, evaluation, cleanup, and all combinations; do not use it as the per-method efficiency metric.",
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "============================================================"
echo "[DONE] ${total} 组 batch-size-8 网格实验全部完成"
echo "[TIME] grid total wall time: ${GRID_TOTAL_WALL_SEC} sec"
echo "[CSV]  ${GRID_SUMMARY_CSV}"
echo "[JSONL] ${GRID_SUMMARY_JSONL}"
echo "[GRID] ${GRID_RUN_SUMMARY}"
echo "============================================================"
