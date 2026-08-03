#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate openunlearning

cd /unlearning/experment_data/open-unlearning

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPORTTO="none"

MODEL="Llama-2-7b-chat-hf"
TRAINER="SampleStopMarginalNPO"

FORGETTING_RHO="${FORGETTING_RHO:?FORGETTING_RHO is required}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
REACTIVATION_K="${REACTIVATION_K:-3}"

: "${GRID_SPLIT:?GRID_SPLIT is required}"
: "${GRID_LR:?GRID_LR is required}"
: "${GRID_BSZ:?GRID_BSZ is required}"
: "${GRID_GA:?GRID_GA is required}"

GRID_EFFECTIVE_BATCH="${GRID_EFFECTIVE_BATCH:-8}"
GRID_EPOCHS="${GRID_EPOCHS:-10}"
GRID_SEED="${GRID_SEED:-0}"

PRETRAINED_PATH="/model/finetune_models/tofu_${MODEL}_full"

# Nanosecond-resolution wall-clock helpers. The train.py process does not
# return until its CUDA work has completed, so process-level timing is valid.
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

TRAIN_METADATA_FILE=""
EVAL_TIMINGS_FILE=""
cleanup_temp_files() {
    [[ -z "${TRAIN_METADATA_FILE}" ]] || rm -f "${TRAIN_METADATA_FILE}"
    [[ -z "${EVAL_TIMINGS_FILE}" ]] || rm -f "${EVAL_TIMINGS_FILE}"
}
trap cleanup_temp_files EXIT

for value_name in GRID_BSZ GRID_GA GRID_EFFECTIVE_BATCH GRID_EPOCHS GRID_SEED WINDOW_SIZE REACTIVATION_K; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} 必须是非负整数，当前值：${value}"
        exit 1
    fi
done

if (( GRID_BSZ <= 0 || GRID_GA <= 0 || GRID_EFFECTIVE_BATCH <= 0 || GRID_EPOCHS <= 0 || WINDOW_SIZE <= 0 || REACTIVATION_K <= 0 )); then
    echo "[ERROR] batch size、gradient accumulation、epochs、window size 和 reactivation k 必须大于 0"
    exit 1
fi

if (( WINDOW_SIZE != 3 )); then
    echo "[ERROR] SampleStopMarginalNPO 的 progress window 必须等于 3"
    echo "        当前值：${WINDOW_SIZE}"
    exit 1
fi

actual_effective_batch=$((GRID_BSZ * GRID_GA))
if (( actual_effective_batch != GRID_EFFECTIVE_BATCH )); then
    echo "[ERROR] effective batch size 不一致"
    echo "        GRID_BSZ=${GRID_BSZ}"
    echo "        GRID_GA=${GRID_GA}"
    echo "        实际 effective batch=${actual_effective_batch}"
    echo "        预期 effective batch=${GRID_EFFECTIVE_BATCH}"
    exit 1
fi

if (( GRID_EFFECTIVE_BATCH != 8 )); then
    echo "[ERROR] 本 worker 仅用于 effective batch size = 8"
    echo "        当前值：${GRID_EFFECTIVE_BATCH}"
    exit 1
fi

if ! python - "${FORGETTING_RHO}" <<'PY'
import math
import sys

try:
    rho = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)

if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
    raise SystemExit(1)
PY
then
    echo "[ERROR] FORGETTING_RHO 必须是 [0, 1] 内的有限数值"
    echo "        当前值：${FORGETTING_RHO}"
    exit 1
fi

read -r forget_split holdout_split retain_split extra <<< "${GRID_SPLIT}"
if [[ -z "${forget_split:-}" || -z "${holdout_split:-}" || -z "${retain_split:-}" || -n "${extra:-}" ]]; then
    echo "[ERROR] GRID_SPLIT 格式必须是：forget_split holdout_split retain_split"
    echo "        当前值：${GRID_SPLIT}"
    exit 1
fi

lr="${GRID_LR}"
bsz="${GRID_BSZ}"
grad_acc="${GRID_GA}"
epochs="${GRID_EPOCHS}"
seed="${GRID_SEED}"

RETAIN_LOGS="/model/evals/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

test -d "${PRETRAINED_PATH}" || {
    echo "[ERROR] Model not found: ${PRETRAINED_PATH}"
    exit 1
}

test -s "${RETAIN_LOGS}" || {
    echo "[ERROR] Retain log not found or empty: ${RETAIN_LOGS}"
    exit 1
}

SUFFIX="lr${lr}_eb${actual_effective_batch}"
SUFFIX+="_b${bsz}_ga${grad_acc}"
SUFFIX+="_e${epochs}_s${seed}"
SUFFIX+="_rho${FORGETTING_RHO}_w${WINDOW_SIZE}_rk${REACTIVATION_K}"

TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"
FINAL_RESULT_DIR="./saves/eval_checkpoints/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

# Every invocation reruns the combination from scratch.
if [[ -d "${OUTPUT_DIR}" || -d "${FINAL_RESULT_DIR}" ]]; then
    echo "[CLEAN] 清理已有结果并重新运行：${SUFFIX}"
    rm -rf "${OUTPUT_DIR}" "${FINAL_RESULT_DIR}"
fi

mkdir -p "${FINAL_RESULT_DIR}"
TRAIN_METADATA_FILE=$(mktemp)
EVAL_TIMINGS_FILE=$(mktemp)

COMBO_START_NS=$(now_ns)

printf '%s\n' "=================================================="
printf '%-13s: %s\n' "Trainer" "${TRAINER}"
printf '%-13s: %s\n' "Model" "${MODEL}"
printf '%-13s: %s\n' "Forget split" "${forget_split}"
printf '%-13s: %s\n' "Holdout split" "${holdout_split}"
printf '%-13s: %s\n' "Retain split" "${retain_split}"
printf '%-13s: %s\n' "Learning rate" "${lr}"
printf '%-13s: %s\n' "Micro batch" "${bsz}"
printf '%-13s: %s\n' "Grad acc" "${grad_acc}"
printf '%-13s: %s\n' "Effective BS" "${actual_effective_batch}"
printf '%-13s: %s\n' "Max epochs" "${epochs}"
printf '%-13s: %s\n' "Seed" "${seed}"
printf '%-13s: %s\n' "Progress rho" "${FORGETTING_RHO}"
printf '%-13s: %s\n' "Progress win" "${WINDOW_SIZE}"
printf '%-13s: %s\n' "Reactivation k" "${REACTIVATION_K}"
printf '%-13s: %s\n' "Output" "${OUTPUT_DIR}"
printf '%s\n' "=================================================="

# ---------------------------------------------------------------------------
# Training-process wall time
# Includes Python startup, model/data loading, trainer.train(), logging, and
# checkpoint writes. It excludes the separate TOFU evaluation below.
# ---------------------------------------------------------------------------
TRAIN_START_NS=$(now_ns)

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer="${TRAINER}" \
    trainer.method_args.forgetting_progress_threshold="${FORGETTING_RHO}" \
    trainer.method_args.window_size="${WINDOW_SIZE}" \
    trainer.method_args.track_stopped_samples=true \
    trainer.method_args.reactivation_enabled=true \
    trainer.method_args.reactivation_patience="${REACTIVATION_K}" \
    model="${MODEL}" \
    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    model.model_args.torch_dtype=bfloat16 \
    model.model_args.attn_implementation=flash_attention_2 \
    ++model.model_args.use_cache=false \
    forget_split="${forget_split}" \
    holdout_split="${holdout_split}" \
    retain_split="${retain_split}" \
    task_name="${TASK_NAME}" \
    paths.output_dir="${OUTPUT_DIR}" \
    ++do_save=true \
    eval.tofu.retain_logs_path="${RETAIN_LOGS}" \
    ++trainer.args.seed="${seed}" \
    ++trainer.args.learning_rate="${lr}" \
    ++trainer.args.per_device_train_batch_size="${bsz}" \
    ++trainer.args.gradient_accumulation_steps="${grad_acc}" \
    ++trainer.args.num_train_epochs="${epochs}" \
    ++trainer.args.max_steps=-1 \
    ++trainer.args.dataloader_drop_last=false \
    ++trainer.args.gradient_checkpointing=true \
    ++trainer.args.bf16=true \
    ++trainer.args.fp16=false \
    ++trainer.args.ddp_find_unused_parameters=false \
    ++trainer.args.remove_unused_columns=false \
    ++trainer.args.report_to="${REPORTTO}" \
    ++trainer.args.run_name="${TASK_NAME}" \
    ++trainer.args.logging_steps=1 \
    ++trainer.args.eval_strategy=no \
    ++trainer.args.eval_on_start=false \
    ++trainer.args.do_eval=false \
    ++trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=1 \
    ++trainer.args.save_only_model=true

TRAIN_END_NS=$(now_ns)
TRAIN_PROCESS_WALL_SEC=$(elapsed_sec "${TRAIN_START_NS}" "${TRAIN_END_NS}")

echo "[TIME] train process wall time: ${TRAIN_PROCESS_WALL_SEC} sec"

# Evaluate the checkpoint(s) retained by Trainer. With save_total_limit=1,
# this normally evaluates only the final retained checkpoint.
EVAL_ROOT="${FINAL_RESULT_DIR}"

mapfile -t CHECKPOINTS < <(
    find "${OUTPUT_DIR}" \
        -maxdepth 1 \
        -mindepth 1 \
        -type d \
        -name "checkpoint-*" \
        | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "[ERROR] No checkpoints found in ${OUTPUT_DIR}"
    exit 1
fi

MAX_CHECKPOINT_STEP=0
for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_step="${checkpoint##*-}"
    if [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] && (( checkpoint_step > MAX_CHECKPOINT_STEP )); then
        MAX_CHECKPOINT_STEP="${checkpoint_step}"
    fi
done

# Preserve Hugging Face Trainer metrics before checkpoints are deleted.
python - "${OUTPUT_DIR}" "${MAX_CHECKPOINT_STEP}" > "${TRAIN_METADATA_FILE}" <<'PY'
import json
import pathlib
import sys
from typing import Any

output_dir = pathlib.Path(sys.argv[1])
fallback_step = int(sys.argv[2])

result: dict[str, Any] = {
    "trainer_state_path": None,
    "global_step": fallback_step,
    "epoch": None,
    "trainer_metrics": {},
}

state_paths = sorted(output_dir.rglob("trainer_state.json"))
if state_paths:
    state_path = state_paths[-1]
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        result["trainer_state_path"] = str(state_path)
        result["global_step"] = state.get("global_step", fallback_step)
        result["epoch"] = state.get("epoch")

        history = state.get("log_history") or []
        for entry in reversed(history):
            if isinstance(entry, dict) and (
                "train_runtime" in entry
                or "train_loss" in entry
                or "train_steps_per_second" in entry
            ):
                result["trainer_metrics"].update(entry)
                break
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result["trainer_state_error"] = str(exc)

for filename in ("train_results.json", "all_results.json"):
    for path in sorted(output_dir.rglob(filename)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                result["trainer_metrics"].update(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

print(json.dumps(result, ensure_ascii=False))
PY

printf '[INFO] Found %d checkpoint(s); final step=%s.\n' "${#CHECKPOINTS[@]}" "${MAX_CHECKPOINT_STEP}"

EVAL_PHASE_START_NS=$(now_ns)

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    STEP="${CHECKPOINT##*-}"
    EVAL_TASK="${TASK_NAME}_checkpoint_${STEP}_eval"
    EVAL_DIR="${EVAL_ROOT}/checkpoint-${STEP}"

    echo "=================================================="
    echo "Evaluating checkpoint: ${CHECKPOINT}"
    echo "Evaluation output    : ${EVAL_DIR}"
    echo "=================================================="

    rm -rf "${EVAL_DIR}"
    mkdir -p "${EVAL_DIR}"

    CHECKPOINT_EVAL_START_NS=$(now_ns)

    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${CHECKPOINT}" \
        model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        retain_logs_path="${RETAIN_LOGS}" \
        task_name="${EVAL_TASK}" \
        paths.output_dir="${EVAL_DIR}" \
        eval.tofu.overwrite=true

    CHECKPOINT_EVAL_END_NS=$(now_ns)
    CHECKPOINT_EVAL_WALL_SEC=$(elapsed_sec "${CHECKPOINT_EVAL_START_NS}" "${CHECKPOINT_EVAL_END_NS}")
    printf '%s\t%s\n' "${STEP}" "${CHECKPOINT_EVAL_WALL_SEC}" >> "${EVAL_TIMINGS_FILE}"
    echo "[TIME] checkpoint-${STEP} eval process wall time: ${CHECKPOINT_EVAL_WALL_SEC} sec"

    EVAL_JSON=$(
        find "${EVAL_DIR}" \
            -type f \
            -name "TOFU_EVAL.json" \
            -print \
            -quit
    )

    SUMMARY_JSON=$(
        find "${EVAL_DIR}" \
            -type f \
            -name "TOFU_SUMMARY.json" \
            -print \
            -quit
    )

    if [[ -z "${EVAL_JSON}" ]]; then
        echo "[ERROR] TOFU_EVAL.json was not generated."
        echo "[INFO] Keeping checkpoint: ${CHECKPOINT}"
        exit 1
    fi

    if [[ -z "${SUMMARY_JSON}" ]]; then
        echo "[ERROR] TOFU_SUMMARY.json was not generated."
        echo "[INFO] Keeping checkpoint: ${CHECKPOINT}"
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    cp "${EVAL_JSON}" "${TEMP_DIR}/TOFU_EVAL.json"
    cp "${SUMMARY_JSON}" "${TEMP_DIR}/TOFU_SUMMARY.json"

    rm -rf "${EVAL_DIR}"
    mkdir -p "${EVAL_DIR}"

    mv "${TEMP_DIR}/TOFU_EVAL.json" "${EVAL_DIR}/TOFU_EVAL.json"
    mv "${TEMP_DIR}/TOFU_SUMMARY.json" "${EVAL_DIR}/TOFU_SUMMARY.json"
    rm -rf "${TEMP_DIR}"

    # Delete weights only after both evaluation files have been preserved.
    rm -rf "${CHECKPOINT}"

    echo "[OK] Finished checkpoint-${STEP}"
    echo "[OK] Kept only:"
    echo "     ${EVAL_DIR}/TOFU_EVAL.json"
    echo "     ${EVAL_DIR}/TOFU_SUMMARY.json"
done

EVAL_PHASE_END_NS=$(now_ns)
EVALUATION_PHASE_WALL_SEC=$(elapsed_sec "${EVAL_PHASE_START_NS}" "${EVAL_PHASE_END_NS}")
echo "[TIME] complete evaluation phase wall time: ${EVALUATION_PHASE_WALL_SEC} sec"

POSTPROCESS_START_NS=$(now_ns)

STOP_RESULT_DIR="${FINAL_RESULT_DIR}"
mkdir -p "${STOP_RESULT_DIR}"

cp "${OUTPUT_DIR}/sample_stop_summary.json" "${STOP_RESULT_DIR}/" 2>/dev/null || true
cp "${OUTPUT_DIR}/sample_stop_log.jsonl" "${STOP_RESULT_DIR}/" 2>/dev/null || true
cp "${OUTPUT_DIR}/sample_trajectory_log.jsonl" "${STOP_RESULT_DIR}/" 2>/dev/null || true
cp "${OUTPUT_DIR}/sample_violation_log.jsonl" "${STOP_RESULT_DIR}/" 2>/dev/null || true
cp "${OUTPUT_DIR}/sample_reactivation_log.jsonl" "${STOP_RESULT_DIR}/" 2>/dev/null || true

for required_file in \
    sample_stop_summary.json \
    sample_trajectory_log.jsonl
do
    if [[ ! -s "${STOP_RESULT_DIR}/${required_file}" ]]; then
        echo "[ERROR] 缺少或为空："
        echo "        ${STOP_RESULT_DIR}/${required_file}"
        exit 1
    fi
done

# No sample reaching the threshold is a valid experimental outcome, so the
# stop-event log may exist but be empty.
if [[ ! -f "${STOP_RESULT_DIR}/sample_stop_log.jsonl" ]]; then
    echo "[ERROR] 缺少停止事件日志："
    echo "        ${STOP_RESULT_DIR}/sample_stop_log.jsonl"
    exit 1
fi

# Rebound checks are enabled. Either log may be empty when no threshold violation or reactivation occurs.
if [[ ! -f "${STOP_RESULT_DIR}/sample_violation_log.jsonl" ]]; then
    echo "[ERROR] 缺少 violation 日志文件："
    echo "        ${STOP_RESULT_DIR}/sample_violation_log.jsonl"
    exit 1
fi

if [[ ! -f "${STOP_RESULT_DIR}/sample_reactivation_log.jsonl" ]]; then
    echo "[ERROR] 缺少 reactivation 日志文件："
    echo "        ${STOP_RESULT_DIR}/sample_reactivation_log.jsonl"
    exit 1
fi

if ! find "${EVAL_ROOT}" \
    -mindepth 2 -maxdepth 2 \
    -type f -name "TOFU_EVAL.json" \
    -size +0c -print -quit | grep -q .
then
    echo "[ERROR] 缺少 TOFU_EVAL.json"
    exit 1
fi

if ! find "${EVAL_ROOT}" \
    -mindepth 2 -maxdepth 2 \
    -type f -name "TOFU_SUMMARY.json" \
    -size +0c -print -quit | grep -q .
then
    echo "[ERROR] 缺少 TOFU_SUMMARY.json"
    exit 1
fi

# All required outputs have been copied, so the temporary training tree can go.
rm -rf "${OUTPUT_DIR}"

POSTPROCESS_END_NS=$(now_ns)
POSTPROCESS_WALL_SEC=$(elapsed_sec "${POSTPROCESS_START_NS}" "${POSTPROCESS_END_NS}")
COMBO_END_NS=$(now_ns)
TOTAL_COMBO_WALL_SEC=$(elapsed_sec "${COMBO_START_NS}" "${COMBO_END_NS}")

# Build one machine-readable efficiency record per grid combination.
export METRIC_SCHEMA_VERSION="2"
export METRIC_TASK_NAME="${TASK_NAME}"
export METRIC_MODEL="${MODEL}"
export METRIC_TRAINER="${TRAINER}"
export METRIC_FORGET_SPLIT="${forget_split}"
export METRIC_HOLDOUT_SPLIT="${holdout_split}"
export METRIC_RETAIN_SPLIT="${retain_split}"
export METRIC_LR="${lr}"
export METRIC_FORGETTING_RHO="${FORGETTING_RHO}"
export METRIC_WINDOW_SIZE="${WINDOW_SIZE}"
export METRIC_REACTIVATION_K="${REACTIVATION_K}"
export METRIC_MICRO_BATCH="${bsz}"
export METRIC_GRAD_ACC="${grad_acc}"
export METRIC_EFFECTIVE_BATCH="${actual_effective_batch}"
export METRIC_MAX_EPOCHS="${epochs}"
export METRIC_SEED="${seed}"
export METRIC_TRAIN_PROCESS_SEC="${TRAIN_PROCESS_WALL_SEC}"
export METRIC_EVALUATION_PHASE_SEC="${EVALUATION_PHASE_WALL_SEC}"
export METRIC_POSTPROCESS_SEC="${POSTPROCESS_WALL_SEC}"
export METRIC_TOTAL_COMBO_SEC="${TOTAL_COMBO_WALL_SEC}"
export METRIC_FINAL_CHECKPOINT_STEP="${MAX_CHECKPOINT_STEP}"
export METRIC_NUM_CHECKPOINTS="${#CHECKPOINTS[@]}"
export METRIC_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

python - \
    "${TRAIN_METADATA_FILE}" \
    "${EVAL_TIMINGS_FILE}" \
    "${STOP_RESULT_DIR}/sample_stop_summary.json" \
    "${STOP_RESULT_DIR}/efficiency_metrics.json" <<'PY'
import json
import math
import os
import pathlib
import sys
from typing import Any

train_metadata_path = pathlib.Path(sys.argv[1])
eval_timings_path = pathlib.Path(sys.argv[2])
sample_summary_path = pathlib.Path(sys.argv[3])
output_path = pathlib.Path(sys.argv[4])


def as_int(name: str) -> int:
    return int(os.environ[name])


def as_float(name: str) -> float:
    return float(os.environ[name])


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"value": payload}
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}


def flatten_numeric_dict(value: Any, prefix: str = "") -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric_dict(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return result
        result[prefix] = value
    return result


trainer_metadata = load_json(train_metadata_path)
sample_stop_summary = load_json(sample_summary_path)

eval_timings: list[dict[str, Any]] = []
if eval_timings_path.exists():
    for line in eval_timings_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step_text, seconds_text = line.split("\t", maxsplit=1)
        eval_timings.append(
            {
                "checkpoint_step": int(step_text),
                "eval_process_wall_time_sec": float(seconds_text),
            }
        )

trainer_metrics = trainer_metadata.get("trainer_metrics")
if not isinstance(trainer_metrics, dict):
    trainer_metrics = {}

trainer_runtime = trainer_metrics.get("train_runtime")
if not isinstance(trainer_runtime, (int, float)):
    trainer_runtime = None

optimizer_steps = trainer_metadata.get("global_step")
if not isinstance(optimizer_steps, (int, float)):
    optimizer_steps = as_int("METRIC_FINAL_CHECKPOINT_STEP")

completed_epochs = trainer_metadata.get("epoch")
if not isinstance(completed_epochs, (int, float)):
    completed_epochs = trainer_metrics.get("epoch")
if not isinstance(completed_epochs, (int, float)):
    completed_epochs = None

train_process_sec = as_float("METRIC_TRAIN_PROCESS_SEC")
eval_process_total = sum(item["eval_process_wall_time_sec"] for item in eval_timings)
total_combo_sec = as_float("METRIC_TOTAL_COMBO_SEC")

metrics = {
    "schema_version": as_int("METRIC_SCHEMA_VERSION"),
    "task_name": os.environ["METRIC_TASK_NAME"],
    "model": os.environ["METRIC_MODEL"],
    "trainer": os.environ["METRIC_TRAINER"],
    "splits": {
        "forget": os.environ["METRIC_FORGET_SPLIT"],
        "holdout": os.environ["METRIC_HOLDOUT_SPLIT"],
        "retain": os.environ["METRIC_RETAIN_SPLIT"],
    },
    "hyperparameters": {
        "learning_rate": os.environ["METRIC_LR"],
        "forgetting_progress_threshold": float(os.environ["METRIC_FORGETTING_RHO"]),
        "window_size": as_int("METRIC_WINDOW_SIZE"),
        "reactivation_enabled": True,
        "reactivation_patience": as_int("METRIC_REACTIVATION_K"),
        "micro_batch_size": as_int("METRIC_MICRO_BATCH"),
        "gradient_accumulation_steps": as_int("METRIC_GRAD_ACC"),
        "effective_batch_size": as_int("METRIC_EFFECTIVE_BATCH"),
        "max_epochs": as_int("METRIC_MAX_EPOCHS"),
        "seed": as_int("METRIC_SEED"),
    },
    "timing": {
        "train_process_wall_time_sec": train_process_sec,
        "trainer_reported_train_runtime_sec": trainer_runtime,
        "evaluation_phase_wall_time_sec": as_float("METRIC_EVALUATION_PHASE_SEC"),
        "eval_process_wall_time_sec_total": eval_process_total,
        "per_checkpoint_eval": eval_timings,
        "postprocess_and_cleanup_wall_time_sec": as_float("METRIC_POSTPROCESS_SEC"),
        "total_combo_wall_time_sec": total_combo_sec,
        "non_training_combo_overhead_sec": max(total_combo_sec - train_process_sec, 0.0),
        "single_gpu_train_process_hours": train_process_sec / 3600.0,
        "measurement_notes": {
            "train_process_wall_time_sec": (
                "Wall time of the complete src/train.py process. It includes Python startup, "
                "model/data loading, trainer training, logging, and checkpoint writes; it excludes TOFU evaluation."
            ),
            "trainer_reported_train_runtime_sec": (
                "Hugging Face Trainer train_runtime when present in Trainer outputs; null otherwise."
            ),
        },
    },
    "compute": {
        "optimizer_steps": int(optimizer_steps),
        "completed_epochs": completed_epochs,
        "configured_max_epochs": as_int("METRIC_MAX_EPOCHS"),
        "final_checkpoint_step": as_int("METRIC_FINAL_CHECKPOINT_STEP"),
        "num_checkpoints_evaluated": as_int("METRIC_NUM_CHECKPOINTS"),
        "cuda_visible_devices": os.environ["METRIC_CUDA_VISIBLE_DEVICES"],
    },
    "trainer_metadata": trainer_metadata,
    "sample_stop_summary": sample_stop_summary,
    "sample_stop_numeric_metrics": flatten_numeric_dict(sample_stop_summary),
}

output_path.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [[ ! -s "${STOP_RESULT_DIR}/efficiency_metrics.json" ]]; then
    echo "[ERROR] efficiency_metrics.json 未生成或为空"
    exit 1
fi

touch "${STOP_RESULT_DIR}/GRID_DONE"

echo "[TIME] postprocess and cleanup wall time: ${POSTPROCESS_WALL_SEC} sec"
echo "[TIME] total combination wall time: ${TOTAL_COMBO_WALL_SEC} sec"
echo "[METRICS] ${STOP_RESULT_DIR}/efficiency_metrics.json"
echo "=================================================="
echo "All checkpoint evaluations completed."
echo "Results: ${EVAL_ROOT}"
echo "=================================================="
