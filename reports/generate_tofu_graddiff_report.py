#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path("/unlearning/experment_data/checkpoints/tofu")
OUT = Path("reports/tofu_graddiff_results_20260818")

FIELDS = [
    "split", "num_forget_samples", "model", "method", "trainer",
    "controller_family", "monitoring_signal", "run_name", "is_baseline",
    "is_old_copy", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "effective_batch_size_single_gpu",
    "epochs_configured", "beta", "alpha", "gamma", "retain_loss_type",
    "sampling_policy", "retain_policy_after_stop", "forget_loss_denominator",
    "rebound_semantics", "moving_average_window", "criterion_threshold",
    "stop_ratio_threshold", "absolute_nll_gain_threshold", "warm_up_epochs",
    "stop_patience", "rebound_ratio_threshold", "rebound_delta",
    "reactivation_patience", "forget_quality", "model_utility",
    "forget_truth_ratio", "exact_memorization", "extraction_strength",
    "forget_Q_A_Prob", "forget_Q_A_ROUGE", "privleak", "final_active_samples",
    "final_stopped_samples", "stop_events", "reactivation_events",
    "saved_forget_instances", "saved_forget_instances_pct_of_10epoch_budget",
    "trajectory_final_epoch", "trainer_completed_epoch", "global_step",
    "train_runtime_seconds", "train_runtime_minutes", "summary_json",
    "trajectory_file", "trainer_state_json", "run_directory",
]


def rx(s, pattern, cast=str):
    m = re.search(pattern, s, re.I)
    if not m:
        return ""
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return m.group(1)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def last_jsonl(path):
    if not path:
        return {}
    last = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        return json.loads(last) if last else {}
    except (OSError, json.JSONDecodeError):
        return {}


def first_existing(run_dir, patterns):
    for pattern in patterns:
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def parse_summary(summary_path):
    run_dir = summary_path.parent.parent
    rel = summary_path.relative_to(ROOT).parts
    split, model, trainer = rel[0], rel[1], rel[2]
    run = rel[3]
    low = (trainer + "/" + run).lower()
    if "graddiff" not in low and "grad_diff" not in low:
        return None

    if "marginalratio" in low or "marginal_ratio" in low:
        family = "marginal_ratio"
        signal = "per_sample_length_normalized_nll_epoch_gain_ratio"
    elif "d2" in low or "second" in low:
        family = "second_difference"
        signal = "per_sample_length_normalized_nll_second_difference"
    elif "sampleearlystop" in low or "normnll_gain" in low:
        family = "absolute_nll_gain"
        signal = "per_sample_length_normalized_nll_gain_from_initial"
    else:
        family = "baseline"
        signal = "none"

    traj = first_existing(run_dir, ["*history.jsonl", "sample_normalized_nll.jsonl"])
    tail = last_jsonl(traj)
    state_path = first_existing(run_dir, ["*state.json"])
    state = load_json(state_path) if state_path else {}
    trainer_path = run_dir / "trainer_state.json"
    trainer_state = load_json(trainer_path)
    metrics = load_json(summary_path)

    def value(*keys):
        for source in (tail, state):
            for key in keys:
                if key in source:
                    return source[key]
        return ""

    active = value("active", "num_active_samples")
    stopped = value("stopped", "num_stopped_samples")
    if isinstance(active, list):
        active = len(active)
    if isinstance(stopped, list):
        stopped = len(stopped)
    transitions = state.get("transition_history", [])
    stop_events = sum(1 for x in transitions if x.get("event") in ("stop", "stopped"))
    react_events = sum(1 for x in transitions if x.get("event") in ("reactivate", "reactivated"))
    if not stop_events:
        stop_events = value("stop_events", "total_stop_events")
    if not react_events:
        react_events = value("reactivation_events", "total_reactivation_events")
    saved = value("cumulative_saved_forget_instances", "saved_forget_instances")
    num_forget = {"forget01": 40, "forget05": 200, "forget10": 400}.get(split, "")
    saved_pct = ""
    if isinstance(saved, (int, float)) and num_forget:
        saved_pct = 100.0 * saved / (num_forget * 10)

    lr = rx(run, r"(?:^|_)lr([^_]+)")
    batch = rx(run, r"(?:^|_)b(\d+)", int)
    ga = rx(run, r"(?:^|_)ga(\d+)", int)
    epochs = rx(run, r"(?:^|_)e(\d+)", int)
    alpha = rx(run, r"(?:^|_)alpha([0-9.]+)", float)
    threshold = rx(run, r"(?:ge|tau)([0-9.]+)", float)
    stop_ratio = rx(run, r"(?:stop)([0-9.]+)", float) if family == "marginal_ratio" else ""
    ma = rx(run, r"(?:^|_)ma(\d+)", int)
    warm = rx(run, r"warm([0-9.]+)", float)
    patience = rx(run, r"(?<!r)patience(\d+)", int)
    rebound_delta = rx(run, r"rebounddelta([0-9.]+)", float)
    rebound_ratio = rx(run, r"rebound([0-9.]+)", float) if family == "marginal_ratio" else ""
    rpat = rx(run, r"rpat(\d+)", int)

    if family == "baseline":
        sampling = "baseline"
        retain_policy = "baseline_full_retain"
        denominator = "baseline_mean"
        rebound_semantics = "none"
    elif "samplingbaseline_masked" in low or "fullretain" in low:
        sampling = "baseline_forget_anchor_masked"
        retain_policy = "full_retain"
        denominator = "fixed_original_valid_tokens"
        rebound_semantics = ("marginal_recovery_vs_historical_peak" if family == "marginal_ratio"
                             else "absolute_gain_below_threshold_minus_delta")
    else:
        sampling = "legacy_or_unspecified"
        retain_policy = "legacy_or_unspecified"
        denominator = "legacy_or_unspecified"
        rebound_semantics = "legacy_or_unspecified"

    runtime = ""
    for item in reversed(trainer_state.get("log_history", [])):
        if "train_runtime" in item:
            runtime = item["train_runtime"]
            break

    row = {key: "" for key in FIELDS}
    row.update({
        "split": split, "num_forget_samples": num_forget, "model": model,
        "method": "GradDiff", "trainer": trainer, "controller_family": family,
        "monitoring_signal": tail.get("monitoring_signal", signal), "run_name": run,
        "is_baseline": family == "baseline", "is_old_copy": "old" in low,
        "learning_rate": lr, "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": ga,
        "effective_batch_size_single_gpu": batch * ga if batch and ga else "",
        "epochs_configured": epochs, "alpha": alpha, "retain_loss_type": "NLL",
        "sampling_policy": sampling, "retain_policy_after_stop": retain_policy,
        "forget_loss_denominator": denominator, "rebound_semantics": rebound_semantics,
        "moving_average_window": value("moving_average_window") or ma,
        "criterion_threshold": threshold or stop_ratio,
        "stop_ratio_threshold": value("stop_ratio_threshold") or stop_ratio,
        "absolute_nll_gain_threshold": threshold if family == "absolute_nll_gain" else "",
        "warm_up_epochs": warm, "stop_patience": value("stop_patience") or patience,
        "rebound_ratio_threshold": value("rebound_ratio_threshold") or rebound_ratio,
        "rebound_delta": rebound_delta, "reactivation_patience": value("reactivation_patience") or rpat,
        "final_active_samples": active, "final_stopped_samples": stopped,
        "stop_events": stop_events, "reactivation_events": react_events,
        "saved_forget_instances": saved,
        "saved_forget_instances_pct_of_10epoch_budget": saved_pct,
        "trajectory_final_epoch": value("epoch"),
        "trainer_completed_epoch": trainer_state.get("epoch", ""),
        "global_step": trainer_state.get("global_step", ""),
        "train_runtime_seconds": runtime,
        "train_runtime_minutes": runtime / 60 if isinstance(runtime, (int, float)) else "",
        "summary_json": str(summary_path), "trajectory_file": str(traj) if traj else "",
        "trainer_state_json": str(trainer_path) if trainer_path.exists() else "",
        "run_directory": str(run_dir),
    })
    for key in ("forget_quality", "model_utility", "forget_truth_ratio", "exact_memorization",
                "extraction_strength", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "privleak"):
        row[key] = metrics.get(key, "")
    return row


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in ROOT.glob("**/eval/TOFU_SUMMARY.json"):
        row = parse_summary(summary)
        if row:
            rows.append(row)
    rows.sort(key=lambda x: (x["split"], x["model"], x["controller_family"], x["run_name"]))
    write_csv(OUT / "tofu_graddiff_all_results.csv", rows)
    files = ["tofu_graddiff_all_results.csv"]
    for split in ("forget01", "forget05", "forget10"):
        name = f"tofu_graddiff_{split}.csv"
        write_csv(OUT / name, [r for r in rows if r["split"] == split])
        files.append(name)
    counts = Counter(f'{r["split"]}|{r["controller_family"]}' for r in rows)
    manifest = {
        "generated_at": str(date.today()), "checkpoint_root": str(ROOT),
        "completed_rows": len(rows), "counts": dict(sorted(counts.items())), "files": files,
        "note": "All completed TOFU GradDiff runs with eval/TOFU_SUMMARY.json; controller families remain separated. Missing trajectory/runtime fields mean those artifacts were cleaned or were not produced.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
