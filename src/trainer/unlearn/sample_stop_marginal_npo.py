import json
import os
from collections import defaultdict, deque
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll


class SampleStopMarginalNPO(NPO):
    """
    NPO with sample-wise stopping based on mean forgetting progress.

    For sample i, let L_i(0) be the first observed per-sample NPO loss and
    L_i(t) its loss at the current observation. Define the normalized residual
    and forgetting progress as:

        r_i(t) = L_i(t) / (L_i(0) + epsilon)
        P_i(t) = 1 - r_i(t)

    Using the latest three progress observations:

        mean_P_i(t) = [P_i(t-2) + P_i(t-1) + P_i(t)] / 3

    An active sample stops if and only if three progress observations are
    available and:

        mean_P_i(t) > forgetting_progress_threshold

    A sample stopped at the current observation still contributes to the
    current update. It is excluded from backward from its next occurrence.

    Stopped samples continue to receive forward-only checks. If the latest
    mean forgetting progress is strictly below the same threshold for
    reactivation_patience consecutive stopped-sample observations, the sample
    is reactivated. The triggering check is forward-only; the sample resumes
    gradient updates from its next occurrence.
    """

    def __init__(
        self,
        beta: float = 0.1,
        forgetting_progress_threshold: float = 0.80,
        window_size: int = 3,
        track_stopped_samples: bool = True,
        reactivation_enabled: bool = True,
        reactivation_patience: int = 3,
        log_interval: int = 50,
        epsilon: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(beta=beta, *args, **kwargs)

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        if not 0.0 <= forgetting_progress_threshold <= 1.0:
            raise ValueError(
                "forgetting_progress_threshold must be in [0, 1], "
                f"got {forgetting_progress_threshold}"
            )

        if window_size != 3:
            raise ValueError(
                "The mean forgetting-progress criterion requires exactly "
                f"three observations; got window_size={window_size}"
            )

        if reactivation_patience <= 0:
            raise ValueError(
                "reactivation_patience must be positive, "
                f"got {reactivation_patience}"
            )

        if reactivation_enabled and not track_stopped_samples:
            raise ValueError(
                "reactivation requires track_stopped_samples=true because "
                "stopped samples must continue receiving forward checks"
            )

        if log_interval < 0:
            raise ValueError(
                "log_interval must be non-negative, "
                f"got {log_interval}"
            )

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.forgetting_progress_threshold = float(
            forgetting_progress_threshold
        )
        self.window_size = 3
        self.track_stopped_samples = bool(track_stopped_samples)
        self.reactivation_enabled = bool(reactivation_enabled)
        self.reactivation_patience = int(reactivation_patience)
        self.sample_stop_log_interval = int(log_interval)
        self.sample_stop_epsilon = float(epsilon)

        # Per-sample state.
        self.sample_state: Dict[int, str] = {}
        self.initial_loss: Dict[int, float] = {}

        # Keep the latest three raw losses and forgetting-progress values.
        self.loss_history = defaultdict(lambda: deque(maxlen=3))
        self.progress_history = defaultdict(lambda: deque(maxlen=3))

        self.latest_loss: Dict[int, float] = {}
        self.latest_normalized_loss_ratio: Dict[int, float] = {}
        self.latest_forgetting_progress: Dict[int, float] = {}
        self.latest_mean_forgetting_progress: Dict[int, float] = {}

        self.observation_count = defaultdict(int)
        # Latest transition locations, plus complete histories because one
        # sample can stop and reactivate multiple times.
        self.stop_step: Dict[int, int] = {}
        self.stop_observation: Dict[int, int] = {}
        self.stop_count = defaultdict(int)
        self.stop_steps_by_sample = defaultdict(list)
        self.stop_observations_by_sample = defaultdict(list)

        self.reactivation_step: Dict[int, int] = {}
        self.reactivation_observation: Dict[int, int] = {}
        self.reactivation_count = defaultdict(int)
        self.reactivation_steps_by_sample = defaultdict(list)
        self.reactivation_observations_by_sample = defaultdict(list)

        # Sample-occurrence statistics.
        self.total_seen_samples = 0
        self.total_active_samples = 0
        self.total_stopped_samples = 0

        # Forward-only stopped-sample diagnostics.
        self.total_stopped_forward_check_occurrences = 0
        self.total_stopped_ready_check_occurrences = 0
        self.total_stopped_violation_observations = 0
        self.total_stopped_stable_observations = 0
        self.total_progress_threshold_violations = 0

        self.samples_with_violation = set()
        self.violation_count = defaultdict(int)
        self.consecutive_violations = defaultdict(int)
        self.max_consecutive_violations = defaultdict(int)
        self.latest_stopped_check_satisfied: Dict[int, bool] = {}
        self.latest_stopped_violation_reason: Dict[int, Optional[str]] = {}

        # Rebound state: count consecutive stopped-sample checks whose mean
        # forgetting progress is strictly below the threshold.
        self.consecutive_below_threshold = defaultdict(int)
        self.max_consecutive_below_threshold = defaultdict(int)
        self.total_reactivation_events = 0
        self.samples_reactivated = set()

        self._last_logged_step = -1

        os.makedirs(self.args.output_dir, exist_ok=True)

        self.sample_log_path = os.path.join(
            self.args.output_dir,
            "sample_stop_log.jsonl",
        )
        self.trajectory_log_path = os.path.join(
            self.args.output_dir,
            "sample_trajectory_log.jsonl",
        )
        self.violation_log_path = os.path.join(
            self.args.output_dir,
            "sample_violation_log.jsonl",
        )
        self.reactivation_log_path = os.path.join(
            self.args.output_dir,
            "sample_reactivation_log.jsonl",
        )
        self.summary_path = os.path.join(
            self.args.output_dir,
            "sample_stop_summary.json",
        )

        # Empty logs are valid outcomes and should still exist.
        for path in (
            self.sample_log_path,
            self.trajectory_log_path,
            self.violation_log_path,
            self.reactivation_log_path,
        ):
            open(path, "a", encoding="utf-8").close()

        print(
            "[SampleStopMarginalNPO] Initialized with "
            f"beta={self.beta}, "
            "criterion=mean_forgetting_progress, "
            "progress_definition=1-L_t/(L_0+epsilon), "
            f"forgetting_progress_threshold="
            f"{self.forgetting_progress_threshold}, "
            "progress_window_size=3, "
            f"track_stopped_samples={self.track_stopped_samples}, "
            f"reactivation_enabled={self.reactivation_enabled}, "
            f"reactivation_patience={self.reactivation_patience}"
        )

    @staticmethod
    def _select_model_inputs(batch):
        required_keys = (
            "input_ids",
            "attention_mask",
            "labels",
        )

        missing = [
            key
            for key in required_keys
            if key not in batch
        ]

        if missing:
            raise KeyError(
                f"Batch is missing required keys: {missing}. "
                f"Available keys: {list(batch.keys())}"
            )

        return {
            key: batch[key]
            for key in required_keys
        }

    @staticmethod
    def _slice_model_inputs(model_inputs, positions):
        index = torch.as_tensor(
            positions,
            dtype=torch.long,
            device=model_inputs["input_ids"].device,
        )

        return {
            key: value.index_select(0, index)
            for key, value in model_inputs.items()
        }

    def _get_positions_by_state(self, sample_ids):
        active_positions = []
        stopped_positions = []

        for position, sample_id in enumerate(sample_ids):
            if sample_id not in self.sample_state:
                self.sample_state[sample_id] = "active"

            state = self.sample_state[sample_id]
            if state == "active":
                active_positions.append(position)
            elif state == "stopped":
                stopped_positions.append(position)
            else:
                raise ValueError(
                    f"Unknown sample state for sample {sample_id}: {state}"
                )

        return active_positions, stopped_positions

    def _compute_npo_loss_per_sample(self, model, forget_inputs):
        """Compute the original NPO objective independently per sample."""
        current_nll, current_outputs = compute_batch_nll(
            model,
            forget_inputs,
        )

        with torch.no_grad():
            reference_nll, _ = compute_batch_nll(
                self.ref_model,
                forget_inputs,
            )

        lose_log_ratio = -(current_nll - reference_nll)

        per_sample_loss = (
            -2.0
            / self.beta
            * F.logsigmoid(-self.beta * lose_log_ratio)
        )

        return per_sample_loss, current_outputs

    @staticmethod
    def _get_sample_ids(forget_batch):
        if "index" not in forget_batch:
            raise KeyError(
                "SampleStopMarginalNPO requires "
                "inputs['forget']['index'], but the forget batch "
                "contains no index field."
            )

        sample_ids = forget_batch["index"]

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = (
                sample_ids
                .detach()
                .cpu()
                .view(-1)
                .tolist()
            )
        else:
            sample_ids = list(sample_ids)

        return [int(sample_id) for sample_id in sample_ids]

    @staticmethod
    def _append_jsonl(path, records):
        if not records:
            return

        with open(path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    def _observe_sample(
        self,
        sample_id: int,
        current_loss: float,
        phase: str,
    ):
        """
        Update one sample trajectory.

        phase="active": the sample may transition to stopped.
        phase="stopped_check": forward-only check; the sample may reactivate.
        """
        if phase not in {"active", "stopped_check"}:
            raise ValueError(f"Unknown observation phase: {phase}")

        current_step = int(self.state.global_step)
        current_loss = float(current_loss)

        if sample_id not in self.sample_state:
            self.sample_state[sample_id] = "active"

        expected_state = "active" if phase == "active" else "stopped"
        if self.sample_state[sample_id] != expected_state:
            raise RuntimeError(
                f"Sample {sample_id} has state {self.sample_state[sample_id]} "
                f"but was observed in phase {phase}."
            )

        self.observation_count[sample_id] += 1
        observation = self.observation_count[sample_id]
        self.latest_loss[sample_id] = current_loss

        if sample_id not in self.initial_loss:
            self.initial_loss[sample_id] = current_loss

        initial_loss = self.initial_loss[sample_id]
        denominator = initial_loss + self.sample_stop_epsilon

        if denominator <= 0.0:
            raise RuntimeError(
                f"Sample {sample_id} has invalid initial NPO loss "
                f"{initial_loss}; L_i(0) + epsilon must be positive."
            )

        normalized_loss_ratio = current_loss / denominator
        forgetting_progress = 1.0 - normalized_loss_ratio

        self.latest_normalized_loss_ratio[sample_id] = float(
            normalized_loss_ratio
        )
        self.latest_forgetting_progress[sample_id] = float(
            forgetting_progress
        )

        loss_hist = self.loss_history[sample_id]
        progress_hist = self.progress_history[sample_id]
        loss_hist.append(current_loss)
        progress_hist.append(forgetting_progress)

        mean_forgetting_progress = None
        if len(progress_hist) == self.window_size:
            mean_forgetting_progress = sum(progress_hist) / self.window_size
            self.latest_mean_forgetting_progress[sample_id] = float(
                mean_forgetting_progress
            )

        loss_window = list(loss_hist)
        progress_window = list(progress_hist)
        progress_window_ready = (
            len(progress_window) == self.window_size
            and mean_forgetting_progress is not None
        )

        criterion_satisfied = (
            progress_window_ready
            and mean_forgetting_progress
            > self.forgetting_progress_threshold
        )

        stopped_now = phase == "active" and criterion_satisfied

        if stopped_now:
            self.sample_state[sample_id] = "stopped"
            self.stop_step[sample_id] = current_step
            self.stop_observation[sample_id] = observation
            self.stop_count[sample_id] += 1
            self.stop_steps_by_sample[sample_id].append(current_step)
            self.stop_observations_by_sample[sample_id].append(observation)
            self.consecutive_violations[sample_id] = 0
            self.consecutive_below_threshold[sample_id] = 0

        violation = False
        violation_reason = None
        below_threshold = False
        rebound_streak = 0
        reactivated_now = False

        if phase == "stopped_check":
            self.total_stopped_forward_check_occurrences += 1

            if progress_window_ready:
                self.total_stopped_ready_check_occurrences += 1
                self.latest_stopped_check_satisfied[sample_id] = bool(
                    criterion_satisfied
                )

                below_threshold = (
                    mean_forgetting_progress
                    < self.forgetting_progress_threshold
                )

                if criterion_satisfied:
                    self.total_stopped_stable_observations += 1
                    self.consecutive_violations[sample_id] = 0
                    self.consecutive_below_threshold[sample_id] = 0
                    self.latest_stopped_violation_reason[sample_id] = None
                else:
                    violation = True
                    violation_reason = (
                        "mean_forgetting_progress_below_threshold"
                        if below_threshold
                        else "mean_forgetting_progress_equal_to_threshold"
                    )
                    self.total_stopped_violation_observations += 1
                    self.total_progress_threshold_violations += 1
                    self.samples_with_violation.add(sample_id)
                    self.violation_count[sample_id] += 1
                    self.consecutive_violations[sample_id] += 1
                    self.max_consecutive_violations[sample_id] = max(
                        self.max_consecutive_violations[sample_id],
                        self.consecutive_violations[sample_id],
                    )
                    self.latest_stopped_violation_reason[
                        sample_id
                    ] = violation_reason

                    if below_threshold:
                        self.consecutive_below_threshold[sample_id] += 1
                    else:
                        # Equality does not satisfy the strict rebound rule.
                        self.consecutive_below_threshold[sample_id] = 0

                    self.max_consecutive_below_threshold[sample_id] = max(
                        self.max_consecutive_below_threshold[sample_id],
                        self.consecutive_below_threshold[sample_id],
                    )

                rebound_streak = self.consecutive_below_threshold[sample_id]

                if (
                    self.reactivation_enabled
                    and below_threshold
                    and rebound_streak >= self.reactivation_patience
                ):
                    self.sample_state[sample_id] = "active"
                    reactivated_now = True
                    self.total_reactivation_events += 1
                    self.samples_reactivated.add(sample_id)
                    self.reactivation_count[sample_id] += 1
                    self.reactivation_step[sample_id] = current_step
                    self.reactivation_observation[sample_id] = observation
                    self.reactivation_steps_by_sample[sample_id].append(
                        current_step
                    )
                    self.reactivation_observations_by_sample[sample_id].append(
                        observation
                    )
                    # Preserve rebound_streak in the record, then reset the
                    # counter for the next stopped period.
                    self.consecutive_below_threshold[sample_id] = 0

        record = {
            "event": (
                "reactivated"
                if reactivated_now
                else (
                    "stopped"
                    if stopped_now
                    else (
                        "stopped_sample_violation"
                        if violation
                        else (
                            "stopped_sample_check"
                            if phase == "stopped_check"
                            else "active_observation"
                        )
                    )
                )
            ),
            "phase": phase,
            "sample_id": sample_id,
            "global_step": current_step,
            "observation": observation,
            "initial_loss": initial_loss,
            "loss": current_loss,
            "normalized_loss_ratio": normalized_loss_ratio,
            "forgetting_progress": forgetting_progress,
            "loss_window": loss_window,
            "progress_window": progress_window,
            "window_size": self.window_size,
            "progress_window_ready": progress_window_ready,
            "mean_forgetting_progress": mean_forgetting_progress,
            "forgetting_progress_threshold": (
                self.forgetting_progress_threshold
            ),
            "criterion_satisfied": criterion_satisfied,
            "below_threshold": below_threshold,
            "reactivation_patience": self.reactivation_patience,
            "rebound_streak": rebound_streak,
            "state": self.sample_state[sample_id],
            "stopped_now": stopped_now,
            "violation": violation,
            "violation_reason": violation_reason,
            "consecutive_violations": (
                self.consecutive_violations[sample_id]
                if phase == "stopped_check"
                else 0
            ),
            "reactivated": reactivated_now,
        }

        return record

    def _update_active_sample_states(self, sample_ids, sample_losses):
        detached_losses = (
            sample_losses
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        trajectory_records = []
        newly_stopped = []

        for sample_id, current_loss in zip(sample_ids, detached_losses):
            record = self._observe_sample(
                sample_id=sample_id,
                current_loss=current_loss,
                phase="active",
            )
            trajectory_records.append(record)

            if record["stopped_now"]:
                newly_stopped.append(record)

        self._append_jsonl(
            self.trajectory_log_path,
            trajectory_records,
        )
        self._append_jsonl(
            self.sample_log_path,
            newly_stopped,
        )

    def _update_stopped_sample_checks(self, sample_ids, sample_losses):
        detached_losses = (
            sample_losses
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        trajectory_records = []
        violation_records = []
        reactivation_records = []

        for sample_id, current_loss in zip(sample_ids, detached_losses):
            record = self._observe_sample(
                sample_id=sample_id,
                current_loss=current_loss,
                phase="stopped_check",
            )
            trajectory_records.append(record)

            if record["violation"]:
                violation_records.append(record)
            if record["reactivated"]:
                reactivation_records.append(record)

        self._append_jsonl(
            self.trajectory_log_path,
            trajectory_records,
        )
        self._append_jsonl(
            self.violation_log_path,
            violation_records,
        )
        self._append_jsonl(
            self.reactivation_log_path,
            reactivation_records,
        )

    @staticmethod
    def _mean_or_none(values):
        return sum(values) / len(values) if values else None

    def _write_summary(self):
        seen = len(self.sample_state)

        stopped = sum(
            state == "stopped"
            for state in self.sample_state.values()
        )
        active = seen - stopped

        stopped_ids = {
            sample_id
            for sample_id, state in self.sample_state.items()
            if state == "stopped"
        }

        stopped_initial_losses = [
            self.initial_loss[sample_id]
            for sample_id in stopped_ids
            if sample_id in self.initial_loss
        ]
        stopped_losses = [
            self.latest_loss[sample_id]
            for sample_id in stopped_ids
            if sample_id in self.latest_loss
        ]
        stopped_normalized_ratios = [
            self.latest_normalized_loss_ratio[sample_id]
            for sample_id in stopped_ids
            if sample_id in self.latest_normalized_loss_ratio
        ]
        stopped_progresses = [
            self.latest_forgetting_progress[sample_id]
            for sample_id in stopped_ids
            if sample_id in self.latest_forgetting_progress
        ]
        stopped_mean_progresses = [
            self.latest_mean_forgetting_progress[sample_id]
            for sample_id in stopped_ids
            if sample_id in self.latest_mean_forgetting_progress
        ]

        violated_stopped_ids = stopped_ids & self.samples_with_violation

        final_evaluable_stopped_ids = {
            sample_id
            for sample_id in stopped_ids
            if sample_id in self.latest_stopped_check_satisfied
        }
        final_satisfied_stopped_ids = {
            sample_id
            for sample_id in final_evaluable_stopped_ids
            if self.latest_stopped_check_satisfied[sample_id]
        }

        ever_stopped_ids = set(self.stop_count)
        reactivated_ids = set(self.samples_reactivated)

        max_consecutive = max(
            self.max_consecutive_violations.values(),
            default=0,
        )
        max_rebound_streak = max(
            self.max_consecutive_below_threshold.values(),
            default=0,
        )

        summary = {
            "global_step": int(self.state.global_step),
            "criterion": "mean_forgetting_progress",
            "seen_unique_samples": seen,
            "active_unique_samples": active,
            "stopped_unique_samples": stopped,
            "ever_stopped_unique_samples": len(ever_stopped_ids),
            "reactivated_unique_samples": len(reactivated_ids),
            "total_stop_events": sum(self.stop_count.values()),
            "total_reactivation_events": self.total_reactivation_events,
            "active_ratio": active / seen if seen > 0 else 1.0,
            "mean_stopped_initial_loss": self._mean_or_none(
                stopped_initial_losses
            ),
            "mean_stopped_latest_loss": self._mean_or_none(
                stopped_losses
            ),
            "mean_stopped_normalized_loss_ratio": self._mean_or_none(
                stopped_normalized_ratios
            ),
            "mean_stopped_latest_forgetting_progress": self._mean_or_none(
                stopped_progresses
            ),
            "mean_stopped_window_forgetting_progress": self._mean_or_none(
                stopped_mean_progresses
            ),
            "total_batch_sample_occurrences": self.total_seen_samples,
            "total_active_sample_occurrences": self.total_active_samples,
            "total_stopped_sample_occurrences": self.total_stopped_samples,
            "effective_update_saving": (
                1.0
                - (
                    self.total_active_samples
                    / self.total_seen_samples
                )
                if self.total_seen_samples > 0
                else 0.0
            ),
            "stopped_sample_diagnostics": {
                "enabled": self.track_stopped_samples,
                "forward_check_occurrences": (
                    self.total_stopped_forward_check_occurrences
                ),
                "ready_check_occurrences": (
                    self.total_stopped_ready_check_occurrences
                ),
                "criterion_satisfied_observations": (
                    self.total_stopped_stable_observations
                ),
                "violation_observations": (
                    self.total_stopped_violation_observations
                ),
                "observation_violation_rate": (
                    self.total_stopped_violation_observations
                    / self.total_stopped_ready_check_occurrences
                    if self.total_stopped_ready_check_occurrences > 0
                    else 0.0
                ),
                "samples_with_violation": len(violated_stopped_ids),
                "sample_violation_rate": (
                    len(violated_stopped_ids) / stopped
                    if stopped > 0
                    else 0.0
                ),
                "progress_threshold_violation_observations": (
                    self.total_progress_threshold_violations
                ),
                "final_evaluable_stopped_samples": len(
                    final_evaluable_stopped_ids
                ),
                "final_satisfied_stopped_samples": len(
                    final_satisfied_stopped_ids
                ),
                "final_satisfied_rate_among_evaluable": (
                    len(final_satisfied_stopped_ids)
                    / len(final_evaluable_stopped_ids)
                    if final_evaluable_stopped_ids
                    else None
                ),
                "final_satisfied_rate_among_all_stopped": (
                    len(final_satisfied_stopped_ids) / stopped
                    if stopped > 0
                    else None
                ),
                "max_consecutive_violations": max_consecutive,
                "max_consecutive_below_threshold": max_rebound_streak,
                "current_rebound_streak_by_sample": dict(
                    self.consecutive_below_threshold
                ),
                "max_rebound_streak_by_sample": dict(
                    self.max_consecutive_below_threshold
                ),
                "violation_count_by_sample": dict(
                    self.violation_count
                ),
                "max_consecutive_violations_by_sample": dict(
                    self.max_consecutive_violations
                ),
                "latest_check_satisfied_by_sample": dict(
                    self.latest_stopped_check_satisfied
                ),
                "latest_violation_reason_by_sample": dict(
                    self.latest_stopped_violation_reason
                ),
            },
            "initial_losses": self.initial_loss,
            "stop_steps": self.stop_step,
            "stop_observations": self.stop_observation,
            "stop_count_by_sample": dict(self.stop_count),
            "stop_steps_by_sample": dict(self.stop_steps_by_sample),
            "stop_observations_by_sample": dict(
                self.stop_observations_by_sample
            ),
            "reactivation_steps": self.reactivation_step,
            "reactivation_observations": self.reactivation_observation,
            "reactivation_count_by_sample": dict(self.reactivation_count),
            "reactivation_steps_by_sample": dict(
                self.reactivation_steps_by_sample
            ),
            "reactivation_observations_by_sample": dict(
                self.reactivation_observations_by_sample
            ),
            "config": {
                "beta": self.beta,
                "criterion": "mean_forgetting_progress",
                "forgetting_progress_threshold": (
                    self.forgetting_progress_threshold
                ),
                "window_size": self.window_size,
                "window_semantics": (
                    "latest three per-sample forgetting-progress observations"
                ),
                "normalized_loss_ratio_formula": (
                    "r_i(t) = L_i(t) / (L_i(0) + epsilon)"
                ),
                "forgetting_progress_formula": (
                    "P_i(t) = 1 - L_i(t) / (L_i(0) + epsilon)"
                ),
                "mean_progress_formula": (
                    "mean_P_i(t) = (P_i(t-2)+P_i(t-1)+P_i(t))/3"
                ),
                "equivalent_mean_residual_condition": (
                    "mean_r_i(t) < 1 - forgetting_progress_threshold"
                ),
                "stop_condition": (
                    "mean_forgetting_progress "
                    "> forgetting_progress_threshold"
                ),
                "track_stopped_samples": self.track_stopped_samples,
                "reactivation_enabled": self.reactivation_enabled,
                "reactivation_patience": self.reactivation_patience,
                "reactivation_condition": (
                    "mean_forgetting_progress < "
                    "forgetting_progress_threshold for "
                    "reactivation_patience consecutive stopped checks"
                ),
                "reactivation_timing": (
                    "triggering check is forward-only; gradient updates "
                    "resume from the next sample occurrence"
                ),
                "progress_clipping_enabled": False,
                "epsilon": self.sample_stop_epsilon,
            },
        }

        with open(self.summary_path, "w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return summary

    def _maybe_log_statistics(self):
        current_step = int(self.state.global_step)

        should_log = (
            self.sample_stop_log_interval > 0
            and current_step % self.sample_stop_log_interval == 0
            and current_step != self._last_logged_step
        )

        if not should_log:
            return

        self._last_logged_step = current_step
        summary = self._write_summary()
        diagnostics = summary["stopped_sample_diagnostics"]

        mean_progress = summary[
            "mean_stopped_window_forgetting_progress"
        ]

        self.log(
            {
                "sample_stop/active_unique": (
                    summary["active_unique_samples"]
                ),
                "sample_stop/stopped_unique": (
                    summary["stopped_unique_samples"]
                ),
                "sample_stop/active_ratio": summary["active_ratio"],
                "sample_stop/mean_forgetting_progress": (
                    mean_progress if mean_progress is not None else 0.0
                ),
                "sample_stop/update_saving": (
                    summary["effective_update_saving"]
                ),
                "sample_stop/violation_observation_rate": (
                    diagnostics["observation_violation_rate"]
                ),
                "sample_stop/sample_violation_rate": (
                    diagnostics["sample_violation_rate"]
                ),
                "sample_stop/total_reactivations": (
                    summary["total_reactivation_events"]
                ),
            }
        )

        print(
            "[SampleStopMarginalNPO] "
            f"step={current_step}, "
            f"seen={summary['seen_unique_samples']}, "
            f"active={summary['active_unique_samples']}, "
            f"stopped={summary['stopped_unique_samples']}, "
            f"active_ratio={summary['active_ratio']:.4f}, "
            "mean_stopped_progress="
            f"{mean_progress if mean_progress is not None else 0.0:.4f}, "
            "effective_update_saving="
            f"{summary['effective_update_saving']:.4f}, "
            "stopped_checks="
            f"{diagnostics['forward_check_occurrences']}, "
            "violation_observations="
            f"{diagnostics['violation_observations']}, "
            "reactivations="
            f"{summary['total_reactivation_events']}"
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_batch = inputs["forget"]

        sample_ids = self._get_sample_ids(forget_batch)
        forget_inputs = self._select_model_inputs(forget_batch)

        active_positions, stopped_positions = (
            self._get_positions_by_state(sample_ids)
        )

        active_sample_ids = [
            sample_ids[position]
            for position in active_positions
        ]
        stopped_sample_ids = [
            sample_ids[position]
            for position in stopped_positions
        ]

        batch_size = len(sample_ids)
        active_count = len(active_positions)
        stopped_count = len(stopped_positions)

        self.total_seen_samples += batch_size
        self.total_active_samples += active_count
        self.total_stopped_samples += stopped_count

        forget_loss = None
        forget_outputs = (None, None)

        # Active samples: forward with gradients and participate in backward.
        if active_count > 0:
            active_forget_inputs = self._slice_model_inputs(
                forget_inputs,
                active_positions,
            )

            per_sample_forget_loss, current_outputs = (
                self._compute_npo_loss_per_sample(
                    model=model,
                    forget_inputs=active_forget_inputs,
                )
            )

            # A sample stopped here still contributes to this current update.
            self._update_active_sample_states(
                sample_ids=active_sample_ids,
                sample_losses=per_sample_forget_loss,
            )

            forget_loss = per_sample_forget_loss.mean()
            forget_outputs = (None, current_outputs)

        # Stopped samples: forward-only rebound checks, never backward in
        # this occurrence. A reactivated sample resumes backward from its next
        # occurrence.
        if self.track_stopped_samples and stopped_count > 0:
            stopped_forget_inputs = self._slice_model_inputs(
                forget_inputs,
                stopped_positions,
            )

            with torch.no_grad():
                stopped_per_sample_loss, _ = (
                    self._compute_npo_loss_per_sample(
                        model=model,
                        forget_inputs=stopped_forget_inputs,
                    )
                )

            self._update_stopped_sample_checks(
                sample_ids=stopped_sample_ids,
                sample_losses=stopped_per_sample_loss,
            )

        retain_batch = inputs["retain"]
        retain_inputs = self._select_model_inputs(retain_batch)

        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs,
        )

        if forget_loss is None:
            # Differentiable zero when all forget samples have stopped.
            forget_loss = retain_loss * 0.0

        self.log(
            {
                "forget_loss": (
                    forget_loss.detach().float().item()
                ),
                "retain_loss": (
                    retain_loss.detach().float().item()
                ),
                "sample_stop/current_active_samples": active_count,
                "sample_stop/current_stopped_samples": stopped_count,
            }
        )

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_loss
        )

        self._maybe_log_statistics()

        if return_outputs:
            return loss, forget_outputs

        return loss

    def save_model(
        self,
        output_dir=None,
        _internal_call=False,
    ):
        self._write_summary()

        return super().save_model(
            output_dir=output_dir,
            _internal_call=_internal_call,
        )
