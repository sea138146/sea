import json
import os
from collections import defaultdict, deque
from typing import Dict

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll


class SampleStopNPO(NPO):
    """
    NPO with sample-wise stopping based on:

        1. sufficient forgetting progress;
        2. small marginal forgetting gain;
        3. stable marginal gain measured by a second-order difference.

    State transition:
        active -> stopped

    Stopped samples no longer contribute to the forget loss from their next
    occurrence onward. The original NPO objective and retain objective remain
    unchanged.
    """

    def __init__(
        self,
        beta: float = 0.1,
        progress_threshold: float = 0.70,
        gain_threshold: float = 0.03,
        curvature_threshold: float = 0.02,
        use_progress_condition: bool = True,
        window_size: int = 4,
        patience: int = 2,
        warmup_observations: int = 4,
        log_interval: int = 50,
        epsilon: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(beta=beta, *args, **kwargs)

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if not 0.0 < progress_threshold < 1.0:
            raise ValueError(
                "progress_threshold must be in (0, 1), "
                f"got {progress_threshold}"
            )
        if gain_threshold < 0:
            raise ValueError(
                f"gain_threshold must be non-negative, got {gain_threshold}"
            )
        if curvature_threshold < 0:
            raise ValueError(
                "curvature_threshold must be non-negative, "
                f"got {curvature_threshold}"
            )
        if window_size < 3:
            raise ValueError(
                f"window_size must be at least 3, got {window_size}"
            )
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if warmup_observations < 3:
            raise ValueError(
                "warmup_observations must be at least 3, "
                f"got {warmup_observations}"
            )

        if log_interval < 0:
            raise ValueError(
                "log_interval must be non-negative, "
                f"got {log_interval}"
            )

        if epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive, got {epsilon}"
            )

        self.progress_threshold = float(progress_threshold)
        self.gain_threshold = float(gain_threshold)
        self.curvature_threshold = float(curvature_threshold)
        self.use_progress_condition = bool(use_progress_condition)

        self.window_size = int(window_size)
        self.stop_patience = int(patience)
        self.warmup_observations = int(warmup_observations)

        self.sample_stop_log_interval = int(log_interval)
        self.sample_stop_epsilon = float(epsilon)

        # Sample states.
        self.sample_state: Dict[int, str] = {}
        self.initial_loss: Dict[int, float] = {}

        # Each history is indexed by the number of times that sample has been
        # observed, rather than by the global optimization step.
        self.progress_history = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.gain_history = defaultdict(
            lambda: deque(maxlen=self.window_size - 1)
        )
        self.curvature_history = defaultdict(
            lambda: deque(maxlen=max(1, self.window_size - 2))
        )

        self.observation_count = defaultdict(int)
        self.stable_counter = defaultdict(int)
        self.stop_step: Dict[int, int] = {}
        self.stop_observation: Dict[int, int] = {}

        # Sample-occurrence statistics.
        self.total_seen_samples = 0
        self.total_active_samples = 0
        self.total_stopped_samples = 0
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
        self.summary_path = os.path.join(
            self.args.output_dir,
            "sample_stop_summary.json",
        )

        print(
            "[SampleStopNPO] Initialized with "
            f"beta={self.beta}, "
            f"progress_threshold={self.progress_threshold}, "
            f"gain_threshold={self.gain_threshold}, "
            f"curvature_threshold={self.curvature_threshold}, "
            f"window_size={self.window_size}, "
            f"patience={self.stop_patience}, "
            f"warmup_observations={self.warmup_observations}"
        )

    @staticmethod
    def _select_model_inputs(batch):
        required_keys = ("input_ids", "attention_mask", "labels")
        missing = [key for key in required_keys if key not in batch]

        if missing:
            raise KeyError(
                f"Batch is missing required keys: {missing}. "
                f"Available keys: {list(batch.keys())}"
            )

        return {key: batch[key] for key in required_keys}

    @staticmethod
    def _slice_model_inputs(model_inputs, positions):
        """Select active samples before the forget forward pass."""
        index = torch.as_tensor(
            positions,
            dtype=torch.long,
            device=model_inputs["input_ids"].device,
        )

        return {
            key: value.index_select(0, index)
            for key, value in model_inputs.items()
        }

    def _get_active_positions(self, sample_ids):
        """Return positions whose samples have not previously stopped."""
        active_positions = []

        for position, sample_id in enumerate(sample_ids):
            if sample_id not in self.sample_state:
                self.sample_state[sample_id] = "active"

            if self.sample_state[sample_id] == "active":
                active_positions.append(position)

        return active_positions

    def _compute_npo_loss_per_sample(self, model, forget_inputs):
        """
        Compute the original NPO objective per sample.

        lose_log_ratio_i = -(NLL_theta_i - NLL_ref_i)

        loss_i =
            -2 / beta * log_sigmoid(-beta * lose_log_ratio_i)
        """
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
                "SampleStopNPO requires inputs['forget']['index'], "
                "but the forget batch contains no index field."
            )

        sample_ids = forget_batch["index"]

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.detach().cpu().view(-1).tolist()
        else:
            sample_ids = list(sample_ids)

        return [int(sample_id) for sample_id in sample_ids]

    def _append_jsonl(self, path, records):
        if not records:
            return

        with open(path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    def _update_sample_states(self, sample_ids, sample_losses):
        """
        Update per-sample trajectories and return the active mask.

        Definitions:

            progress_t = 1 - loss_t / initial_loss

            gain_t = progress_t - progress_{t-1}

            curvature_t = gain_t - gain_{t-1}

        Stop when, for `patience` consecutive observations:

            progress_t >= progress_threshold
            0 <= gain_t <= gain_threshold
            |curvature_t| <= curvature_threshold

        A negative gain means the sample's forgetting state has regressed, so
        it cannot trigger stopping.
        """
        current_step = int(self.state.global_step)
        detached_losses = sample_losses.detach().float().cpu().tolist()

        active_mask = []
        newly_stopped = []
        trajectory_records = []

        for sample_id, current_loss in zip(sample_ids, detached_losses):
            if sample_id not in self.sample_state:
                self.sample_state[sample_id] = "active"

            was_active = self.sample_state[sample_id] == "active"
            active_mask.append(was_active)

            if not was_active:
                continue

            current_loss = float(current_loss)
            self.observation_count[sample_id] += 1
            observation = self.observation_count[sample_id]

            if sample_id not in self.initial_loss:
                self.initial_loss[sample_id] = max(
                    current_loss,
                    self.sample_stop_epsilon,
                )

            initial_loss = self.initial_loss[sample_id]

            raw_progress = 1.0 - (
                current_loss
                / (initial_loss + self.sample_stop_epsilon)
            )

            # Do not allow numerical overshoot to make the stopping score
            # arbitrarily large, but preserve negative values for regressions.
            progress = min(raw_progress, 1.0)

            progress_hist = self.progress_history[sample_id]
            previous_progress = (
                progress_hist[-1] if len(progress_hist) >= 1 else None
            )
            progress_hist.append(progress)

            gain = None
            curvature = None

            if previous_progress is not None:
                gain = progress - previous_progress
                gain_hist = self.gain_history[sample_id]

                previous_gain = (
                    gain_hist[-1] if len(gain_hist) >= 1 else None
                )
                gain_hist.append(gain)

                if previous_gain is not None:
                    curvature = gain - previous_gain
                    self.curvature_history[sample_id].append(curvature)

            enough_observations = (
                observation >= self.warmup_observations
            )
            sufficient_progress = (
                (not self.use_progress_condition) or (progress >= self.progress_threshold)
            )
            small_positive_gain = (
                gain is not None
                and 0.0 <= gain <= self.gain_threshold
            )
            stable_curvature = (
                curvature is not None
                and abs(curvature) <= self.curvature_threshold
            )

            condition_met = (
                enough_observations
                and sufficient_progress
                and small_positive_gain
                and stable_curvature
            )

            if condition_met:
                self.stable_counter[sample_id] += 1
            else:
                self.stable_counter[sample_id] = 0

            trajectory_records.append(
                {
                    "sample_id": sample_id,
                    "global_step": current_step,
                    "observation": observation,
                    "loss": current_loss,
                    "initial_loss": initial_loss,
                    "progress": progress,
                    "gain": gain,
                    "curvature": curvature,
                    "sufficient_progress": sufficient_progress,
                    "small_positive_gain": small_positive_gain,
                    "stable_curvature": stable_curvature,
                    "condition_met": condition_met,
                    "patience_count": self.stable_counter[sample_id],
                    "state": self.sample_state[sample_id],
                }
            )

            if self.stable_counter[sample_id] >= self.stop_patience:
                self.sample_state[sample_id] = "stopped"
                self.stop_step[sample_id] = current_step
                self.stop_observation[sample_id] = observation

                newly_stopped.append(
                    {
                        "sample_id": sample_id,
                        "step": current_step,
                        "observation": observation,
                        "loss": current_loss,
                        "initial_loss": initial_loss,
                        "progress": progress,
                        "gain": gain,
                        "curvature": curvature,
                        "progress_history": list(progress_hist),
                        "gain_history": list(
                            self.gain_history[sample_id]
                        ),
                        "curvature_history": list(
                            self.curvature_history[sample_id]
                        ),
                    }
                )

        self._append_jsonl(
            self.trajectory_log_path,
            trajectory_records,
        )
        self._append_jsonl(
            self.sample_log_path,
            newly_stopped,
        )

        return torch.tensor(
            active_mask,
            dtype=torch.bool,
            device=sample_losses.device,
        )

    def _write_summary(self):
        seen = len(self.sample_state)
        stopped = sum(
            state == "stopped"
            for state in self.sample_state.values()
        )
        active = seen - stopped

        stopped_progress = [
            list(self.progress_history[sample_id])[-1]
            for sample_id, state in self.sample_state.items()
            if state == "stopped"
            and len(self.progress_history[sample_id]) > 0
        ]

        summary = {
            "global_step": int(self.state.global_step),
            "seen_unique_samples": seen,
            "active_unique_samples": active,
            "stopped_unique_samples": stopped,
            "active_ratio": active / seen if seen > 0 else 1.0,
            "mean_stopped_progress": (
                sum(stopped_progress) / len(stopped_progress)
                if stopped_progress
                else None
            ),
            "total_batch_sample_occurrences": self.total_seen_samples,
            "total_active_sample_occurrences": self.total_active_samples,
            "total_stopped_sample_occurrences": self.total_stopped_samples,
            "effective_update_saving": (
                1.0
                - self.total_active_samples / self.total_seen_samples
                if self.total_seen_samples > 0
                else 0.0
            ),
            "stop_steps": self.stop_step,
            "stop_observations": self.stop_observation,
            "config": {
                "beta": self.beta,
                "use_progress_condition": self.use_progress_condition,
                "progress_threshold": self.progress_threshold,
                "gain_threshold": self.gain_threshold,
                "curvature_threshold": self.curvature_threshold,
                "window_size": self.window_size,
                "patience": self.stop_patience,
                "warmup_observations": self.warmup_observations,
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

        self.log(
            {
                "sample_stop/active_unique": summary[
                    "active_unique_samples"
                ],
                "sample_stop/stopped_unique": summary[
                    "stopped_unique_samples"
                ],
                "sample_stop/active_ratio": summary[
                    "active_ratio"
                ],
                "sample_stop/update_saving": summary[
                    "effective_update_saving"
                ],
            }
        )

        print(
            "[SampleStopNPO] "
            f"step={current_step}, "
            f"seen={summary['seen_unique_samples']}, "
            f"active={summary['active_unique_samples']}, "
            f"stopped={summary['stopped_unique_samples']}, "
            f"active_ratio={summary['active_ratio']:.4f}, "
            "effective_update_saving="
            f"{summary['effective_update_saving']:.4f}"
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

        # This mask is based on the state before the current occurrence.
        # Previously stopped samples are excluded before both the current-model
        # and reference-model forget forward passes.
        active_positions = self._get_active_positions(sample_ids)
        active_sample_ids = [
            sample_ids[position]
            for position in active_positions
        ]

        batch_size = len(sample_ids)
        active_count = len(active_positions)

        self.total_seen_samples += batch_size
        self.total_active_samples += active_count
        self.total_stopped_samples += batch_size - active_count

        forget_loss = None
        forget_outputs = (None, None)

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

            # A sample stopped at this observation still contributes to the
            # current update. It is skipped from its next occurrence.
            self._update_sample_states(
                sample_ids=active_sample_ids,
                sample_losses=per_sample_forget_loss,
            )

            forget_loss = per_sample_forget_loss.mean()
            forget_outputs = (None, current_outputs)

        retain_batch = inputs["retain"]
        retain_inputs = self._select_model_inputs(retain_batch)

        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs,
        )

        if forget_loss is None:
            # Keep a valid differentiable zero without performing a forget
            # forward when every forget sample in the batch has stopped.
            forget_loss = retain_loss * 0.0

        self.log(
            {
                "forget_loss": forget_loss.detach().float().item(),
                "retain_loss": retain_loss.detach().float().item(),
            }
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self._maybe_log_statistics()

        return (loss, forget_outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        self._write_summary()
        return super().save_model(
            output_dir=output_dir,
            _internal_call=_internal_call,
        )
