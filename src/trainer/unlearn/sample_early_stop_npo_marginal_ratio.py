import json
import math
import os

from trainer.unlearn.npo import NPO
from trainer.unlearn.sample_early_stop_npo_loss_irreversible import (
    SampleEarlyStopNPOLossIrreversible,
)


class SampleEarlyStopNPOMarginalRatio(NPO):
    """NPO with self-normalized marginal-progress stopping and rebound.

    The NPO objective, forget-anchored sampler, retain updates, and stopped
    sample loss masking are unchanged. Active samples stop when their marginal
    forgetting moving average decays relative to the global historical peak.
    Stopped samples reactivate when their marginal recovery moving average
    grows relative to that same peak. The historical peak is never reset.
    """

    _NLLGainEpochEndCallback = (
        SampleEarlyStopNPOLossIrreversible._NLLGainEpochEndCallback
    )
    _gain_model_inputs = staticmethod(
        SampleEarlyStopNPOLossIrreversible._gain_model_inputs
    )
    _gain_token_mean_nll = staticmethod(
        SampleEarlyStopNPOLossIrreversible._gain_token_mean_nll
    )
    _gain_slice_batch = staticmethod(
        SampleEarlyStopNPOLossIrreversible._gain_slice_batch
    )
    _scan_sample_nll = SampleEarlyStopNPOLossIrreversible._scan_sample_nll
    _initial_nll_cache_metadata = (
        SampleEarlyStopNPOLossIrreversible._initial_nll_cache_metadata
    )
    _initial_nll_payload = (
        SampleEarlyStopNPOLossIrreversible._initial_nll_payload
    )
    _load_initial_nll_cache = (
        SampleEarlyStopNPOLossIrreversible._load_initial_nll_cache
    )
    _write_initial_nll_cache = (
        SampleEarlyStopNPOLossIrreversible._write_initial_nll_cache
    )
    _scan_initial_sample_nll = (
        SampleEarlyStopNPOLossIrreversible._scan_initial_sample_nll
    )
    get_train_dataloader = (
        SampleEarlyStopNPOLossIrreversible.get_train_dataloader
    )
    compute_loss = SampleEarlyStopNPOLossIrreversible.compute_loss

    def __init__(
        self,
        moving_average_window=3,
        stop_ratio_threshold=0.1,
        rebound_ratio_threshold=0.2,
        ratio_epsilon=1e-8,
        initial_nll_cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nll_gain_log_prefix = "[NPO-MARGINAL-RATIO]"
        self.moving_average_window = int(moving_average_window)
        self.stop_ratio_threshold = float(stop_ratio_threshold)
        self.rebound_ratio_threshold = float(rebound_ratio_threshold)
        self.ratio_epsilon = float(ratio_epsilon)
        self.initial_nll_cache_path = (
            os.path.abspath(os.path.expanduser(initial_nll_cache_path))
            if initial_nll_cache_path
            else None
        )
        if self.moving_average_window < 1:
            raise ValueError("moving_average_window must be >= 1")
        if not 0.0 <= self.stop_ratio_threshold < 1.0:
            raise ValueError("stop_ratio_threshold must be in [0, 1)")
        if not 0.0 <= self.rebound_ratio_threshold <= 1.0:
            raise ValueError("rebound_ratio_threshold must be in [0, 1]")
        if self.ratio_epsilon <= 0.0:
            raise ValueError("ratio_epsilon must be positive")
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "SampleEarlyStopNPOMarginalRatio supports one process"
            )
        if int(self.args.dataloader_num_workers) != 0:
            raise RuntimeError(
                "SampleEarlyStopNPOMarginalRatio requires "
                "dataloader_num_workers=0"
            )
        if not hasattr(self.train_dataset, "forget") or not hasattr(
            self.train_dataset, "retain"
        ):
            raise RuntimeError(
                "train_dataset must expose forget and retain datasets"
            )

        self.forget_dataset = self.train_dataset.forget
        self.retain_dataset = self.train_dataset.retain
        self.num_forget_samples = len(self.forget_dataset)
        if self.num_forget_samples == 0:
            raise RuntimeError("forget dataset must be non-empty")
        if self.retain_dataset is None or len(self.retain_dataset) == 0:
            raise RuntimeError("retain dataset must be non-empty")

        self.all_sample_indices = set(range(self.num_forget_samples))
        self.active_samples = set(self.all_sample_indices)
        self.stopped_samples = set()
        self.initial_sample_nll = {}
        self.sample_nll_history = {
            i: [] for i in self.all_sample_indices
        }
        # This retains the existing cumulative L(t)-L(0) trajectory.
        self.nll_gain_history = {i: [] for i in self.all_sample_indices}
        self.epoch_gain_history = {i: [] for i in self.all_sample_indices}
        self.moving_average_history = {
            i: [] for i in self.all_sample_indices
        }
        # Local gain windows reset after reactivation; the historical maximum does not.
        self.episode_gain_history = {
            i: [] for i in self.all_sample_indices
        }
        self.episode_moving_average_history = {
            i: [] for i in self.all_sample_indices
        }
        self.historical_max_moving_average = {
            i: None for i in self.all_sample_indices
        }
        self.stopped_recovery_gain_history = {
            i: [] for i in self.all_sample_indices
        }
        self.stopped_recovery_moving_average_history = {
            i: [] for i in self.all_sample_indices
        }
        self.transition_history = {
            i: [] for i in self.all_sample_indices
        }
        self.stop_epoch = {}
        self.stop_nll = {}
        self.stop_progress = {}
        self.stop_moving_average = {}
        self.stop_ratio = {}
        self._last_finalized_epoch = 0
        self._saved_forget_instances = 0
        self._stream = None

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.initial_state_path = os.path.join(
            self.args.output_dir, "ies_nll_gain_initial_state.json"
        )
        self.history_path = os.path.join(
            self.args.output_dir, "ies_nll_gain_history.jsonl"
        )
        self.state_path = os.path.join(
            self.args.output_dir, "ies_nll_gain_state.json"
        )
        if self.is_world_process_zero():
            open(self.history_path, "w", encoding="utf-8").close()

        self._scan_initial_sample_nll()
        self.add_callback(self._NLLGainEpochEndCallback(self))
        print(
            f"{self._nll_gain_log_prefix} "
            f"moving_average_window={self.moving_average_window} "
            f"stop_ratio_threshold={self.stop_ratio_threshold} "
            f"rebound_ratio_threshold={self.rebound_ratio_threshold} "
            f"ratio_epsilon={self.ratio_epsilon} "
            "stop_patience=1 rebound_patience=1 "
            "rebound_signal=stopped_recovery_ma_over_historical_peak "
            "sampling=baseline_forget_anchor_masked",
            flush=True,
        )

    def _reset_active_episode(self, index):
        self.episode_gain_history[index] = []
        self.episode_moving_average_history[index] = []

    def _active_metrics(self, index, current_nll, previous_nll):
        epoch_gain = float(current_nll - previous_nll)
        progress = float(current_nll - self.initial_sample_nll[index])
        self.epoch_gain_history[index].append(epoch_gain)
        self.episode_gain_history[index].append(epoch_gain)

        moving_average = None
        historical_max_before = self.historical_max_moving_average[index]
        historical_max = historical_max_before
        marginal_ratio = None
        enough_gains = (
            len(self.episode_gain_history[index])
            >= self.moving_average_window
        )
        has_peak_comparison = False
        if enough_gains:
            window = self.episode_gain_history[index][
                -self.moving_average_window:
            ]
            moving_average = float(sum(window) / len(window))
            self.moving_average_history[index].append(moving_average)
            self.episode_moving_average_history[index].append(
                moving_average
            )
            historical_max = (
                moving_average
                if historical_max is None
                else max(historical_max, moving_average)
            )
            self.historical_max_moving_average[index] = historical_max
            # Initially the first moving average establishes the peak. After
            # reactivation, the retained global peak is immediately available.
            has_peak_comparison = historical_max_before is not None
            if historical_max > 0.0:
                marginal_ratio = max(moving_average, 0.0) / (
                    historical_max + self.ratio_epsilon
                )

        positive_peak_exists = (
            historical_max is not None and historical_max > 0.0
        )
        stop_eligible = (
            has_peak_comparison and positive_peak_exists and progress > 0.0
        )
        threshold_hit = (
            stop_eligible
            and marginal_ratio is not None
            and marginal_ratio < self.stop_ratio_threshold
        )
        return {
            "epoch_gain": epoch_gain,
            "cumulative_progress": progress,
            "moving_average_gain": moving_average,
            "historical_max_moving_average_gain": historical_max,
            "marginal_forgetting_ratio": marginal_ratio,
            "positive_peak_exists": positive_peak_exists,
            "enough_gains": enough_gains,
            "has_peak_comparison": has_peak_comparison,
            "stop_eligible": stop_eligible,
            "stop_threshold_hit": threshold_hit,
        }

    def _finalize_nll_gain_epoch(self, model=None):
        completed_epoch = int(round(float(self.state.epoch or 0.0)))
        if completed_epoch <= self._last_finalized_epoch:
            return
        active_before = sorted(self.active_samples)
        stopped_before = set(self.stopped_samples)
        observed = self._scan_sample_nll(model, self.all_sample_indices)
        newly_stopped = []
        reactivated = []
        sample_records = {}

        for index in active_before:
            current_nll = observed[index]
            initial_nll = self.initial_sample_nll[index]
            previous_nll = self.sample_nll_history[index][-1]
            metrics = self._active_metrics(index, current_nll, previous_nll)
            progress = metrics["cumulative_progress"]
            self.sample_nll_history[index].append(current_nll)
            self.nll_gain_history[index].append(progress)
            stopped_now = metrics["stop_threshold_hit"]
            if stopped_now:
                newly_stopped.append(index)
                self.stopped_recovery_gain_history[index] = []
                self.stopped_recovery_moving_average_history[index] = []
                self.stop_epoch[index] = completed_epoch
                self.stop_nll[index] = current_nll
                self.stop_progress[index] = progress
                self.stop_moving_average[index] = metrics[
                    "moving_average_gain"
                ]
                self.stop_ratio[index] = metrics[
                    "marginal_forgetting_ratio"
                ]
                self.transition_history[index].append(
                    {
                        "type": "stop",
                        "epoch": completed_epoch,
                        "length_normalized_sample_nll": current_nll,
                        "forgetting_progress_at_stop": progress,
                        **metrics,
                    }
                )
            sample_records[str(index)] = {
                "observed_this_epoch": True,
                "monitoring_mode": "forget_training",
                "length_normalized_sample_nll": current_nll,
                "initial_sample_nll": initial_nll,
                "nll_gain": progress,
                **metrics,
                "episode_gain_count": len(
                    self.episode_gain_history[index]
                ),
                "episode_moving_average_count": len(
                    self.episode_moving_average_history[index]
                ),
                "stopped_now": stopped_now,
                "rebound_ratio": None,
                "reactivated_now": False,
                "state_after": "stopped" if stopped_now else "active",
            }

        self.stopped_samples.update(newly_stopped)
        self.active_samples.difference_update(newly_stopped)

        for index in sorted(stopped_before):
            current_nll = observed[index]
            initial_nll = self.initial_sample_nll[index]
            previous_nll = self.sample_nll_history[index][-1]
            epoch_gain = float(current_nll - previous_nll)
            progress = float(current_nll - initial_nll)
            self.sample_nll_history[index].append(current_nll)
            self.nll_gain_history[index].append(progress)
            self.epoch_gain_history[index].append(epoch_gain)

            progress_at_stop = self.stop_progress[index]
            recovery_gain = float(previous_nll - current_nll)
            recovery_history = self.stopped_recovery_gain_history[index]
            recovery_history.append(recovery_gain)
            enough_recovery_gains = (
                len(recovery_history) >= self.moving_average_window
            )
            recovery_moving_average = None
            if enough_recovery_gains:
                recovery_window = recovery_history[
                    -self.moving_average_window:
                ]
                recovery_moving_average = float(
                    sum(recovery_window) / len(recovery_window)
                )
                self.stopped_recovery_moving_average_history[index].append(
                    recovery_moving_average
                )

            historical_peak = self.historical_max_moving_average[index]
            positive_peak_exists = (
                historical_peak is not None and historical_peak > 0.0
            )
            rebound_eligible = (
                enough_recovery_gains and positive_peak_exists
            )
            rebound_ratio = (
                max(recovery_moving_average, 0.0)
                / (historical_peak + self.ratio_epsilon)
                if rebound_eligible
                else None
            )
            reactivation_hit = (
                rebound_eligible
                and rebound_ratio is not None
                and rebound_ratio > self.rebound_ratio_threshold
            )
            if reactivation_hit:
                reactivated.append(index)
                self.transition_history[index].append(
                    {
                        "type": "reactivate",
                        "epoch": completed_epoch,
                        "length_normalized_sample_nll": current_nll,
                        "forgetting_progress_at_stop": progress_at_stop,
                        "recovery_gain": recovery_gain,
                        "recovery_moving_average_gain": (
                            recovery_moving_average
                        ),
                        "historical_max_moving_average_gain": (
                            historical_peak
                        ),
                        "recovery_gain_count": len(recovery_history),
                        "enough_recovery_gains": enough_recovery_gains,
                        "rebound_ratio": rebound_ratio,
                    }
                )
            sample_records[str(index)] = {
                "observed_this_epoch": True,
                "monitoring_mode": "stopped_forward_only",
                "length_normalized_sample_nll": current_nll,
                "initial_sample_nll": initial_nll,
                "nll_gain": progress,
                "epoch_gain": epoch_gain,
                "cumulative_progress": progress,
                "stop_epoch": self.stop_epoch[index],
                "stop_nll": self.stop_nll[index],
                "forgetting_progress_at_stop": progress_at_stop,
                "recovery_gain": recovery_gain,
                "recovery_moving_average_gain": recovery_moving_average,
                "historical_max_moving_average_gain": historical_peak,
                "recovery_gain_count": len(recovery_history),
                "enough_recovery_gains": enough_recovery_gains,
                "rebound_eligible": rebound_eligible,
                "rebound_ratio": rebound_ratio,
                "rebound_threshold_hit": reactivation_hit,
                "stopped_now": False,
                "reactivated_now": reactivation_hit,
                "state_after": (
                    "active" if reactivation_hit else "stopped"
                ),
            }

        for index in reactivated:
            self.stopped_samples.discard(index)
            self.active_samples.add(index)
            self._reset_active_episode(index)
            self.stop_epoch.pop(index, None)
            self.stop_nll.pop(index, None)
            self.stop_progress.pop(index, None)
            self.stop_moving_average.pop(index, None)
            self.stop_ratio.pop(index, None)

        self._saved_forget_instances += len(stopped_before)
        self._last_finalized_epoch = completed_epoch
        epoch_record = {
            "epoch": completed_epoch,
            "monitoring_signal": (
                "per_sample_length_normalized_nll_epoch_gain"
            ),
            "epoch_gain_definition": "normalized_nll_t - normalized_nll_t-1",
            "moving_average_window": self.moving_average_window,
            "marginal_ratio_definition": (
                "max(current_moving_average,0) / "
                "(max(historical_max_moving_average,0) + epsilon)"
            ),
            "stop_ratio_threshold": self.stop_ratio_threshold,
            "stop_comparison": "marginal_forgetting_ratio < threshold",
            "stop_patience": 1,
            "recovery_gain_definition": (
                "previous_normalized_nll - current_normalized_nll"
            ),
            "rebound_ratio_definition": (
                "max(recovery_moving_average,0) / "
                "(max(historical_max_moving_average,0) + epsilon)"
            ),
            "rebound_ratio_threshold": self.rebound_ratio_threshold,
            "reactivation_comparison": "rebound_ratio > threshold",
            "reactivation_patience": 1,
            "ratio_epsilon": self.ratio_epsilon,
            "rebound_enabled": True,
            "rebound_signal": (
                "stopped_recovery_moving_average_over_global_"
                "historical_forgetting_peak"
            ),
            "historical_peak_reset_on_reactivation": False,
            "recovery_window_reset_on_stop": True,
            "sampling_mode": "baseline_forget_anchor_masked",
            "active": len(self.active_samples),
            "stopped": len(self.stopped_samples),
            "newly_stopped": newly_stopped,
            "reactivated": reactivated,
            "stopped_forward_only_instances": len(stopped_before),
            "saved_forget_instances_this_epoch": len(stopped_before),
            "cumulative_saved_forget_instances": self._saved_forget_instances,
            "retain_steps_this_epoch": math.ceil(
                self.num_forget_samples
                / int(self.args.per_device_train_batch_size)
            ),
            "samples": sample_records,
        }
        if self.is_world_process_zero():
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record) + "\n")
            state_record = {
                **epoch_record,
                "active_indices": sorted(self.active_samples),
                "stopped_indices": sorted(self.stopped_samples),
                "initial_sample_nll": self._string_keys(
                    self.initial_sample_nll
                ),
                "sample_nll_history": self._string_keys(
                    self.sample_nll_history
                ),
                "nll_gain_history": self._string_keys(
                    self.nll_gain_history
                ),
                "epoch_gain_history": self._string_keys(
                    self.epoch_gain_history
                ),
                "moving_average_history": self._string_keys(
                    self.moving_average_history
                ),
                "episode_gain_history": self._string_keys(
                    self.episode_gain_history
                ),
                "episode_moving_average_history": self._string_keys(
                    self.episode_moving_average_history
                ),
                "historical_max_moving_average": self._string_keys(
                    self.historical_max_moving_average
                ),
                "stopped_recovery_gain_history": self._string_keys(
                    self.stopped_recovery_gain_history
                ),
                "stopped_recovery_moving_average_history": (
                    self._string_keys(
                        self.stopped_recovery_moving_average_history
                    )
                ),
                "transition_history": self._string_keys(
                    self.transition_history
                ),
                "stop_epoch": self._string_keys(self.stop_epoch),
                "stop_nll": self._string_keys(self.stop_nll),
                "stop_progress": self._string_keys(self.stop_progress),
                "stop_moving_average": self._string_keys(
                    self.stop_moving_average
                ),
                "stop_ratio": self._string_keys(self.stop_ratio),
            }
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump(state_record, handle, indent=2)

        self.log(
            {
                "npo_marginal_ratio_active": len(self.active_samples),
                "npo_marginal_ratio_stopped": len(self.stopped_samples),
                "npo_marginal_ratio_newly_stopped": len(newly_stopped),
                "npo_marginal_ratio_reactivated": len(reactivated),
                "npo_marginal_ratio_saved_forget_total": (
                    self._saved_forget_instances
                ),
            }
        )
        print(
            f"{self._nll_gain_log_prefix} epoch={completed_epoch} "
            f"active={len(self.active_samples)} "
            f"stopped={len(self.stopped_samples)} "
            f"new={len(newly_stopped)} "
            f"reactivated={len(reactivated)} "
            f"stopped_forward={len(stopped_before)} "
            "sampling=baseline_forget_anchor_masked",
            flush=True,
        )

    @staticmethod
    def _string_keys(mapping):
        return {str(key): value for key, value in mapping.items()}
