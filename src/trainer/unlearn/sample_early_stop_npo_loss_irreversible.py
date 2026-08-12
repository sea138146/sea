import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll


class SampleEarlyStopNPOLossIrreversible(NPO):
    """Reversible per-sample stopping based on normalized-NLL gain.

    Epochs up to and including warm_up only record trajectories. After
    warm-up, an active sample leaves forget updates when its length-normalized
    NLL gain is at least gain_threshold for patience consecutive epoch-end
    snapshots. A stopped sample is monitored forward-only and re-enters forget
    updates after its gain is at most gain_threshold - rebound_delta for
    reactivation_patience consecutive snapshots. Retain updates continue for
    every original training step.
    """

    class _NLLGainEpochEndCallback(TrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            self.owner._finalize_nll_gain_epoch(model=model)
            return control

    def __init__(
        self,
        warm_up=2,
        gain_threshold=2.0,
        patience=2,
        rebound_delta=0.2,
        reactivation_patience=None,
        initial_nll_cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nll_gain_log_prefix = "[IES-NPO-NLL-Gain]"
        self.warm_up = int(warm_up)
        self.gain_threshold = float(gain_threshold)
        self.patience = int(patience)
        self.rebound_delta = float(rebound_delta)
        self.reactivation_patience = (
            self.patience
            if reactivation_patience is None
            else int(reactivation_patience)
        )
        self.rebound_threshold = (
            self.gain_threshold - self.rebound_delta
        )
        self.initial_nll_cache_path = (
            os.path.abspath(os.path.expanduser(initial_nll_cache_path))
            if initial_nll_cache_path
            else None
        )
        if self.warm_up < 0:
            raise ValueError("warm_up must be non-negative")
        if self.gain_threshold < 0:
            raise ValueError("gain_threshold must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if not 0.0 <= self.rebound_delta <= self.gain_threshold:
            raise ValueError(
                "rebound_delta must be between 0 and gain_threshold"
            )
        if self.reactivation_patience < 1:
            raise ValueError("reactivation_patience must be >= 1")
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "SampleEarlyStopNPOLossIrreversible currently supports one process"
            )
        if int(self.args.dataloader_num_workers) != 0:
            raise RuntimeError(
                "SampleEarlyStopNPOLossIrreversible requires dataloader_num_workers=0"
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
            index: [] for index in self.all_sample_indices
        }
        self.nll_gain_history = {
            index: [] for index in self.all_sample_indices
        }
        self.gain_streak = {
            index: 0 for index in self.all_sample_indices
        }
        self.reactivation_streak = {
            index: 0 for index in self.all_sample_indices
        }
        self.transition_history = {
            index: [] for index in self.all_sample_indices
        }
        self.stop_epoch = {}
        self.stop_nll = {}
        self.stop_gain = {}
        self._last_finalized_epoch = 0
        self._saved_forget_instances = 0
        self._stream = None

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.initial_state_path = os.path.join(
            self.args.output_dir,
            "ies_nll_gain_initial_state.json",
        )
        self.history_path = os.path.join(
            self.args.output_dir,
            "ies_nll_gain_history.jsonl",
        )
        self.state_path = os.path.join(
            self.args.output_dir,
            "ies_nll_gain_state.json",
        )
        if self.is_world_process_zero():
            open(self.history_path, "w", encoding="utf-8").close()

        self._scan_initial_sample_nll()
        self.add_callback(self._NLLGainEpochEndCallback(self))
        print(
            f"{self._nll_gain_log_prefix} "
            f"warm_up={self.warm_up} "
            f"gain_threshold={self.gain_threshold} "
            f"patience={self.patience} "
            f"rebound_delta={self.rebound_delta} "
            f"rebound_threshold={self.rebound_threshold} "
            f"reactivation_patience={self.reactivation_patience} "
            "rebound=true sampling=baseline_forget_anchor_masked",
            flush=True,
        )

    @staticmethod
    def _gain_model_inputs(batch):
        return {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "labels")
        }

    @staticmethod
    def _gain_token_mean_nll(sequence_nll, labels):
        # compute_batch_nll returns one token-summed NLL per sequence.
        # Divide exactly once here to obtain mean NLL per valid token.
        valid_token_count = (
            labels[..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
        )
        return sequence_nll / valid_token_count.to(sequence_nll.dtype)

    @staticmethod
    def _gain_slice_batch(batch, positions):
        return {
            key: value.index_select(0, positions.to(value.device))
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    def _scan_sample_nll(self, model, indices):
        model = model if model is not None else self.model
        indices = sorted(indices)
        if not indices:
            return {}
        loader = DataLoader(
            self.forget_dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            sampler=indices,
            drop_last=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )
        model_device = next(model.parameters()).device
        if model_device != self.accelerator.device:
            model.to(self.accelerator.device)
        was_training = model.training
        model.eval()
        observed = {}
        try:
            with torch.no_grad():
                for batch in loader:
                    if "index" not in batch:
                        raise RuntimeError(
                            "forget samples require index values"
                        )
                    batch = self._prepare_inputs(batch)
                    model_inputs = self._gain_model_inputs(batch)
                    with self.compute_loss_context_manager():
                        sequence_nll, _ = compute_batch_nll(
                            model, model_inputs
                        )
                    values = self._gain_token_mean_nll(
                        sequence_nll, model_inputs["labels"]
                    ).detach().float().cpu().tolist()
                    sample_indices = (
                        batch["index"].detach().cpu().tolist()
                    )
                    for sample_index, value in zip(
                        sample_indices, values
                    ):
                        index = int(sample_index)
                        if index in observed:
                            raise RuntimeError(
                                f"duplicate forget sample index {index}"
                            )
                        observed[index] = float(value)
        finally:
            model.train(was_training)

        if set(observed) != set(indices):
            missing = sorted(set(indices) - set(observed))
            raise RuntimeError(
                f"NLL scan missed forget sample indices: {missing}"
            )
        return observed

    def _initial_nll_cache_metadata(self):
        dataset = getattr(self.forget_dataset, "data", None)
        model_config = getattr(self.model, "config", None)
        return {
            "schema_version": 1,
            "definition": "sequence_nll / valid_answer_token_count",
            "num_forget_samples": self.num_forget_samples,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "dataset_max_length": getattr(
                self.forget_dataset, "max_length", None
            ),
            "model_name_or_path": getattr(
                model_config, "_name_or_path", None
            ),
            "tokenizer_name_or_path": getattr(
                self.processing_class, "name_or_path", None
            ),
        }

    def _initial_nll_payload(self, observed):
        return {
            **self._initial_nll_cache_metadata(),
            "gain_definition": (
                "current_normalized_nll - initial_normalized_nll"
            ),
            "initial_sample_nll": {
                str(index): observed[index]
                for index in sorted(observed)
            },
        }

    def _load_initial_nll_cache(self):
        if not self.initial_nll_cache_path or not os.path.isfile(
            self.initial_nll_cache_path
        ):
            return None
        with open(
            self.initial_nll_cache_path, "r", encoding="utf-8"
        ) as handle:
            payload = json.load(handle)
        raw_values = payload.get("initial_sample_nll")
        if not isinstance(raw_values, dict):
            raise RuntimeError(
                "initial NLL cache has no initial_sample_nll mapping: "
                f"{self.initial_nll_cache_path}"
            )
        observed = {
            int(key): float(value) for key, value in raw_values.items()
        }
        if set(observed) != self.all_sample_indices:
            raise RuntimeError(
                "initial NLL cache sample indices do not match the forget set: "
                f"{self.initial_nll_cache_path}"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in observed.values()
        ):
            raise RuntimeError(
                "initial NLL cache contains an invalid value: "
                f"{self.initial_nll_cache_path}"
            )
        expected = self._initial_nll_cache_metadata()
        for key in (
            "definition",
            "num_forget_samples",
            "dataset_fingerprint",
            "dataset_max_length",
            "model_name_or_path",
            "tokenizer_name_or_path",
        ):
            cached_value = payload.get(key)
            expected_value = expected[key]
            if cached_value is not None and cached_value != expected_value:
                raise RuntimeError(
                    f"initial NLL cache {key} mismatch: "
                    f"cached={cached_value!r}, expected={expected_value!r}, "
                    f"path={self.initial_nll_cache_path}"
                )
        print(
            f"{self._nll_gain_log_prefix} initial_nll_cache=hit "
            f"path={self.initial_nll_cache_path}",
            flush=True,
        )
        return observed

    def _write_initial_nll_cache(self, observed):
        if not self.initial_nll_cache_path:
            return
        cache_dir = os.path.dirname(self.initial_nll_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        temporary_path = f"{self.initial_nll_cache_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                self._initial_nll_payload(observed),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_path, self.initial_nll_cache_path)
        print(
            f"{self._nll_gain_log_prefix} initial_nll_cache=written "
            f"path={self.initial_nll_cache_path}",
            flush=True,
        )

    def _scan_initial_sample_nll(self):
        observed = self._load_initial_nll_cache()
        cache_hit = observed is not None
        if observed is None:
            print(
                f"{self._nll_gain_log_prefix} initial_nll_cache=miss; "
                "scanning fixed initial model",
                flush=True,
            )
            observed = self._scan_sample_nll(
                self.model, self.all_sample_indices
            )
            self._write_initial_nll_cache(observed)
        self.initial_sample_nll = observed
        for index in sorted(self.all_sample_indices):
            self.sample_nll_history[index].append(observed[index])
            self.nll_gain_history[index].append(0.0)
        if self.is_world_process_zero():
            with open(
                self.initial_state_path, "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        **self._initial_nll_payload(observed),
                        "source": "cache" if cache_hit else "model_forward",
                        "cache_path": self.initial_nll_cache_path,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

    def get_train_dataloader(self):
        # Match the baseline forget-anchored dataset, sampler, and random
        # retain pairing. Stopped samples remain in the loader and are masked
        # only in compute_loss, keeping optimizer steps unchanged.
        self._stream = None
        return Trainer.get_train_dataloader(self)

    def _finalize_nll_gain_epoch(self, model=None):
        completed_epoch = int(round(float(self.state.epoch or 0.0)))
        if completed_epoch <= self._last_finalized_epoch:
            return

        active_before = sorted(self.active_samples)
        stopped_before = set(self.stopped_samples)

        # Every sample receives one fixed epoch-end snapshot. Stopped samples
        # are forward-only in this epoch; a reactivated sample returns to the
        # forget stream from the next epoch.
        observed = self._scan_sample_nll(
            model, self.all_sample_indices
        )
        newly_stopped = []
        reactivated = []
        sample_records = {}

        for index in active_before:
            current_nll = observed[index]
            initial_nll = self.initial_sample_nll[index]
            previous_nll = self.sample_nll_history[index][-1]
            gain = float(current_nll - initial_nll)
            nll_change = float(current_nll - previous_nll)
            self.sample_nll_history[index].append(current_nll)
            self.nll_gain_history[index].append(gain)
            self.reactivation_streak[index] = 0

            eligible = completed_epoch > self.warm_up
            threshold_hit = eligible and gain >= self.gain_threshold
            if threshold_hit:
                self.gain_streak[index] += 1
            else:
                self.gain_streak[index] = 0

            stopped_now = (
                eligible
                and self.gain_streak[index] >= self.patience
            )
            if stopped_now:
                newly_stopped.append(index)
                self.stop_epoch[index] = completed_epoch
                self.stop_nll[index] = current_nll
                self.stop_gain[index] = gain
                self.reactivation_streak[index] = 0
                self.transition_history[index].append(
                    {
                        "type": "stop",
                        "epoch": completed_epoch,
                        "length_normalized_sample_nll": current_nll,
                        "nll_gain": gain,
                    }
                )

            sample_records[str(index)] = {
                "observed_this_epoch": True,
                "length_normalized_sample_nll": current_nll,
                "initial_sample_nll": initial_nll,
                "nll_gain": gain,
                "nll_change_since_previous_epoch": nll_change,
                "nll_change_since_stop": (
                    0.0 if stopped_now else None
                ),
                "monitoring_mode": "forget_training",
                "eligible_after_warm_up": eligible,
                "stop_threshold_hit": threshold_hit,
                "stop_consecutive_hit_count": self.gain_streak[index],
                "stopped_now": stopped_now,
                "reactivation_threshold_hit": False,
                "reactivation_consecutive_hit_count": 0,
                "reactivated_now": False,
                "state_after": "stopped" if stopped_now else "active",
            }

        self.stopped_samples.update(newly_stopped)
        self.active_samples.difference_update(newly_stopped)

        for index in sorted(stopped_before):
            current_nll = observed[index]
            initial_nll = self.initial_sample_nll[index]
            previous_nll = self.sample_nll_history[index][-1]
            gain = float(current_nll - initial_nll)
            nll_change = float(current_nll - previous_nll)
            current_stop_epoch = self.stop_epoch[index]
            current_stop_nll = self.stop_nll[index]
            current_stop_gain = self.stop_gain[index]
            nll_change_since_stop = float(
                current_nll - current_stop_nll
            )
            self.sample_nll_history[index].append(current_nll)
            self.nll_gain_history[index].append(gain)

            reactivation_hit = gain <= self.rebound_threshold
            if reactivation_hit:
                self.reactivation_streak[index] += 1
            else:
                self.reactivation_streak[index] = 0
            reactivated_now = (
                self.reactivation_streak[index]
                >= self.reactivation_patience
            )
            if reactivated_now:
                reactivated.append(index)
                self.transition_history[index].append(
                    {
                        "type": "reactivate",
                        "epoch": completed_epoch,
                        "length_normalized_sample_nll": current_nll,
                        "nll_gain": gain,
                        "rebound_threshold": self.rebound_threshold,
                    }
                )

            sample_records[str(index)] = {
                "observed_this_epoch": True,
                "monitoring_mode": "stopped_forward_only",
                "length_normalized_sample_nll": current_nll,
                "initial_sample_nll": initial_nll,
                "nll_gain": gain,
                "nll_change_since_previous_epoch": nll_change,
                "stop_nll": current_stop_nll,
                "stop_gain": current_stop_gain,
                "nll_change_since_stop": nll_change_since_stop,
                "epochs_since_stop": (
                    completed_epoch - current_stop_epoch
                ),
                "eligible_after_warm_up": True,
                "stop_threshold_hit": gain >= self.gain_threshold,
                "stop_consecutive_hit_count": self.gain_streak[index],
                "stopped_now": False,
                "reactivation_threshold": self.rebound_threshold,
                "reactivation_threshold_hit": reactivation_hit,
                "reactivation_consecutive_hit_count": (
                    self.reactivation_streak[index]
                ),
                "reactivated_now": reactivated_now,
                "state_after": (
                    "active" if reactivated_now else "stopped"
                ),
                "stop_epoch": current_stop_epoch,
            }

        for index in reactivated:
            self.stopped_samples.discard(index)
            self.active_samples.add(index)
            self.gain_streak[index] = 0
            self.reactivation_streak[index] = 0
            self.stop_epoch.pop(index, None)
            self.stop_nll.pop(index, None)
            self.stop_gain.pop(index, None)

        self._saved_forget_instances += len(stopped_before)
        self._last_finalized_epoch = completed_epoch
        epoch_record = {
            "epoch": completed_epoch,
            "monitoring_signal": (
                "per_sample_length_normalized_nll_gain"
            ),
            "gain_definition": (
                "current_normalized_nll - initial_normalized_nll"
            ),
            "warm_up": self.warm_up,
            "gain_threshold": self.gain_threshold,
            "stop_comparison": "nll_gain >= gain_threshold",
            "patience": self.patience,
            "rebound_delta": self.rebound_delta,
            "rebound_threshold": self.rebound_threshold,
            "reactivation_comparison": (
                "nll_gain <= gain_threshold - rebound_delta"
            ),
            "reactivation_patience": self.reactivation_patience,
            "rebound_enabled": True,
            "sampling_mode": "baseline_forget_anchor_masked",
            "active": len(self.active_samples),
            "stopped": len(self.stopped_samples),
            "newly_stopped": newly_stopped,
            "reactivated": reactivated,
            "stopped_forward_only_instances": len(stopped_before),
            "saved_forget_instances_this_epoch": len(stopped_before),
            "cumulative_saved_forget_instances": (
                self._saved_forget_instances
            ),
            "retain_steps_this_epoch": (
                math.ceil(
                    self.num_forget_samples
                    / int(self.args.per_device_train_batch_size)
                )
            ),
            "samples": sample_records,
        }

        if self.is_world_process_zero():
            with open(
                self.history_path, "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(epoch_record, ensure_ascii=False) + "\n"
                )
            with open(
                self.state_path, "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        **epoch_record,
                        "active_indices": sorted(self.active_samples),
                        "stopped_indices": sorted(self.stopped_samples),
                        "initial_sample_nll": {
                            str(index): value
                            for index, value
                            in self.initial_sample_nll.items()
                        },
                        "sample_nll_history": {
                            str(index): values
                            for index, values
                            in self.sample_nll_history.items()
                        },
                        "nll_gain_history": {
                            str(index): values
                            for index, values
                            in self.nll_gain_history.items()
                        },
                        "gain_streak": {
                            str(index): value
                            for index, value in self.gain_streak.items()
                        },
                        "reactivation_streak": {
                            str(index): value
                            for index, value
                            in self.reactivation_streak.items()
                        },
                        "transition_history": {
                            str(index): values
                            for index, values
                            in self.transition_history.items()
                        },
                        "stop_epoch": {
                            str(index): value
                            for index, value in self.stop_epoch.items()
                        },
                        "stop_nll": {
                            str(index): value
                            for index, value in self.stop_nll.items()
                        },
                        "stop_gain": {
                            str(index): value
                            for index, value in self.stop_gain.items()
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        self.log(
            {
                "ies_nll_gain_active": len(self.active_samples),
                "ies_nll_gain_stopped": len(self.stopped_samples),
                "ies_nll_gain_newly_stopped": len(newly_stopped),
                "ies_nll_gain_reactivated": len(reactivated),
                "ies_nll_gain_saved_forget_total": (
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

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        retain_inputs = self._gain_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(
            model=model, retain_inputs=retain_inputs
        )

        forget_batch = inputs.get("forget")
        forget_outputs = None
        active_count = 0
        forget_loss = None

        if forget_batch is not None:
            sample_indices = (
                forget_batch["index"].detach().cpu().tolist()
            )
            original_batch_size = len(sample_indices)
            active_positions = [
                position
                for position, index in enumerate(sample_indices)
                if int(index) in self.active_samples
            ]
            if active_positions:
                positions = torch.tensor(
                    active_positions, dtype=torch.long
                )
                forget_batch = self._gain_slice_batch(
                    forget_batch, positions
                )
                forget_inputs = self._gain_model_inputs(forget_batch)
                current_nll, forget_outputs = compute_batch_nll(
                    model, forget_inputs
                )
                with torch.no_grad():
                    reference_nll, _ = compute_batch_nll(
                        self.ref_model, forget_inputs
                    )
                per_sample_npo = (
                    -2.0
                    / self.beta
                    * F.logsigmoid(
                        self.beta * (current_nll - reference_nll)
                    )
                )
                active_count = len(active_positions)
                forget_scale = active_count / original_batch_size
                forget_loss = per_sample_npo.mean() * forget_scale

        loss = self.alpha * retain_loss
        log_values = {
            "retain_loss": float(retain_loss.detach()),
            "ies_nll_gain_active_batch": active_count,
        }
        if forget_loss is not None:
            loss = loss + self.gamma * forget_loss
            log_values["forget_loss"] = float(forget_loss.detach())
            log_values["ies_nll_gain_forget_scale"] = forget_scale
        self.log(log_values)

        if return_outputs:
            return loss, forget_outputs
        return loss
