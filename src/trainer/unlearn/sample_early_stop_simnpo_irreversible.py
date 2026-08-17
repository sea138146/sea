import os

import torch
import torch.nn.functional as F

from trainer.unlearn.sample_early_stop_npo_loss_irreversible import (
    SampleEarlyStopNPOLossIrreversible,
)
from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll


class SampleEarlyStopSimNPOIrreversible(SimNPO):
    """SIMNPO with reversible per-sample normalized-NLL-gain stopping.

    The controller is identical to the NPO version: active forget samples
    leave gradient updates after reaching the gain threshold, stopped samples
    remain under forward-only monitoring, and samples whose gain rebounds
    below the reactivation threshold return to forget updates. Retain updates
    continue for every original training step.
    """

    # These methods implement only the method-independent controller. The
    # SIMNPO-specific objective remains in compute_loss below.
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
    _finalize_nll_gain_epoch = (
        SampleEarlyStopNPOLossIrreversible._finalize_nll_gain_epoch
    )

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
        self._nll_gain_log_prefix = "[IES-SIMNPO-NLL-Gain]"
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
                "SampleEarlyStopSimNPOIrreversible supports one process"
            )
        if int(self.args.dataloader_num_workers) != 0:
            raise RuntimeError(
                "SampleEarlyStopSimNPOIrreversible requires "
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
            "ies_simnpo_nll_gain_initial_state.json",
        )
        self.history_path = os.path.join(
            self.args.output_dir,
            "ies_simnpo_nll_gain_history.jsonl",
        )
        self.state_path = os.path.join(
            self.args.output_dir,
            "ies_simnpo_nll_gain_state.json",
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
                sequence_nll, forget_outputs = compute_batch_nll(
                    model, forget_inputs
                )

                # compute_batch_nll returns a token-summed sequence NLL. This
                # division preserves the original SIMNPO objective exactly and
                # is reduced only over active forget samples.
                loss_mask = forget_inputs["labels"].ne(-100)
                per_sample_simnpo = (
                    sequence_nll / loss_mask.sum(-1).clamp_min(1)
                    - self.delta
                )
                active_mean_forget_loss = (
                    -F.logsigmoid(
                        self.beta * per_sample_simnpo
                    ).mean()
                    * 2.0
                    / self.beta
                )
                active_count = len(active_positions)
                forget_scale = active_count / original_batch_size
                forget_loss = active_mean_forget_loss * forget_scale

        loss = self.alpha * retain_loss
        if forget_loss is not None:
            loss = loss + self.gamma * forget_loss

        if getattr(self, "log_step_details", True):
            log_values = {
                "retain_loss": float(retain_loss.detach()),
                "ies_simnpo_nll_gain_active_batch": active_count,
            }
            if forget_loss is not None:
                log_values["forget_loss"] = float(forget_loss.detach())
                log_values["ies_simnpo_nll_gain_forget_scale"] = (
                    forget_scale
                )
            self.log(log_values)

        if return_outputs:
            return loss, forget_outputs
        return loss
