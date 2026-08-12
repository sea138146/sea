import os

import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.sample_early_stop_npo_loss_irreversible import (
    SampleEarlyStopNPOLossIrreversible,
)


class SampleEarlyStopGradDiffIrreversible(GradDiff):
    """GradDiff with reversible per-sample normalized-NLL-gain stopping.

    The stopping controller is method-independent and matches the NPO and
    SimNPO implementations. Active forget samples leave gradient updates after
    reaching the gain threshold. Stopped samples remain under forward-only NLL
    monitoring and return to forget updates if their gain falls below the
    rebound threshold. The baseline forget-anchored loader remains unchanged; stopped samples are
    masked only in the forget objective, while retain updates continue normally.

    The GradDiff objective itself is unchanged:

        loss = gamma * (-forget_cross_entropy) + alpha * retain_loss
    """

    # Reuse only the method-independent monitoring and stream controller. The
    # GradDiff-specific forget objective remains in compute_loss below.
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
        patience=1,
        rebound_delta=0.2,
        reactivation_patience=None,
        initial_nll_cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nll_gain_log_prefix = "[IES-GRADIFF-NLL-Gain]"
        self.warm_up = int(warm_up)
        self.gain_threshold = float(gain_threshold)
        self.patience = int(patience)
        self.rebound_delta = float(rebound_delta)
        self.reactivation_patience = (
            self.patience
            if reactivation_patience is None
            else int(reactivation_patience)
        )
        self.rebound_threshold = self.gain_threshold - self.rebound_delta
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
                "SampleEarlyStopGradDiffIrreversible supports one process"
            )
        if int(self.args.dataloader_num_workers) != 0:
            raise RuntimeError(
                "SampleEarlyStopGradDiffIrreversible requires "
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
            "ies_grad_diff_nll_gain_initial_state.json",
        )
        self.history_path = os.path.join(
            self.args.output_dir,
            "ies_grad_diff_nll_gain_history.jsonl",
        )
        self.state_path = os.path.join(
            self.args.output_dir,
            "ies_grad_diff_nll_gain_state.json",
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
        forget_loss = None
        active_count = 0

        if forget_batch is not None:
            if "index" not in forget_batch:
                raise RuntimeError(
                    "forget batch has no index; use "
                    "DataCollatorForSupervisedDatasetwithIndex"
                )
            sample_indices = forget_batch["index"].detach().cpu().tolist()
            original_valid_tokens = int(
                forget_batch["labels"][..., 1:].ne(-100).sum().item()
            )
            active_positions = [
                position
                for position, index in enumerate(sample_indices)
                if int(index) in self.active_samples
            ]
            if active_positions:
                positions = torch.tensor(active_positions, dtype=torch.long)
                forget_batch = self._gain_slice_batch(
                    forget_batch, positions
                )
                forget_inputs = self._gain_model_inputs(forget_batch)
                forget_outputs = model(**forget_inputs)

                # Preserve the original GradDiff objective exactly. The NLL
                # gain controller is evaluated separately at epoch end and is
                # never substituted for this forget loss.
                active_valid_tokens = int(
                    forget_inputs["labels"][..., 1:].ne(-100).sum().item()
                )
                forget_scale = active_valid_tokens / max(
                    original_valid_tokens, 1
                )
                forget_loss = -forget_outputs.loss * forget_scale
                active_count = len(active_positions)

        loss = self.alpha * retain_loss
        log_values = {
            "retain_loss": float(retain_loss.detach()),
            "ies_grad_diff_nll_gain_active_batch": active_count,
        }
        if forget_loss is not None:
            loss = loss + self.gamma * forget_loss
            log_values["forget_loss"] = float(forget_loss.detach())
            log_values["ies_grad_diff_nll_gain_forget_scale"] = forget_scale
        self.log(log_values)

        if return_outputs:
            return loss, forget_outputs
        return loss
