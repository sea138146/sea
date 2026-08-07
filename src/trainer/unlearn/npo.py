import json as _ies_json
import os as _ies_os

import torch
import torch.nn.functional as _ies_F
from torch.utils.data import (
    DataLoader as _IESDataLoader,
    Sampler as _IESSampler,
    Subset as _IESSubset,
)
from transformers import TrainerCallback as _IESTrainerCallback
from trainer.utils import compute_batch_nll as _ies_compute_batch_nll

from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    class _SampleNLLEpochCallback(_IESTrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            self.owner._log_sample_nll_snapshot(
                model=model,
                epoch=float(state.epoch or 0.0),
                snapshot_type="epoch_end",
            )
            return control

    def __init__(
        self,
        beta=1.0,
        log_per_sample_normalized_nll=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.log_per_sample_normalized_nll = bool(
            log_per_sample_normalized_nll
        )
        self.npo_sample_nll_log_path = None
        self.npo_initial_sample_nll = {}
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        if self.log_per_sample_normalized_nll:
            if not hasattr(self.train_dataset, "forget"):
                raise RuntimeError(
                    "Per-sample NPO logging requires train_dataset.forget."
                )
            _ies_os.makedirs(self.args.output_dir, exist_ok=True)
            self.npo_sample_nll_log_path = _ies_os.path.join(
                self.args.output_dir, "npo_sample_normalized_nll.jsonl"
            )
            if self.is_world_process_zero():
                open(self.npo_sample_nll_log_path, "w", encoding="utf-8").close()
            self._log_sample_nll_snapshot(
                model=self.model,
                epoch=0.0,
                snapshot_type="initial",
            )
            self.add_callback(self._SampleNLLEpochCallback(self))

    @staticmethod
    def _nll_model_inputs(batch):
        return {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "labels")
        }

    def _log_sample_nll_snapshot(self, model, epoch, snapshot_type):
        model = model or self.model
        forget_dataset = self.train_dataset.forget
        loader = _IESDataLoader(
            forget_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )
        was_training = model.training
        model.eval()
        observed = {}
        try:
            with torch.no_grad():
                for batch in loader:
                    if "index" not in batch:
                        raise RuntimeError(
                            "Per-sample NPO logging requires forget sample indices."
                        )
                    batch = self._prepare_inputs(batch)
                    model_inputs = self._nll_model_inputs(batch)
                    sequence_nll, _ = _ies_compute_batch_nll(model, model_inputs)
                    valid_token_count = (
                        model_inputs["labels"][..., 1:]
                        .ne(-100)
                        .sum(dim=-1)
                        .clamp_min(1)
                    )
                    normalized_nll = sequence_nll / valid_token_count.to(
                        sequence_nll.dtype
                    )
                    indices = batch["index"].detach().cpu().tolist()
                    nll_values = normalized_nll.detach().float().cpu().tolist()
                    token_counts = valid_token_count.detach().cpu().tolist()
                    for sample_index, nll_value, token_count in zip(
                        indices, nll_values, token_counts
                    ):
                        index = int(sample_index)
                        if index in observed:
                            raise RuntimeError(
                                f"Duplicate forget sample index in NLL scan: {index}"
                            )
                        observed[index] = (float(nll_value), int(token_count))
        finally:
            model.train(was_training)

        expected_count = len(forget_dataset)
        if len(observed) != expected_count:
            raise RuntimeError(
                "Per-sample NPO NLL scan covered "
                f"{len(observed)} of {expected_count} forget samples."
            )

        if snapshot_type == "initial":
            self.npo_initial_sample_nll = {
                index: value[0] for index, value in observed.items()
            }
        elif set(observed) != set(self.npo_initial_sample_nll):
            raise RuntimeError(
                "Epoch NLL scan sample indices differ from the initial scan."
            )

        records = []
        for sample_index in sorted(observed):
            sample_nll, valid_token_count = observed[sample_index]
            initial_sample_nll = self.npo_initial_sample_nll[sample_index]
            record = {
                "snapshot_type": snapshot_type,
                "epoch": float(epoch),
                "global_step": int(self.state.global_step),
                "sample_index": sample_index,
                "valid_token_count": valid_token_count,
                "length_normalized_sample_nll": sample_nll,
                "initial_sample_nll": initial_sample_nll,
                "nll_gain": sample_nll - initial_sample_nll,
            }
            records.append(record)
            if self.is_world_process_zero():
                print(
                    "[NPO-SAMPLE-NLL] "
                    "snapshot={snapshot_type} epoch={epoch:.6f} "
                    "step={global_step} index={sample_index} "
                    "length_normalized_sample_nll="
                    "{length_normalized_sample_nll:.8f} "
                    "initial_sample_nll={initial_sample_nll:.8f} "
                    "nll_gain={nll_gain:.8f}".format(**record),
                    flush=True,
                )

        if self.is_world_process_zero():
            with open(
                self.npo_sample_nll_log_path, "a", encoding="utf-8"
            ) as handle:
                for record in records:
                    handle.write(_ies_json.dumps(record) + "\n")
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        current_sequence_nll, forget_model_outputs = _ies_compute_batch_nll(
            model, forget_inputs
        )
        with torch.no_grad():
            reference_sequence_nll, _ = _ies_compute_batch_nll(
                self.ref_model, forget_inputs
            )
        forget_loss = (
            -2.0 / self.beta * _ies_F.logsigmoid(
                self.beta * (current_sequence_nll - reference_sequence_nll)
            ).mean()
        )
        valid_token_count = (
            forget_inputs["labels"][..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
        ).to(dtype=current_sequence_nll.dtype)
        current_per_sample_normalized_nll = (
            current_sequence_nll / valid_token_count
        )
        reference_per_sample_normalized_nll = (
            reference_sequence_nll / valid_token_count
        )
        current_normalized_nll = current_per_sample_normalized_nll.mean()
        reference_normalized_nll = reference_per_sample_normalized_nll.mean()
        self.log({
            "forget_normalized_nll": float(current_normalized_nll.detach()),
            "reference_forget_normalized_nll": float(reference_normalized_nll.detach()),
            "forget_normalized_nll_delta": float(
                (current_normalized_nll - reference_normalized_nll).detach()
            ),
        })
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_model_outputs) if return_outputs else loss


class SampleEarlyStopNPO(NPO):
    """NPO using the original IES sample-selection mechanism.

    Each forget sample stores one length-normalized current-model NLL per
    epoch. Active samples obtain the NLL from the training forward pass.
    Excluded samples obtain it from remaining_loader under torch.no_grad().

    The IES criterion is:

      d2_t = m_t - 2*m_{t-1} + m_{t-2}

    The absolute second differences are smoothed with a moving average.
    A sample is excluded when the sum of the latest k moving-average values
    is smaller than threshold.

    The excluded set is recomputed after every epoch. Therefore, an excluded
    sample can be reactivated when it no longer satisfies the criterion.
    """

    class _MutableSubsetRandomSampler(_IESSampler):
        def __init__(self, indices, seed=0):
            self.indices = list(indices)
            self.seed = int(seed)
            self.epoch = 0

        def update(self, indices):
            self.indices = list(indices)

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            if not self.indices:
                return iter(())

            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)

            order = torch.randperm(
                len(self.indices),
                generator=generator,
            ).tolist()

            return iter(self.indices[pos] for pos in order)

        def __len__(self):
            return len(self.indices)

    class _EpochEndCallback(_IESTrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_end(
            self,
            args,
            state,
            control,
            model=None,
            **kwargs,
        ):
            self.owner._ies_finalize_epoch(model=model)
            return control

    def __init__(
        self,
        threshold=1.0e-3,
        moving_average_rate=3,
        k=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        if int(moving_average_rate) < 1:
            raise ValueError("moving_average_rate must be >= 1")

        if int(k) < 1:
            raise ValueError("k must be >= 1")

        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "IES dynamic loaders currently support only "
                "single-process training."
            )

        if self.train_dataset is None:
            raise RuntimeError("train_dataset is required")

        if getattr(self.train_dataset, "anchor", None) != "forget":
            raise RuntimeError(
                "IES loader partitioning requires data.anchor=forget."
            )

        if not hasattr(self.train_dataset, "forget"):
            raise RuntimeError(
                "train_dataset must expose its forget dataset."
            )

        self.threshold = float(threshold)
        self.moving_average_rate = int(moving_average_rate)
        self.k = int(k)

        self.forget_dataset = self.train_dataset.forget
        self.num_forget_samples = len(self.forget_dataset)

        self.all_sample_indices = set(
            range(self.num_forget_samples)
        )
        self.active_samples = set(self.all_sample_indices)
        self.excluded_samples = set()

        self.sample_loss_history = {
            idx: []
            for idx in range(self.num_forget_samples)
        }
        self.derivative_history = {
            idx: []
            for idx in range(self.num_forget_samples)
        }

        self._ies_sampler = None
        self.remaining_loader = None
        self._ies_last_finalized_epoch = 0
        self._ies_total_saved_backprop_instances = 0

        _ies_os.makedirs(
            self.args.output_dir,
            exist_ok=True,
        )

        self.ies_log_path = _ies_os.path.join(
            self.args.output_dir,
            "ies_sample_history.jsonl",
        )
        self.ies_state_path = _ies_os.path.join(
            self.args.output_dir,
            "ies_sample_state.json",
        )

        self.add_callback(
            self._EpochEndCallback(self)
        )

    @staticmethod
    def _model_inputs(batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    @staticmethod
    def _token_mean_nll(sequence_nll, labels):
        """
        Convert each sample's sequence-summed NLL into mean NLL
        over valid target tokens.

        compute_batch_nll shifts labels by one token internally,
        so the valid-token count must use labels[..., 1:].
        """
        shifted_labels = labels[..., 1:]

        valid_token_count = (
            shifted_labels.ne(-100)
            .sum(dim=-1)
            .clamp_min(1)
            .to(dtype=sequence_nll.dtype)
        )

        return sequence_nll / valid_token_count

    @staticmethod
    def _second_difference(losses):
        return float(
            losses[-1]
            - 2.0 * losses[-2]
            + losses[-3]
        )

    @staticmethod
    def _moving_average_abs(values, window_size):
        abs_values = [
            abs(float(value))
            for value in values
        ]

        window_sum = sum(
            abs_values[:window_size]
        )
        result = [
            window_sum / window_size
        ]

        for pos in range(
            window_size,
            len(abs_values),
        ):
            window_sum += (
                abs_values[pos]
                - abs_values[pos - window_size]
            )
            result.append(
                window_sum / window_size
            )

        return result

    def _get_train_sampler(self):
        self._ies_sampler = (
            self._MutableSubsetRandomSampler(
                sorted(self.active_samples),
                seed=int(self.args.seed),
            )
        )
        return self._ies_sampler

    def _build_remaining_loader(self, indices):
        if not indices:
            return None

        dataset = _IESSubset(
            self.forget_dataset,
            list(indices),
        )

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(
                self.args.per_device_train_batch_size
            ),
            "shuffle": False,
            "collate_fn": self.data_collator,
            "num_workers": int(
                self.args.dataloader_num_workers
            ),
            "pin_memory": bool(
                self.args.dataloader_pin_memory
            ),
            "drop_last": False,
        }

        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = bool(
                self.args.dataloader_persistent_workers
            )

        return _IESDataLoader(
            **loader_kwargs
        )

    def _append_sample_losses(
        self,
        indices,
        losses,
    ):
        for sample_idx, sample_loss in zip(
            indices,
            losses,
        ):
            idx = int(sample_idx)

            if idx not in self.all_sample_indices:
                raise RuntimeError(
                    f"forget sample index {idx} is outside "
                    f"[0, {self.num_forget_samples - 1}]"
                )

            self.sample_loss_history[idx].append(
                float(sample_loss)
            )

    def _monitor_remaining_samples(
        self,
        model,
    ):
        if self.remaining_loader is None:
            return 0

        if model is None:
            model = self.model

        was_training = model.training
        model.eval()

        monitored = 0

        with torch.no_grad():
            for batch in self.remaining_loader:
                if "index" not in batch:
                    raise RuntimeError(
                        "remaining forget batch has no 'index'. "
                        "Set the training collator argument "
                        "index: index."
                    )

                batch = self._prepare_inputs(batch)

                sample_indices = (
                    batch["index"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                model_inputs = self._model_inputs(
                    batch
                )

                with self.compute_loss_context_manager():
                    current_nll, _ = (
                        _ies_compute_batch_nll(
                            model,
                            model_inputs,
                        )
                    )

                monitoring_nll = self._token_mean_nll(
                    sequence_nll=current_nll,
                    labels=batch["labels"],
                )

                self._append_sample_losses(
                    sample_indices,
                    monitoring_nll
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                )

                monitored += len(
                    sample_indices
                )

        if was_training:
            model.train()

        return monitored

    def _refresh_derivatives_and_partition(self):
        # A moving-average window of w and the latest k averages
        # require w + k - 1 second-difference observations.
        history_limit = (
            self.moving_average_rate
            + self.k
            - 1
        )

        # Official IES logic:
        # calculate one latest second difference per epoch.
        for idx, losses in (
            self.sample_loss_history.items()
        ):
            if len(losses) < 3:
                continue

            latest_derivative = (
                self._second_difference(losses)
            )

            derivatives = (
                self.derivative_history[idx]
            )
            derivatives.append(
                latest_derivative
            )

            if len(derivatives) > history_limit:
                self.derivative_history[idx] = (
                    derivatives[-history_limit:]
                )

        # Recompute excluded_samples from scratch.
        # This permits reactivation.
        new_excluded = set()
        sample_scores = {}

        for idx, derivatives in (
            self.derivative_history.items()
        ):
            if len(derivatives) < history_limit:
                continue

            moving_averages = (
                self._moving_average_abs(
                    derivatives,
                    self.moving_average_rate,
                )
            )

            derivative_sum = float(
                sum(
                    moving_averages[-self.k:]
                )
            )

            sample_scores[idx] = (
                derivative_sum
            )

            if derivative_sum < self.threshold:
                new_excluded.add(idx)

        self.excluded_samples = (
            new_excluded
        )
        self.active_samples = (
            self.all_sample_indices
            - self.excluded_samples
        )

        if self._ies_sampler is not None:
            self._ies_sampler.update(
                sorted(self.active_samples)
            )

        self.remaining_loader = (
            self._build_remaining_loader(
                sorted(self.excluded_samples)
            )
        )

        return sample_scores

    def _ies_finalize_epoch(
        self,
        model=None,
    ):
        completed_epoch = int(
            round(
                float(
                    self.state.epoch or 0.0
                )
            )
        )

        if (
            completed_epoch
            <= self._ies_last_finalized_epoch
        ):
            return

        # The samples excluded during this epoch are evaluated
        # with forward-only computation.
        monitored = (
            self._monitor_remaining_samples(
                model
            )
        )

        previous_excluded = set(
            self.excluded_samples
        )

        sample_scores = (
            self._refresh_derivatives_and_partition()
        )

        newly_excluded = sorted(
            self.excluded_samples
            - previous_excluded
        )
        reactivated = sorted(
            previous_excluded
            - self.excluded_samples
        )

        active_count = len(
            self.active_samples
        )
        excluded_count = len(
            self.excluded_samples
        )

        self._ies_total_saved_backprop_instances += (
            excluded_count
        )
        self._ies_last_finalized_epoch = (
            completed_epoch
        )

        if self._ies_sampler is not None:
            self._ies_sampler.set_epoch(
                completed_epoch
            )

        epoch_record = {
            "epoch": completed_epoch,
            "criterion": (
                "IES_second_difference"
            ),
            "threshold": self.threshold,
            "moving_average_rate": (
                self.moving_average_rate
            ),
            "k": self.k,
            "active": active_count,
            "excluded": excluded_count,
            "newly_excluded": (
                newly_excluded
            ),
            "reactivated": reactivated,
            "remaining_forward_instances": (
                monitored
            ),
            "cumulative_saved_backprop_instances": (
                self._ies_total_saved_backprop_instances
            ),
            "scores": {
                str(idx): score
                for idx, score
                in sample_scores.items()
            },
        }

        if self.is_world_process_zero():
            with open(
                self.ies_log_path,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    _ies_json.dumps(
                        epoch_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            state_record = {
                **epoch_record,
                "active_indices": sorted(
                    self.active_samples
                ),
                "excluded_indices": sorted(
                    self.excluded_samples
                ),
                "sample_loss_history": {
                    str(idx): values
                    for idx, values
                    in self.sample_loss_history.items()
                },
                "derivative_history": {
                    str(idx): values
                    for idx, values
                    in self.derivative_history.items()
                },
            }

            with open(
                self.ies_state_path,
                "w",
                encoding="utf-8",
            ) as handle:
                _ies_json.dump(
                    state_record,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        self.log(
            {
                "ies_active": active_count,
                "ies_excluded": excluded_count,
                "ies_newly_excluded": len(
                    newly_excluded
                ),
                "ies_reactivated": len(
                    reactivated
                ),
                "ies_remaining_forward": (
                    monitored
                ),
                "ies_saved_backprop_total": (
                    self._ies_total_saved_backprop_instances
                ),
            }
        )

        print(
            f"[IES-NPO] epoch={completed_epoch} "
            f"active={active_count} "
            f"excluded={excluded_count} "
            f"new={len(newly_excluded)} "
            f"reactivated={len(reactivated)} "
            f"remaining_forward={monitored}"
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_batch = inputs["forget"]
        retain_batch = inputs["retain"]

        if "index" not in forget_batch:
            raise RuntimeError(
                "forget batch has no 'index'. "
                "Set the training collator argument "
                "index: index."
            )

        sample_indices = (
            forget_batch["index"]
            .detach()
            .cpu()
            .tolist()
        )

        forget_inputs = self._model_inputs(
            forget_batch
        )

        current_nll, forget_outputs = (
            _ies_compute_batch_nll(
                model,
                forget_inputs,
            )
        )

        with torch.no_grad():
            reference_nll, _ = (
                _ies_compute_batch_nll(
                    self.ref_model,
                    forget_inputs,
                )
            )

        per_sample_npo = (
            -2.0
            / self.beta
            * _ies_F.logsigmoid(
                self.beta
                * (
                    current_nll
                    - reference_nll
                )
            )
        )

        forget_loss = (
            per_sample_npo.mean()
        )

        # IES monitors the current model's per-sample task loss.
        # For causal LM unlearning, this is length-normalized NLL.
        monitoring_nll = self._token_mean_nll(
            sequence_nll=current_nll,
            labels=forget_batch["labels"],
        )

        self._append_sample_losses(
            sample_indices,
            monitoring_nll
            .detach()
            .float()
            .cpu()
            .tolist(),
        )

        retain_inputs = self._model_inputs(
            retain_batch
        )

        retain_loss = (
            self.compute_retain_loss(
                model=model,
                retain_inputs=retain_inputs,
            )
        )

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_loss
        )

        self.log(
            {
                "forget_loss": float(
                    forget_loss.detach()
                ),
                "retain_loss": float(
                    retain_loss.detach()
                ),
                "ies_active_batch": len(
                    sample_indices
                ),
            }
        )

        return (
            (loss, forget_outputs)
            if return_outputs
            else loss
        )
