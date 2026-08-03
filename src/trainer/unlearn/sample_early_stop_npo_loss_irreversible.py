import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import TrainerCallback

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll


class SampleEarlyStopNPOLossIrreversible(NPO):
    """Irreversible sample stopping based on per-sample NPO loss curvature.

    Monitoring signal:
        m_i^(t) = l_NPO,i^(t)

    Second difference:
        d2_i^(t) = m_i^(t) - 2*m_i^(t-1) + m_i^(t-2)

    A sample is permanently stopped after ``stable_checks`` consecutive
    observations satisfying:

        abs(d2_i^(t)) <= stop_threshold

    The observation that completes the stable streak still participates in the
    current epoch. Stopping becomes effective from the next epoch.

    Once stopped, a sample is removed from the training sampler permanently.
    It receives no further current-model forward, reference-model forward,
    NPO-loss computation, backward pass, or trajectory observation.
    """

    class _MutableSubsetRandomSampler(Sampler):
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

            return iter(
                self.indices[position]
                for position in order
            )

        def __len__(self):
            return len(self.indices)

    class _EpochEndCallback(TrainerCallback):
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
            self.owner._finalize_epoch()
            return control

    def __init__(
        self,
        stop_threshold=1.0e-5,
        stable_checks=2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if float(stop_threshold) < 0.0:
            raise ValueError(
                "stop_threshold must be non-negative"
            )

        if int(stable_checks) < 1:
            raise ValueError(
                "stable_checks must be at least 1"
            )

        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Dynamic irreversible stopping currently supports "
                "single-process training only."
            )

        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is required"
            )

        if getattr(self.train_dataset, "anchor", None) != "forget":
            raise RuntimeError(
                "The training dataset must use data.anchor=forget."
            )

        if not hasattr(self.train_dataset, "forget"):
            raise RuntimeError(
                "train_dataset must expose the forget dataset."
            )

        self.stop_threshold = float(stop_threshold)
        self.stable_checks = int(stable_checks)

        self.forget_dataset = self.train_dataset.forget
        self.num_forget_samples = len(self.forget_dataset)

        self.all_sample_indices = set(
            range(self.num_forget_samples)
        )
        self.active_samples = set(
            self.all_sample_indices
        )
        self.stopped_samples = set()

        # m_i^(t) = unreduced per-sample NPO loss.
        self.npo_loss_history = {
            idx: []
            for idx in range(self.num_forget_samples)
        }
        self.second_difference_history = {
            idx: []
            for idx in range(self.num_forget_samples)
        }

        self.stable_streak = defaultdict(int)
        self.stop_epoch = {}
        self.stop_score = {}

        self._active_sampler = None
        self._last_finalized_epoch = 0

        # Number of stopped sample occurrences that were completely skipped.
        self._saved_sample_computations = 0

        os.makedirs(
            self.args.output_dir,
            exist_ok=True,
        )

        self.history_path = os.path.join(
            self.args.output_dir,
            "ies_npo_loss_irreversible_history.jsonl",
        )
        self.state_path = os.path.join(
            self.args.output_dir,
            "ies_npo_loss_irreversible_state.json",
        )

        if self.is_world_process_zero():
            open(
                self.history_path,
                "w",
                encoding="utf-8",
            ).close()

        self.add_callback(
            self._EpochEndCallback(self)
        )

        print(
            "[IES-NPO-Irreversible] "
            f"stop_threshold={self.stop_threshold}, "
            f"stable_checks={self.stable_checks}, "
            "reactivation=false, "
            "stopped_sample_monitoring=false"
        )

    @staticmethod
    def _model_inputs(batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _get_train_sampler(self):
        self._active_sampler = (
            self._MutableSubsetRandomSampler(
                sorted(self.active_samples),
                seed=int(self.args.seed),
            )
        )
        return self._active_sampler

    def _append_npo_losses(
        self,
        sample_indices,
        per_sample_npo,
    ):
        values = (
            per_sample_npo
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        for sample_idx, value in zip(
            sample_indices,
            values,
        ):
            idx = int(sample_idx)

            if idx in self.stopped_samples:
                raise RuntimeError(
                    f"Stopped sample {idx} unexpectedly appeared "
                    "in the active training loader."
                )

            self.npo_loss_history[idx].append(
                float(value)
            )

    def _finalize_epoch(self):
        completed_epoch = int(
            round(float(self.state.epoch or 0.0))
        )

        if completed_epoch <= self._last_finalized_epoch:
            return

        # These samples were already stopped before this epoch and therefore
        # were completely absent from the epoch's training loader.
        stopped_before_epoch = set(
            self.stopped_samples
        )
        saved_this_epoch = len(
            stopped_before_epoch
        )
        self._saved_sample_computations += (
            saved_this_epoch
        )

        newly_stopped = []
        sample_records = {}

        # Only active samples have a new observation this epoch.
        for idx in sorted(self.active_samples):
            history = self.npo_loss_history[idx]

            second_difference = None
            score = None
            stable_condition = False

            if len(history) >= 3:
                second_difference = float(
                    history[-1]
                    - 2.0 * history[-2]
                    + history[-3]
                )
                score = abs(second_difference)

                self.second_difference_history[idx].append(
                    second_difference
                )

                stable_condition = (
                    score <= self.stop_threshold
                )

                if stable_condition:
                    self.stable_streak[idx] += 1
                else:
                    self.stable_streak[idx] = 0

                if (
                    self.stable_streak[idx]
                    >= self.stable_checks
                ):
                    newly_stopped.append(idx)
                    self.stop_epoch[idx] = completed_epoch
                    self.stop_score[idx] = score

            sample_records[str(idx)] = {
                "npo_loss": (
                    float(history[-1])
                    if history
                    else None
                ),
                "npo_loss_window": [
                    float(value)
                    for value in history[-3:]
                ],
                "second_difference": second_difference,
                "score": score,
                "stable_condition": stable_condition,
                "stable_streak": self.stable_streak[idx],
                "stopped_now": idx in newly_stopped,
            }

        self.stopped_samples.update(
            newly_stopped
        )
        self.active_samples = (
            self.all_sample_indices
            - self.stopped_samples
        )

        if self._active_sampler is not None:
            self._active_sampler.update(
                sorted(self.active_samples)
            )
            self._active_sampler.set_epoch(
                completed_epoch
            )

        self._last_finalized_epoch = (
            completed_epoch
        )

        epoch_record = {
            "epoch": completed_epoch,
            "monitoring_signal": "per_sample_npo_loss",
            "second_difference": (
                "m_t - 2*m_t_minus_1 + m_t_minus_2"
            ),
            "stop_threshold": self.stop_threshold,
            "stable_checks": self.stable_checks,
            "reactivation_enabled": False,
            "stopped_sample_monitoring": False,
            "active": len(self.active_samples),
            "stopped": len(self.stopped_samples),
            "newly_stopped": newly_stopped,
            "saved_sample_computations_this_epoch": (
                saved_this_epoch
            ),
            "cumulative_saved_sample_computations": (
                self._saved_sample_computations
            ),
            "samples_observed_this_epoch": sample_records,
        }

        if self.is_world_process_zero():
            with open(
                self.history_path,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
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
                "stopped_indices": sorted(
                    self.stopped_samples
                ),
                "stop_epoch": {
                    str(idx): epoch
                    for idx, epoch
                    in self.stop_epoch.items()
                },
                "stop_score": {
                    str(idx): score
                    for idx, score
                    in self.stop_score.items()
                },
                "npo_loss_history": {
                    str(idx): [
                        float(value)
                        for value in history
                    ]
                    for idx, history
                    in self.npo_loss_history.items()
                },
                "second_difference_history": {
                    str(idx): [
                        float(value)
                        for value in history
                    ]
                    for idx, history
                    in self.second_difference_history.items()
                },
            }

            with open(
                self.state_path,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    state_record,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        self.log(
            {
                "ies_npo_active": len(
                    self.active_samples
                ),
                "ies_npo_stopped": len(
                    self.stopped_samples
                ),
                "ies_npo_newly_stopped": len(
                    newly_stopped
                ),
                "ies_npo_saved_this_epoch": (
                    saved_this_epoch
                ),
                "ies_npo_saved_total": (
                    self._saved_sample_computations
                ),
            }
        )

        print(
            f"[IES-NPO-Irreversible] "
            f"epoch={completed_epoch} "
            f"active={len(self.active_samples)} "
            f"stopped={len(self.stopped_samples)} "
            f"new={len(newly_stopped)} "
            f"saved_this_epoch={saved_this_epoch} "
            f"saved_total={self._saved_sample_computations}"
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
                "forget batch has no 'index'"
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
            compute_batch_nll(
                model,
                forget_inputs,
            )
        )

        with torch.no_grad():
            reference_nll, _ = (
                compute_batch_nll(
                    self.ref_model,
                    forget_inputs,
                )
            )

        # This is the unreduced loss actually averaged into the NPO
        # forget objective:
        #
        # m_i^(t) = l_NPO,i^(t)
        per_sample_npo = (
            -2.0
            / self.beta
            * F.logsigmoid(
                self.beta
                * (
                    current_nll
                    - reference_nll
                )
            )
        )

        self._append_npo_losses(
            sample_indices,
            per_sample_npo,
        )

        forget_loss = per_sample_npo.mean()

        retain_inputs = self._model_inputs(
            retain_batch
        )
        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs,
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
                "ies_npo_active_batch": len(
                    sample_indices
                ),
            }
        )

        if return_outputs:
            return loss, forget_outputs

        return loss
