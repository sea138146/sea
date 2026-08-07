import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import TrainerCallback

from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll


class SampleEarlyStopSimNPOIrreversible(SimNPO):
    """SIMNPO with the same irreversible stopping criterion as NPO."""

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
                len(self.indices), generator=generator
            ).tolist()
            return iter(self.indices[position] for position in order)

        def __len__(self):
            return len(self.indices)

    class _EpochEndCallback(TrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_end(self, args, state, control, **kwargs):
            self.owner._finalize_epoch()
            return control

    def __init__(self, stop_threshold=1.0e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if float(stop_threshold) < 0.0:
            raise ValueError("stop_threshold must be non-negative")
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Dynamic irreversible stopping supports single-process training only."
            )
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is required")
        if getattr(self.train_dataset, "anchor", None) != "forget":
            raise RuntimeError("The training dataset must use data.anchor=forget.")
        if not hasattr(self.train_dataset, "forget"):
            raise RuntimeError("train_dataset must expose the forget dataset.")

        self.stop_threshold = float(stop_threshold)
        self.forget_dataset = self.train_dataset.forget
        self.all_sample_indices = set(range(len(self.forget_dataset)))
        self.active_samples = set(self.all_sample_indices)
        self.stopped_samples = set()
        self.normalized_nll_history = {
            index: [] for index in self.all_sample_indices
        }
        self.second_difference_history = {
            index: [] for index in self.all_sample_indices
        }
        self.stop_epoch = {}
        self.stop_score = {}
        self._active_sampler = None
        self._last_finalized_epoch = 0
        self._saved_sample_computations = 0

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.history_path = os.path.join(
            self.args.output_dir,
            "ies_simnpo_normalized_nll_irreversible_history.jsonl",
        )
        self.state_path = os.path.join(
            self.args.output_dir,
            "ies_simnpo_normalized_nll_irreversible_state.json",
        )
        if self.is_world_process_zero():
            open(self.history_path, "w", encoding="utf-8").close()
        self.add_callback(self._EpochEndCallback(self))
        print(
            "[IES-SIMNPO-Irreversible] "
            f"stop_threshold={self.stop_threshold}, "
            "reactivation=false, stopped_sample_monitoring=false"
        )

    @staticmethod
    def _model_inputs(batch):
        return {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "labels")
        }

    @staticmethod
    def _token_mean_nll(sequence_nll, labels):
        valid_token_count = labels[..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
        return sequence_nll / valid_token_count.to(dtype=sequence_nll.dtype)

    def _get_train_sampler(self):
        self._active_sampler = self._MutableSubsetRandomSampler(
            sorted(self.active_samples), seed=int(self.args.seed)
        )
        return self._active_sampler

    def _append_normalized_nlls(self, sample_indices, values):
        for sample_index, value in zip(
            sample_indices, values.detach().float().cpu().tolist()
        ):
            index = int(sample_index)
            if index in self.stopped_samples:
                raise RuntimeError(
                    f"Stopped sample {index} appeared in the active loader."
                )
            self.normalized_nll_history[index].append(float(value))

    def _finalize_epoch(self):
        completed_epoch = int(round(float(self.state.epoch or 0.0)))
        if completed_epoch <= self._last_finalized_epoch:
            return

        saved_this_epoch = len(self.stopped_samples)
        self._saved_sample_computations += saved_this_epoch
        newly_stopped = []
        sample_records = {}

        for index in sorted(self.active_samples):
            history = self.normalized_nll_history[index]
            second_difference = None
            second_difference_abs_window = []
            score = None
            stable_condition = False
            if len(history) >= 3:
                second_difference = float(
                    history[-1] - 2.0 * history[-2] + history[-3]
                )
                self.second_difference_history[index].append(second_difference)
                if len(self.second_difference_history[index]) >= 3:
                    second_difference_abs_window = [
                        abs(value)
                        for value in self.second_difference_history[index][-3:]
                    ]
                    score = sum(second_difference_abs_window) / 3.0
                    stable_condition = score <= self.stop_threshold
                    if stable_condition:
                        newly_stopped.append(index)
                        self.stop_epoch[index] = completed_epoch
                        self.stop_score[index] = score

            sample_records[str(index)] = {
                "normalized_nll": float(history[-1]) if history else None,
                "normalized_nll_window": [float(value) for value in history[-3:]],
                "second_difference": second_difference,
                "second_difference_abs_window": second_difference_abs_window,
                "score": score,
                "stable_condition": stable_condition,
                "stopped_now": index in newly_stopped,
            }

        self.stopped_samples.update(newly_stopped)
        self.active_samples = self.all_sample_indices - self.stopped_samples
        if self._active_sampler is not None:
            self._active_sampler.update(sorted(self.active_samples))
            self._active_sampler.set_epoch(completed_epoch)
        self._last_finalized_epoch = completed_epoch

        epoch_record = {
            "epoch": completed_epoch,
            "monitoring_signal": "per_sample_normalized_nll",
            "second_difference": "m_t - 2*m_t_minus_1 + m_t_minus_2",
            "stop_threshold": self.stop_threshold,
            "score_definition": "mean(abs(latest_3_second_differences))",
            "reactivation_enabled": False,
            "stopped_sample_monitoring": False,
            "active": len(self.active_samples),
            "stopped": len(self.stopped_samples),
            "newly_stopped": newly_stopped,
            "saved_sample_computations_this_epoch": saved_this_epoch,
            "cumulative_saved_sample_computations": self._saved_sample_computations,
            "samples_observed_this_epoch": sample_records,
        }
        if self.is_world_process_zero():
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        **epoch_record,
                        "active_indices": sorted(self.active_samples),
                        "stopped_indices": sorted(self.stopped_samples),
                        "stop_epoch": self.stop_epoch,
                        "stop_score": self.stop_score,
                        "normalized_nll_history": self.normalized_nll_history,
                        "second_difference_history": self.second_difference_history,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        self.log(
            {
                "ies_simnpo_active": len(self.active_samples),
                "ies_simnpo_stopped": len(self.stopped_samples),
                "ies_simnpo_newly_stopped": len(newly_stopped),
                "ies_simnpo_saved_this_epoch": saved_this_epoch,
                "ies_simnpo_saved_total": self._saved_sample_computations,
            }
        )
        print(
            f"[IES-SIMNPO-Irreversible] epoch={completed_epoch} "
            f"active={len(self.active_samples)} "
            f"stopped={len(self.stopped_samples)} "
            f"new={len(newly_stopped)} "
            f"saved_this_epoch={saved_this_epoch} "
            f"saved_total={self._saved_sample_computations}"
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_batch = inputs["forget"]
        retain_batch = inputs["retain"]
        if "index" not in forget_batch:
            raise RuntimeError(
                "forget batch has no index; use "
                "DataCollatorForSupervisedDatasetwithIndex."
            )

        forget_inputs = self._model_inputs(forget_batch)
        sequence_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        self._append_normalized_nlls(
            forget_batch["index"].detach().cpu().tolist(),
            self._token_mean_nll(sequence_nll, forget_inputs["labels"]),
        )

        # Preserve the original SIMNPO forget objective exactly.
        loss_mask = forget_inputs["labels"].ne(-100)
        per_sample_simnpo = sequence_nll / loss_mask.sum(-1) - self.delta
        forget_loss = (
            -F.logsigmoid(self.beta * per_sample_simnpo).mean()
            * 2.0
            / self.beta
        )
        retain_loss = self.compute_retain_loss(
            model, self._model_inputs(retain_batch)
        )
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self.log(
            {
                "forget_loss": float(forget_loss.detach()),
                "retain_loss": float(retain_loss.detach()),
                "ies_simnpo_active_batch": len(forget_batch["index"]),
            }
        )
        return (loss, forget_outputs) if return_outputs else loss
