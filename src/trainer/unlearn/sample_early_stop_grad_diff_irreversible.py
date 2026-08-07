import json
import os

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll


class SampleEarlyStopGradDiffIrreversible(GradDiff):
    """Irreversible GradDiff early stopping using normalized-NLL curvature.

    A sample stops when the mean absolute value of its latest three second
    differences is no larger than stop_threshold; no patience is applied.
    """

    class _Sampler(Sampler):
        def __init__(self, indices, seed):
            self.indices, self.seed, self.epoch = list(indices), int(seed), 0

        def update(self, indices):
            self.indices = list(indices)

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            return iter(self.indices[position] for position in order)

        def __len__(self):
            return len(self.indices)

    class _Callback(TrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_end(self, args, state, control, **kwargs):
            self.owner._finalize_epoch()
            return control

    def __init__(self, stop_threshold=1.0e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if float(stop_threshold) < 0:
            raise ValueError("invalid irreversible-stopping parameters")
        if self.accelerator.num_processes != 1:
            raise RuntimeError("Irreversible sample stopping supports single-process training only.")
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is required")
        if hasattr(self.train_dataset, "forget"):
            self.forget_dataset = self.train_dataset.forget
        elif hasattr(self.train_dataset, "forget_dataset"):
            self.forget_dataset = self.train_dataset.forget_dataset
        else:
            raise RuntimeError("train_dataset must expose the forget dataset.")

        self.stop_threshold = float(stop_threshold)
        self.all_sample_indices = set(range(len(self.forget_dataset)))
        self.active_samples = set(self.all_sample_indices)
        self.stopped_samples = set()
        self.forget_nll_history = {idx: [] for idx in self.all_sample_indices}
        self.second_difference_history = {idx: [] for idx in self.all_sample_indices}
        self.stop_epoch, self.stop_score = {}, {}
        self._active_sampler, self._last_finalized_epoch = None, 0
        self._saved_sample_computations = 0

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.history_path = os.path.join(self.args.output_dir, "ies_grad_diff_irreversible_history.jsonl")
        self.state_path = os.path.join(self.args.output_dir, "ies_grad_diff_irreversible_state.json")
        if self.is_world_process_zero():
            open(self.history_path, "w", encoding="utf-8").close()
        self.add_callback(self._Callback(self))

    @staticmethod
    def _model_inputs(batch):
        return {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}

    @staticmethod
    def _token_mean_nll(sequence_nll, labels):
        count = labels[..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
        return sequence_nll / count.to(dtype=sequence_nll.dtype)

    def _get_train_sampler(self):
        self._active_sampler = self._Sampler(sorted(self.active_samples), self.args.seed)
        return self._active_sampler

    def _append_forget_nll(self, indices, values):
        for index, value in zip(indices, values.detach().float().cpu().tolist()):
            index = int(index)
            if index in self.stopped_samples:
                raise RuntimeError(f"Stopped sample {index} appeared in active loader.")
            self.forget_nll_history[index].append(float(value))

    def _finalize_epoch(self):
        epoch = int(round(float(self.state.epoch or 0.0)))
        if epoch <= self._last_finalized_epoch:
            return
        saved = len(self.stopped_samples)
        self._saved_sample_computations += saved
        newly_stopped, records = [], {}
        for index in sorted(self.active_samples):
            history = self.forget_nll_history[index]
            d2, score, stable = None, None, False
            d2_abs_window = []
            if len(history) >= 3:
                d2 = float(history[-1] - 2 * history[-2] + history[-3])
                self.second_difference_history[index].append(d2)
                if len(self.second_difference_history[index]) >= 3:
                    d2_abs_window = [
                        abs(value)
                        for value in self.second_difference_history[index][-3:]
                    ]
                    score = sum(d2_abs_window) / 3.0
                    stable = score <= self.stop_threshold
                    if stable:
                        newly_stopped.append(index)
                        self.stop_epoch[index], self.stop_score[index] = epoch, score
            records[str(index)] = {
                "forget_nll": history[-1] if history else None,
                "forget_nll_window": history[-3:],
                "second_difference": d2,
                "second_difference_abs_window": d2_abs_window,
                "score": score,
                "stable_condition": stable,
                "stopped_now": index in newly_stopped,
            }

        self.stopped_samples.update(newly_stopped)
        self.active_samples = self.all_sample_indices - self.stopped_samples
        if self._active_sampler is not None:
            self._active_sampler.update(sorted(self.active_samples))
            self._active_sampler.set_epoch(epoch)
        self._last_finalized_epoch = epoch
        record = {
            "epoch": epoch,
            "monitoring_signal": "per_sample_length_normalized_nll",
            "second_difference": "m_t - 2*m_t_minus_1 + m_t_minus_2",
            "stop_threshold": self.stop_threshold,
            "score_definition": "mean(abs(latest_3_second_differences))",
            "reactivation_enabled": False,
            "stopped_sample_monitoring": False,
            "active": len(self.active_samples),
            "stopped": len(self.stopped_samples),
            "newly_stopped": newly_stopped,
            "saved_sample_computations_this_epoch": saved,
            "cumulative_saved_sample_computations": self._saved_sample_computations,
            "samples_observed_this_epoch": records,
        }
        if self.is_world_process_zero():
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump({
                    **record,
                    "active_indices": sorted(self.active_samples),
                    "stopped_indices": sorted(self.stopped_samples),
                    "forget_nll_history": self.forget_nll_history,
                    "second_difference_history": self.second_difference_history,
                }, handle, indent=2)
        self.log({
            "ies_grad_diff_active": len(self.active_samples),
            "ies_grad_diff_stopped": len(self.stopped_samples),
            "ies_grad_diff_newly_stopped": len(newly_stopped),
            "ies_grad_diff_saved_this_epoch": saved,
            "ies_grad_diff_saved_total": self._saved_sample_computations,
        })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_batch, retain_batch = inputs["forget"], inputs["retain"]
        if "index" not in forget_batch:
            raise RuntimeError("forget batch has no index; use DataCollatorForSupervisedDatasetwithIndex.")
        forget_inputs = self._model_inputs(forget_batch)
        sequence_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        self._append_forget_nll(
            forget_batch["index"].detach().cpu().tolist(),
            self._token_mean_nll(sequence_nll, forget_inputs["labels"]),
        )
        # This is the original GradDiff forget objective; per-sample NLL is only monitored.
        forget_loss = -forget_outputs.loss
        retain_loss = self.compute_retain_loss(model, self._model_inputs(retain_batch))
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
