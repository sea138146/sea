import copy
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from trainer.utils import compute_batch_nll, compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    sample_nll_log_filename = "grad_diff_sample_normalized_nll.jsonl"
    sample_nll_log_prefix = "[GRADDIFF-SAMPLE-NLL]"
    class _SampleNLLEpochCallback(TrainerCallback):
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
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        log_per_sample_normalized_nll=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.log_per_sample_normalized_nll = bool(
            log_per_sample_normalized_nll
        )
        self.grad_diff_sample_nll_log_path = None
        self.grad_diff_initial_sample_nll = {}
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)
        if self.log_per_sample_normalized_nll:
            self._setup_sample_nll_logging()

    def _setup_sample_nll_logging(self):
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Per-sample GradDiff NLL logging currently supports one process."
            )
        if not hasattr(self.train_dataset, "forget"):
            raise RuntimeError(
                "Per-sample GradDiff logging requires train_dataset.forget."
            )
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.grad_diff_sample_nll_log_path = os.path.join(
            self.args.output_dir,
            self.sample_nll_log_filename,
        )
        if self.is_world_process_zero():
            open(
                self.grad_diff_sample_nll_log_path,
                "w",
                encoding="utf-8",
            ).close()
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

    def _forget_scan_loader(self):
        return DataLoader(
            self.train_dataset.forget,
            batch_size=int(self.args.per_device_train_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )

    def _scan_sample_normalized_nll(self, model):
        was_training = model.training
        model.eval()
        observed = {}
        try:
            with torch.no_grad():
                for batch in self._forget_scan_loader():
                    if "index" not in batch:
                        raise RuntimeError(
                            "Per-sample GradDiff logging requires forget indices."
                        )
                    batch = self._prepare_inputs(batch)
                    model_inputs = self._nll_model_inputs(batch)
                    with self.compute_loss_context_manager():
                        sequence_nll, _ = compute_batch_nll(
                            model,
                            model_inputs,
                        )
                    valid_token_count = (
                        model_inputs["labels"][..., 1:]
                        .ne(-100)
                        .sum(dim=-1)
                        .clamp_min(1)
                    )
                    normalized_nll = sequence_nll / valid_token_count.to(
                        dtype=sequence_nll.dtype
                    )
                    indices = batch["index"].detach().cpu().tolist()
                    nll_values = (
                        normalized_nll.detach().float().cpu().tolist()
                    )
                    token_counts = (
                        valid_token_count.detach().cpu().tolist()
                    )
                    for sample_index, nll_value, token_count in zip(
                        indices,
                        nll_values,
                        token_counts,
                    ):
                        index = int(sample_index)
                        if index in observed:
                            raise RuntimeError(
                                "Duplicate forget sample index in GradDiff "
                                f"NLL scan: {index}"
                            )
                        observed[index] = (
                            float(nll_value),
                            int(token_count),
                        )
        finally:
            model.train(was_training)

        expected_count = len(self.train_dataset.forget)
        if len(observed) != expected_count:
            raise RuntimeError(
                "Per-sample GradDiff NLL scan covered "
                f"{len(observed)} of {expected_count} forget samples."
            )
        return observed

    def _log_sample_nll_snapshot(self, model, epoch, snapshot_type):
        model = model if model is not None else self.model
        observed = self._scan_sample_normalized_nll(model)
        if snapshot_type == "initial":
            self.grad_diff_initial_sample_nll = {
                index: values[0]
                for index, values in observed.items()
            }
        elif set(observed) != set(self.grad_diff_initial_sample_nll):
            raise RuntimeError(
                "GradDiff epoch NLL indices differ from the initial scan."
            )

        records = []
        for sample_index in sorted(observed):
            current_sample_nll, valid_token_count = observed[sample_index]
            initial_sample_nll = self.grad_diff_initial_sample_nll[
                sample_index
            ]
            record = {
                "snapshot_type": snapshot_type,
                "epoch": float(epoch),
                "global_step": int(self.state.global_step),
                "sample_index": sample_index,
                "valid_token_count": valid_token_count,
                "current_sample_nll": current_sample_nll,
                "length_normalized_sample_nll": current_sample_nll,
                "initial_sample_nll": initial_sample_nll,
                "nll_gain": current_sample_nll - initial_sample_nll,
            }
            records.append(record)
            if self.is_world_process_zero():
                print(
                    f"{self.sample_nll_log_prefix} "
                    "snapshot={snapshot_type} epoch={epoch:.6f} "
                    "step={global_step} index={sample_index} "
                    "valid_tokens={valid_token_count} "
                    "initial_sample_nll={initial_sample_nll:.8f} "
                    "current_sample_nll={current_sample_nll:.8f} "
                    "nll_gain={nll_gain:.8f}".format(**record),
                    flush=True,
                )

        if self.is_world_process_zero():
            with open(
                self.grad_diff_sample_nll_log_path,
                "a",
                encoding="utf-8",
            ) as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
