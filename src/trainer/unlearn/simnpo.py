import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    """SIMNPO with optional per-sample normalized-NLL trajectory logging."""

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
        delta=0.0,
        beta=1.0,
        log_per_sample_normalized_nll=False,
        initial_nll_cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta
        self.log_per_sample_normalized_nll = bool(
            log_per_sample_normalized_nll
        )
        self.initial_nll_cache_path = (
            os.path.abspath(os.path.expanduser(initial_nll_cache_path))
            if initial_nll_cache_path
            else None
        )

        self.simnpo_sample_nll_log_path = None
        self.simnpo_sample_nll_state_path = None
        self.simnpo_initial_state_path = None
        self.simnpo_initial_sample_nll = {}
        self.simnpo_sample_nll_history = {}
        self.simnpo_nll_gain_history = {}

        if self.log_per_sample_normalized_nll:
            if self.accelerator.num_processes != 1:
                raise RuntimeError(
                    "Per-sample SIMNPO logging currently supports one process."
                )
            if not hasattr(self.train_dataset, "forget"):
                raise RuntimeError(
                    "Per-sample SIMNPO logging requires train_dataset.forget."
                )
            self.forget_dataset = self.train_dataset.forget
            self.num_forget_samples = len(self.forget_dataset)
            self.all_sample_indices = set(range(self.num_forget_samples))
            if not self.all_sample_indices:
                raise RuntimeError("forget dataset must be non-empty")

            self.simnpo_sample_nll_history = {
                index: [] for index in self.all_sample_indices
            }
            self.simnpo_nll_gain_history = {
                index: [] for index in self.all_sample_indices
            }

            os.makedirs(self.args.output_dir, exist_ok=True)
            self.simnpo_sample_nll_log_path = os.path.join(
                self.args.output_dir,
                "simnpo_sample_normalized_nll.jsonl",
            )
            self.simnpo_sample_nll_state_path = os.path.join(
                self.args.output_dir,
                "simnpo_sample_nll_state.json",
            )
            self.simnpo_initial_state_path = os.path.join(
                self.args.output_dir,
                "simnpo_sample_nll_initial_state.json",
            )
            if self.is_world_process_zero():
                open(
                    self.simnpo_sample_nll_log_path,
                    "w",
                    encoding="utf-8",
                ).close()

            initial_values = self._load_initial_nll_cache()
            cache_hit = initial_values is not None
            if initial_values is None:
                print(
                    "[SIMNPO-SAMPLE-NLL] initial_nll_cache=miss; "
                    "scanning fixed initial model",
                    flush=True,
                )
                initial_observed = self._scan_sample_nll(self.model)
                initial_values = {
                    index: values[0]
                    for index, values in initial_observed.items()
                }
                self._write_initial_nll_cache(initial_values)
            else:
                token_counts = self._scan_valid_token_counts()
                initial_observed = {
                    index: (initial_values[index], token_counts[index])
                    for index in sorted(self.all_sample_indices)
                }

            self.simnpo_initial_sample_nll = initial_values
            self._write_sample_nll_records(
                observed=initial_observed,
                epoch=0.0,
                snapshot_type="initial",
            )
            if self.is_world_process_zero():
                with open(
                    self.simnpo_initial_state_path,
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            **self._initial_nll_payload(initial_values),
                            "source": (
                                "cache" if cache_hit else "model_forward"
                            ),
                            "cache_path": self.initial_nll_cache_path,
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )
            self.add_callback(self._SampleNLLEpochCallback(self))

    @staticmethod
    def _nll_model_inputs(batch):
        return {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "labels")
        }

    @staticmethod
    def _token_mean_nll(sequence_nll, labels):
        # compute_batch_nll returns one token-summed NLL per sequence.
        valid_token_count = (
            labels[..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
        )
        return (
            sequence_nll
            / valid_token_count.to(dtype=sequence_nll.dtype),
            valid_token_count,
        )

    def _forget_scan_loader(self):
        return DataLoader(
            self.forget_dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )

    def _scan_valid_token_counts(self):
        observed = {}
        for batch in self._forget_scan_loader():
            if "index" not in batch:
                raise RuntimeError(
                    "Per-sample SIMNPO logging requires forget sample indices."
                )
            valid_token_count = (
                batch["labels"][..., 1:]
                .ne(-100)
                .sum(dim=-1)
                .clamp_min(1)
                .cpu()
                .tolist()
            )
            indices = batch["index"].cpu().tolist()
            for sample_index, token_count in zip(
                indices, valid_token_count
            ):
                index = int(sample_index)
                if index in observed:
                    raise RuntimeError(
                        f"Duplicate forget sample index: {index}"
                    )
                observed[index] = int(token_count)
        self._validate_observed_indices(observed)
        return observed

    def _scan_sample_nll(self, model=None):
        model = model if model is not None else self.model
        was_training = model.training
        model.eval()
        observed = {}
        try:
            with torch.no_grad():
                for batch in self._forget_scan_loader():
                    if "index" not in batch:
                        raise RuntimeError(
                            "Per-sample SIMNPO logging requires "
                            "forget sample indices."
                        )
                    batch = self._prepare_inputs(batch)
                    model_inputs = self._nll_model_inputs(batch)
                    with self.compute_loss_context_manager():
                        sequence_nll, _ = compute_batch_nll(
                            model, model_inputs
                        )
                    normalized_nll, valid_token_count = (
                        self._token_mean_nll(
                            sequence_nll,
                            model_inputs["labels"],
                        )
                    )
                    indices = batch["index"].detach().cpu().tolist()
                    nll_values = (
                        normalized_nll.detach().float().cpu().tolist()
                    )
                    token_counts = (
                        valid_token_count.detach().cpu().tolist()
                    )
                    for sample_index, nll_value, token_count in zip(
                        indices, nll_values, token_counts
                    ):
                        index = int(sample_index)
                        if index in observed:
                            raise RuntimeError(
                                f"Duplicate forget sample index: {index}"
                            )
                        observed[index] = (
                            float(nll_value),
                            int(token_count),
                        )
        finally:
            model.train(was_training)

        self._validate_observed_indices(observed)
        return observed

    def _validate_observed_indices(self, observed):
        if set(observed) != self.all_sample_indices:
            missing = sorted(self.all_sample_indices - set(observed))
            extra = sorted(set(observed) - self.all_sample_indices)
            raise RuntimeError(
                "SIMNPO NLL scan index mismatch: "
                f"missing={missing}, extra={extra}"
            )

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
            self.initial_nll_cache_path,
            "r",
            encoding="utf-8",
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
        self._validate_observed_indices(observed)
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
            if (
                cached_value is not None
                and cached_value != expected_value
            ):
                raise RuntimeError(
                    f"initial NLL cache {key} mismatch: "
                    f"cached={cached_value!r}, "
                    f"expected={expected_value!r}, "
                    f"path={self.initial_nll_cache_path}"
                )
        print(
            "[SIMNPO-SAMPLE-NLL] initial_nll_cache=hit "
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
        temporary_path = (
            f"{self.initial_nll_cache_path}.tmp.{os.getpid()}"
        )
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                self._initial_nll_payload(observed),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_path, self.initial_nll_cache_path)
        print(
            "[SIMNPO-SAMPLE-NLL] initial_nll_cache=written "
            f"path={self.initial_nll_cache_path}",
            flush=True,
        )

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    def _write_sample_nll_records(
        self,
        observed,
        epoch,
        snapshot_type,
    ):
        records = []
        gains = []
        nll_values = []
        for sample_index in sorted(observed):
            sample_nll, valid_token_count = observed[sample_index]
            initial_sample_nll = self.simnpo_initial_sample_nll[
                sample_index
            ]
            history = self.simnpo_sample_nll_history[sample_index]
            previous_nll = history[-1] if history else None
            nll_change = (
                None
                if previous_nll is None
                else sample_nll - previous_nll
            )
            nll_gain = sample_nll - initial_sample_nll
            history.append(sample_nll)
            self.simnpo_nll_gain_history[sample_index].append(nll_gain)
            record = {
                "snapshot_type": snapshot_type,
                "epoch": float(epoch),
                "global_step": int(self.state.global_step),
                "sample_index": sample_index,
                "valid_token_count": valid_token_count,
                "length_normalized_sample_nll": sample_nll,
                "initial_sample_nll": initial_sample_nll,
                "nll_gain": nll_gain,
                "nll_change_since_previous_epoch": nll_change,
            }
            records.append(record)
            gains.append(nll_gain)
            nll_values.append(sample_nll)
            if self.is_world_process_zero():
                print(
                    "[SIMNPO-SAMPLE-NLL] "
                    "snapshot={snapshot_type} epoch={epoch:.6f} "
                    "step={global_step} index={sample_index} "
                    "valid_tokens={valid_token_count} "
                    "length_normalized_sample_nll="
                    "{length_normalized_sample_nll:.8f} "
                    "initial_sample_nll={initial_sample_nll:.8f} "
                    "nll_gain={nll_gain:.8f} "
                    "nll_change={nll_change_since_previous_epoch}"
                    .format(**record),
                    flush=True,
                )

        summary = {
            "snapshot_type": snapshot_type,
            "epoch": float(epoch),
            "global_step": int(self.state.global_step),
            "num_samples": len(records),
            "mean_length_normalized_sample_nll": (
                sum(nll_values) / len(nll_values)
            ),
            "median_length_normalized_sample_nll": self._median(
                nll_values
            ),
            "mean_nll_gain": sum(gains) / len(gains),
            "median_nll_gain": self._median(gains),
        }

        if self.is_world_process_zero():
            with open(
                self.simnpo_sample_nll_log_path,
                "a",
                encoding="utf-8",
            ) as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
            with open(
                self.simnpo_sample_nll_state_path,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        **summary,
                        "initial_sample_nll": {
                            str(index): value
                            for index, value
                            in self.simnpo_initial_sample_nll.items()
                        },
                        "sample_nll_history": {
                            str(index): values
                            for index, values
                            in self.simnpo_sample_nll_history.items()
                        },
                        "nll_gain_history": {
                            str(index): values
                            for index, values
                            in self.simnpo_nll_gain_history.items()
                        },
                        "latest_samples": {
                            str(record["sample_index"]): record
                            for record in records
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        self.log(
            {
                "simnpo_sample_nll_mean": summary[
                    "mean_length_normalized_sample_nll"
                ],
                "simnpo_nll_gain_mean": summary["mean_nll_gain"],
            }
        )

    def _log_sample_nll_snapshot(
        self,
        model,
        epoch,
        snapshot_type,
    ):
        observed = self._scan_sample_nll(model=model)
        self._write_sample_nll_records(
            observed=observed,
            epoch=epoch,
            snapshot_type=snapshot_type,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        # Strip the sample index before forwarding to the language model.
        forget_inputs = self._nll_model_inputs(inputs["forget"])
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(
            model, forget_inputs
        )
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = (
            -F.logsigmoid(self.beta * forget_loss).mean()
            * 2
            / self.beta
        )

        retain_inputs = self._nll_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs,
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
