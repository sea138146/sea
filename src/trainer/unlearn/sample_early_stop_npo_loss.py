import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import SampleEarlyStopNPO
from trainer.utils import compute_batch_nll


class SampleEarlyStopNPOLoss(SampleEarlyStopNPO):
    """IES-style dynamic stopping based on unreduced per-sample NPO loss.

    Monitoring signal:
        m_i^(t) = l_NPO,i^(t)

    Curvature:
        d2_i^(t) = m_i^(t) - 2*m_i^(t-1) + m_i^(t-2)

    Stopping:
        abs(d2_i^(t)) <= stop_threshold
        for ``stable_checks`` consecutive valid observations.

    Reactivation:
        abs(d2_i^(t)) >= stop_threshold * rebound_multiplier
        for ``rebound_checks`` consecutive forward-only observations.

    This hysteresis prevents a sample from repeatedly switching between active
    and excluded states because of one noisy observation.
    """

    def __init__(
        self,
        stop_threshold=1.0e-3,
        stable_checks=2,
        rebound_multiplier=2.0,
        rebound_checks=2,
        calibration_only=True,
        *args,
        **kwargs,
    ):
        if float(stop_threshold) < 0:
            raise ValueError("stop_threshold must be non-negative")
        if int(stable_checks) < 1:
            raise ValueError("stable_checks must be >= 1")
        if float(rebound_multiplier) <= 1.0:
            raise ValueError("rebound_multiplier must be > 1")
        if int(rebound_checks) < 1:
            raise ValueError("rebound_checks must be >= 1")

        # Parent class provides the dynamic sampler, remaining_loader,
        # epoch-end callback and basic trajectory containers. Its moving-average
        # partition rule is fully overridden below.
        super().__init__(
            threshold=float(stop_threshold),
            moving_average_rate=1,
            k=1,
            *args,
            **kwargs,
        )

        self.stop_threshold = float(stop_threshold)
        self.stable_checks = int(stable_checks)
        self.rebound_multiplier = float(rebound_multiplier)
        self.rebound_threshold = (
            self.stop_threshold * self.rebound_multiplier
        )
        self.rebound_checks = int(rebound_checks)
        self.calibration_only = bool(calibration_only)

        self.stable_streak = defaultdict(int)
        self.rebound_streak = defaultdict(int)

        # The reference model is frozen, so its sequence-level NLL can be
        # cached once per sample and reused by remaining_loader.
        self.reference_nll_cache = {}

        self.ies_log_path = os.path.join(
            self.args.output_dir,
            "ies_npo_loss_history.jsonl",
        )
        self.ies_state_path = os.path.join(
            self.args.output_dir,
            "ies_npo_loss_state.json",
        )

        if self.is_world_process_zero():
            open(self.ies_log_path, "w", encoding="utf-8").close()

        print(
            "[IES-NPO-Loss] "
            f"stop_threshold={self.stop_threshold}, "
            f"stable_checks={self.stable_checks}, "
            f"rebound_threshold={self.rebound_threshold}, "
            f"rebound_checks={self.rebound_checks}, "
            f"calibration_only={self.calibration_only}"
        )

    def _compute_per_sample_npo(
        self,
        model,
        model_inputs,
        *,
        use_cached_reference=False,
        sample_indices=None,
    ):
        current_nll, outputs = compute_batch_nll(
            model,
            model_inputs,
        )

        if use_cached_reference:
            if sample_indices is None:
                raise ValueError(
                    "sample_indices are required when using cached reference NLL"
                )

            missing = [
                int(idx)
                for idx in sample_indices
                if int(idx) not in self.reference_nll_cache
            ]
            if missing:
                raise RuntimeError(
                    "Missing cached reference NLL for samples: "
                    f"{missing}"
                )

            reference_nll = torch.tensor(
                [
                    self.reference_nll_cache[int(idx)]
                    for idx in sample_indices
                ],
                device=current_nll.device,
                dtype=current_nll.dtype,
            )
        else:
            with torch.no_grad():
                reference_nll, _ = compute_batch_nll(
                    self.ref_model,
                    model_inputs,
                )

            if sample_indices is not None:
                reference_values = (
                    reference_nll
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                for idx, value in zip(
                    sample_indices,
                    reference_values,
                ):
                    self.reference_nll_cache[int(idx)] = float(value)

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

        return (
            per_sample_npo,
            current_nll,
            reference_nll,
            outputs,
        )

    def _monitor_remaining_samples(self, model):
        """Forward-only monitoring for currently excluded samples."""

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
                        "remaining forget batch has no 'index'"
                    )

                batch = self._prepare_inputs(batch)

                sample_indices = (
                    batch["index"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                model_inputs = self._model_inputs(batch)

                with self.compute_loss_context_manager():
                    per_sample_npo, _, _, _ = (
                        self._compute_per_sample_npo(
                            model,
                            model_inputs,
                            use_cached_reference=True,
                            sample_indices=sample_indices,
                        )
                    )

                self._append_sample_losses(
                    sample_indices,
                    per_sample_npo
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                )

                monitored += len(sample_indices)

        if was_training:
            model.train()

        return monitored

    def _refresh_derivatives_and_partition(self):
        """Update second differences, stable streaks and rebound streaks."""

        previous_excluded = set(self.excluded_samples)
        next_excluded = set(previous_excluded)

        newly_excluded = []
        reactivated = []
        sample_records = {}

        for idx in sorted(self.all_sample_indices):
            history = self.sample_loss_history[idx]

            if len(history) < 3:
                sample_records[idx] = {
                    "npo_loss": history[-1] if history else None,
                    "second_difference": None,
                    "score": None,
                    "stable_streak": self.stable_streak[idx],
                    "rebound_streak": self.rebound_streak[idx],
                    "state": (
                        "excluded"
                        if idx in previous_excluded
                        else "active"
                    ),
                }
                continue

            second_difference = float(
                history[-1]
                - 2.0 * history[-2]
                + history[-3]
            )
            score = abs(second_difference)

            self.derivative_history[idx].append(
                second_difference
            )

            if idx in previous_excluded:
                self.stable_streak[idx] = 0

                if score >= self.rebound_threshold:
                    self.rebound_streak[idx] += 1
                else:
                    self.rebound_streak[idx] = 0

                if (
                    not self.calibration_only
                    and self.rebound_streak[idx]
                    >= self.rebound_checks
                ):
                    next_excluded.discard(idx)
                    self.rebound_streak[idx] = 0
                    reactivated.append(idx)

            else:
                self.rebound_streak[idx] = 0

                if score <= self.stop_threshold:
                    self.stable_streak[idx] += 1
                else:
                    self.stable_streak[idx] = 0

                if (
                    not self.calibration_only
                    and self.stable_streak[idx]
                    >= self.stable_checks
                ):
                    next_excluded.add(idx)
                    self.stable_streak[idx] = 0
                    newly_excluded.append(idx)

            sample_records[idx] = {
                "npo_loss": float(history[-1]),
                "npo_loss_window": [
                    float(value)
                    for value in history[-3:]
                ],
                "second_difference": second_difference,
                "score": score,
                "stable_condition": (
                    score <= self.stop_threshold
                ),
                "rebound_condition": (
                    score >= self.rebound_threshold
                ),
                "stable_streak": self.stable_streak[idx],
                "rebound_streak": self.rebound_streak[idx],
                "state_before": (
                    "excluded"
                    if idx in previous_excluded
                    else "active"
                ),
                "state_after": (
                    "excluded"
                    if idx in next_excluded
                    else "active"
                ),
            }

        self.excluded_samples = next_excluded
        self.active_samples = (
            self.all_sample_indices
            - self.excluded_samples
        )

        if self._ies_sampler is not None:
            self._ies_sampler.update(
                sorted(self.active_samples)
            )

        self.remaining_loader = self._build_remaining_loader(
            sorted(self.excluded_samples)
        )

        return (
            newly_excluded,
            reactivated,
            sample_records,
        )

    def _ies_finalize_epoch(self, model=None):
        completed_epoch = int(
            round(float(self.state.epoch or 0.0))
        )

        if completed_epoch <= self._ies_last_finalized_epoch:
            return

        # These are the samples that were actually excluded during the epoch
        # that just finished. Each one received forward-only monitoring and
        # therefore represents one genuinely saved backward observation.
        monitored = self._monitor_remaining_samples(model)

        (
            newly_excluded,
            reactivated,
            sample_records,
        ) = self._refresh_derivatives_and_partition()

        active_count = len(self.active_samples)
        excluded_count = len(self.excluded_samples)

        self._ies_total_saved_backprop_instances += monitored
        self._ies_last_finalized_epoch = completed_epoch

        if self._ies_sampler is not None:
            self._ies_sampler.set_epoch(completed_epoch)

        epoch_record = {
            "epoch": completed_epoch,
            "monitoring_signal": "per_sample_npo_loss",
            "second_difference": (
                "m_t - 2*m_t_minus_1 + m_t_minus_2"
            ),
            "stop_threshold": self.stop_threshold,
            "stable_checks": self.stable_checks,
            "rebound_threshold": self.rebound_threshold,
            "rebound_multiplier": self.rebound_multiplier,
            "rebound_checks": self.rebound_checks,
            "calibration_only": self.calibration_only,
            "active": active_count,
            "excluded": excluded_count,
            "newly_excluded": newly_excluded,
            "reactivated": reactivated,
            "remaining_forward_instances": monitored,
            "cumulative_saved_backprop_instances": (
                self._ies_total_saved_backprop_instances
            ),
            "samples": {
                str(idx): record
                for idx, record in sample_records.items()
            },
        }

        if self.is_world_process_zero():
            with open(
                self.ies_log_path,
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
                "active_indices": sorted(self.active_samples),
                "excluded_indices": sorted(
                    self.excluded_samples
                ),
                "npo_loss_history": {
                    str(idx): [
                        float(value)
                        for value in values
                    ]
                    for idx, values
                    in self.sample_loss_history.items()
                },
                "second_difference_history": {
                    str(idx): [
                        float(value)
                        for value in values
                    ]
                    for idx, values
                    in self.derivative_history.items()
                },
            }

            with open(
                self.ies_state_path,
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
                "ies_npo_active": active_count,
                "ies_npo_excluded": excluded_count,
                "ies_npo_newly_excluded": len(
                    newly_excluded
                ),
                "ies_npo_reactivated": len(reactivated),
                "ies_npo_remaining_forward": monitored,
                "ies_npo_saved_backward_total": (
                    self._ies_total_saved_backprop_instances
                ),
            }
        )

        print(
            f"[IES-NPO-Loss] epoch={completed_epoch} "
            f"active={active_count} "
            f"excluded={excluded_count} "
            f"new={len(newly_excluded)} "
            f"reactivated={len(reactivated)} "
            f"remaining_forward={monitored} "
            f"saved_backward_total="
            f"{self._ies_total_saved_backprop_instances}"
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

        forget_inputs = self._model_inputs(forget_batch)

        (
            per_sample_npo,
            _,
            _,
            forget_outputs,
        ) = self._compute_per_sample_npo(
            model,
            forget_inputs,
            use_cached_reference=False,
            sample_indices=sample_indices,
        )

        # The monitored value is exactly the unreduced loss that is averaged
        # into the NPO forget objective.
        self._append_sample_losses(
            sample_indices,
            per_sample_npo
            .detach()
            .float()
            .cpu()
            .tolist(),
        )

        forget_loss = per_sample_npo.mean()

        retain_inputs = self._model_inputs(retain_batch)
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
