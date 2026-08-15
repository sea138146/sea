import torch
import torch.nn.functional as F

from trainer.unlearn.sample_early_stop_npo_marginal_ratio import (
    SampleEarlyStopNPOMarginalRatio,
)
from trainer.utils import compute_batch_nll


class SampleEarlyStopNPOMarginalRatioDropStopped(
    SampleEarlyStopNPOMarginalRatio
):
    """Marginal-ratio NPO ablation that drops stopped batch slots.

    A stopped slot contributes neither forget loss nor retain loss.
    Forget loss keeps the original sample-batch denominator. Retain loss keeps
    the original valid-token denominator used by the causal-LM loss. Epoch-end
    forward-only monitoring and rebound behavior are inherited unchanged.
    """

    # Keep the forget sample denominator and retain token denominator equal
    # to their respective original-batch baseline denominators.
    # For example, B=4 and A=2 gives (loss1 + loss2) / 4, not / 2.
    normalize_active_losses_by_original_denominators = True

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_batch = inputs.get("forget")
        retain_batch = inputs.get("retain")
        forget_outputs = None

        if forget_batch is None or retain_batch is None:
            raise RuntimeError("both forget and retain batches are required")
        sample_indices = forget_batch["index"].detach().cpu().tolist()
        original_batch_size = len(sample_indices)
        if retain_batch["input_ids"].shape[0] != original_batch_size:
            raise RuntimeError(
                "forget and retain batch sizes must match for slot dropping"
            )
        original_retain_valid_tokens = int(
            retain_batch["labels"][..., 1:].ne(-100).sum().item()
        )
        if original_retain_valid_tokens == 0:
            raise RuntimeError("retain batch has no valid causal-LM tokens")
        active_positions = [
            position
            for position, index in enumerate(sample_indices)
            if int(index) in self.active_samples
        ]
        active_count = len(active_positions)

        if active_count == 0:
            # Keep Trainer.backward valid without running either model side.
            loss = next(model.parameters()).reshape(-1)[0] * 0.0
            self.log(
                {
                    "npo_drop_stopped_active_batch": 0,
                    "npo_drop_stopped_dropped_batch": original_batch_size,
                    "npo_drop_stopped_retain_active_batch": 0,
                    "npo_drop_stopped_active_slot_scale": 0.0,
                    "npo_drop_stopped_retain_active_valid_tokens": 0,
                    "npo_drop_stopped_retain_original_valid_tokens": (
                        original_retain_valid_tokens
                    ),
                    "npo_drop_stopped_retain_scale": 0.0,
                }
            )
            return (loss, forget_outputs) if return_outputs else loss

        positions = torch.tensor(active_positions, dtype=torch.long)
        forget_batch = self._gain_slice_batch(forget_batch, positions)
        retain_batch = self._gain_slice_batch(retain_batch, positions)
        active_retain_valid_tokens = int(
            retain_batch["labels"][..., 1:].ne(-100).sum().item()
        )
        if active_retain_valid_tokens == 0:
            raise RuntimeError("active retain batch has no valid causal-LM tokens")
        retain_token_scale = (
            active_retain_valid_tokens / original_retain_valid_tokens
        )
        forget_inputs = self._gain_model_inputs(forget_batch)
        retain_inputs = self._gain_model_inputs(retain_batch)

        current_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        with torch.no_grad():
            reference_nll, _ = compute_batch_nll(
                self.ref_model, forget_inputs
            )
        forget_loss = (
            -2.0
            / self.beta
            * F.logsigmoid(self.beta * (current_nll - reference_nll))
        ).mean()
        active_slot_scale = 1.0
        if self.normalize_active_losses_by_original_denominators:
            # Equivalent to inserting zero loss for every stopped slot and
            # averaging over the original baseline batch:
            #   sum(active per-sample losses) / original_batch_size.
            active_slot_scale = active_count / original_batch_size
            forget_loss = forget_loss * active_slot_scale
        retain_loss = self.compute_retain_loss(
            model=model, retain_inputs=retain_inputs
        )
        if self.normalize_active_losses_by_original_denominators:
            retain_loss = retain_loss * retain_token_scale
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self.log(
            {
                "forget_loss": float(forget_loss.detach()),
                "retain_loss": float(retain_loss.detach()),
                "npo_drop_stopped_active_batch": active_count,
                "npo_drop_stopped_dropped_batch": (
                    original_batch_size - active_count
                ),
                "npo_drop_stopped_retain_active_batch": active_count,
                "npo_drop_stopped_active_slot_scale": active_slot_scale,
                "npo_drop_stopped_retain_active_valid_tokens": (
                    active_retain_valid_tokens
                ),
                "npo_drop_stopped_retain_original_valid_tokens": (
                    original_retain_valid_tokens
                ),
                "npo_drop_stopped_retain_scale": retain_token_scale,
            }
        )
        return (loss, forget_outputs) if return_outputs else loss
