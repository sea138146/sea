import torch

from trainer.unlearn.sample_early_stop_npo_marginal_ratio import (
    SampleEarlyStopNPOMarginalRatio,
)
from trainer.utils import compute_wga_loss


class SampleEarlyStopWGAMarginalRatio(SampleEarlyStopNPOMarginalRatio):
    """WGA with the same marginal-NLL controller used by current NPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nll_gain_log_prefix = "[WGA-MARGINAL-RATIO]"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        retain_inputs = self._gain_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(
            model=model, retain_inputs=retain_inputs
        )

        forget_batch = inputs.get("forget")
        forget_outputs = None
        forget_loss = None
        forget_scale = 0.0
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
                active_batch = self._gain_slice_batch(forget_batch, positions)
                forget_inputs = self._gain_model_inputs(active_batch)
                forget_loss, forget_outputs = compute_wga_loss(
                    model=model, inputs=forget_inputs, beta=self.beta
                )
                active_count = len(active_positions)
                active_valid_tokens = int(
                    forget_inputs["labels"][..., 1:].ne(-100).sum().item()
                )
                # Restore the original baseline batch denominator while
                # stopped tokens contribute zero numerator.
                forget_scale = active_valid_tokens / max(
                    original_valid_tokens, 1
                )
                forget_loss = forget_loss * forget_scale

        loss = self.alpha * retain_loss
        log_values = {
            "retain_loss": float(retain_loss.detach()),
            "wga_marginal_ratio_active_batch": active_count,
        }
        if forget_loss is not None:
            loss = loss + self.gamma * forget_loss
            log_values["forget_loss"] = float(forget_loss.detach())
            log_values["wga_marginal_ratio_forget_scale"] = forget_scale
        self.log(log_values)
        return (loss, forget_outputs) if return_outputs else loss
