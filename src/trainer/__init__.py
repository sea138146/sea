import torch
from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.unlearn.sample_early_stop_npo_loss_irreversible import SampleEarlyStopNPOLossIrreversible
from trainer.unlearn.sample_early_stop_grad_diff_irreversible import SampleEarlyStopGradDiffIrreversible
from trainer.unlearn.sample_early_stop_simnpo_irreversible import SampleEarlyStopSimNPOIrreversible
from trainer.unlearn.sample_early_stop_wga_irreversible import SampleEarlyStopWGAIrreversible
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO, SampleEarlyStopNPO
from trainer.unlearn.sample_early_stop_npo_loss import SampleEarlyStopNPOLoss
from trainer.unlearn.sample_stop_npo import SampleStopNPO
from trainer.unlearn.sample_stop_marginal_npo import SampleStopMarginalNPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        if dataset is None:
            dataset_len = 0
        elif isinstance(dataset, dict):
            if "forget" in dataset:
                dataset_len = len(dataset["forget"])
            elif "train" in dataset:
                dataset_len = len(dataset["train"])
            else:
                dataset_len = len(next(iter(dataset.values())))
        else:
            dataset_len = len(dataset)

        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args
# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(SampleEarlyStopNPOLossIrreversible)
_register_trainer(SampleEarlyStopGradDiffIrreversible)
_register_trainer(SampleEarlyStopSimNPOIrreversible)
_register_trainer(SampleEarlyStopWGAIrreversible)
_register_trainer(SampleEarlyStopNPO)
_register_trainer(SampleEarlyStopNPOLoss)
_register_trainer(SampleStopNPO)
_register_trainer(SampleStopMarginalNPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)
