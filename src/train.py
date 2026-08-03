import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything



class PairedForgetRetainDataset:
    def __init__(self, forget_dataset, retain_dataset):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset

    def __len__(self):
        return len(self.forget_dataset)

    def __getitem__(self, idx):
        return {
            "forget": self.forget_dataset[idx],
            "retain": self.retain_dataset[idx % len(self.retain_dataset)],
        }


class PairedForgetRetainCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        if (
            isinstance(features, list)
            and len(features) > 0
            and isinstance(features[0], dict)
            and "forget" in features[0]
            and "retain" in features[0]
        ):
            forget_features = [x["forget"] for x in features]
            retain_features = [x["retain"] for x in features]
            return {
                "forget": self.base_collator(forget_features),
                "retain": self.base_collator(retain_features),
            }

        return self.base_collator(features)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    # Some unlearning data configs return {forget, retain} instead of {train, eval}.
    # For BDSI/NPO-style unlearning, wrap them into an indexable paired dataset.
    train_dataset = data.get("train", None)
    data_collator = collator

    if train_dataset is None and ("forget" in data and "retain" in data):
        train_dataset = PairedForgetRetainDataset(data["forget"], data["retain"])
        data_collator = PairedForgetRetainCollator(collator)

    eval_dataset = data.get("eval", None)

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
