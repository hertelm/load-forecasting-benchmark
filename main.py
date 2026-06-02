import os
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from utils.main_utils.logging_utils import (
    setup_logging,
    setup_wandb_logger,
)
from utils.main_utils.train_test_utils import (
    setup_environment,
    build_callbacks,
    build_trainer,
    run_training,
    run_testing
)
from utils.load_model import load_model
from dataset.dataset import TimeSeriesDataModule


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Create ouput directory path
    base = to_absolute_path(cfg.output_directory)
    today = datetime.today().strftime("%Y-%m-%d")
    output_dir = os.path.join(base, today)
    output_dir = os.path.join(output_dir, cfg.model.wandb_name.replace(":", ""))

    # Setup
    setup_logging(output_dir)
    setup_environment(cfg)

    # Logger + callbacks + trainer
    wandb_logger = setup_wandb_logger(cfg, output_dir)
    callbacks, ckpt_cb = build_callbacks(cfg, output_dir)
    trainer = build_trainer(cfg, wandb_logger, callbacks)

    # Data & model
    dm = TimeSeriesDataModule(cfg)
    model = load_model(cfg)

    # Train + test
    run_training(trainer, model, dm, cfg)
    run_testing(trainer, model, dm, cfg, ckpt_cb)


if __name__ == "__main__":
    main()
