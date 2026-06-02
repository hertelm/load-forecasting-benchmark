
import logging
from omegaconf import DictConfig
import time
import wandb
import torch
import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.tuner import Tuner

from utils.save_results import save_results
from dataset.dataset import TimeSeriesDataModule

def setup_environment(cfg: DictConfig) -> None:
    """Seed RNGs, fix CuDNN for reproducibility, set matmul precision."""
    torch.set_float32_matmul_precision("high") # recommended for Tensor Cores (e.g. A100, H100)
    pl.seed_everything(cfg.model.seed, workers=True)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    logger = logging.getLogger(__name__)
    logger.info("Set torch float32 matmul precision to 'high'.")
    logger.info(f"Seed set to: {cfg.model.seed}")

def build_callbacks(cfg: DictConfig, output_dir: str):
    """Collect all callbacks from config flags; return (list, checkpoint_cb)."""
    cbs = [RichProgressBar()]

    if cfg.model.lr_monitor:
        cbs.append(LearningRateMonitor(logging_interval="step"))

    if cfg.model.early_stopping:
        cbs.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=cfg.model.early_stopping_min_delta,
                patience=cfg.model.early_stopping_patience,
                mode="min",
            )
        )

    ckpt_cb = None
    if cfg.model.checkpointing:
        ckpt_cb = ModelCheckpoint(
            monitor="val_loss",
            dirpath=output_dir,
            filename=f"{cfg.model.wandb_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            save_top_k=1,
            mode="min",
        )
        cbs.append(ckpt_cb)

    return cbs, ckpt_cb


def build_trainer(
    cfg: DictConfig,
    logger: Logger,
    callbacks: list,
) -> Trainer:
    """Instantiate the PyTorch Lightning Trainer."""
    return Trainer(
        logger=logger,
        max_epochs=cfg.model.epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        devices=cfg.gpus,
        strategy=cfg.strategy,
        val_check_interval=cfg.model.val_every_n_steps
    )


def run_training(
    trainer: Trainer,
    model: pl.LightningModule,
    dm: TimeSeriesDataModule,
    cfg: DictConfig,
) -> None:
    """Optionally tune batch size, then fit the model."""
    if cfg.model.batch_size == 1 and cfg.gpus == 1:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="binsearch")
        cfg.model.batch_size = dm.batch_size
        logging.getLogger(__name__).info(f"Auto-scaled batch size to {dm.batch_size}")

    trainer.fit(model, datamodule=dm)
    runtime = time.time() - dm.training_data_preparation_end_time
    logging.getLogger(__name__).info("Training complete.")
    logging.getLogger(__name__).info(f"Training runtime: {runtime:.2f} seconds.")
    wandb.log({"training_time": runtime})


def run_testing(
    trainer: Trainer,
    model: pl.LightningModule,
    dm: TimeSeriesDataModule,
    cfg: DictConfig,
    ckpt_cb: ModelCheckpoint | None,
) -> None:
    """Run predict on end-of-training and/or best-checkpoint, then save results."""
    # optional end-of-training prediction
    if not cfg.model.checkpointing or cfg.model.evaluate_end:
        preds = trainer.predict(model, datamodule=dm)
        save_results(cfg, preds, dm, name="-end")
        logging.getLogger(__name__).info("Saved end-of-training predictions.")

    # best-checkpoint prediction
    if ckpt_cb is not None and ckpt_cb.best_model_path:
        logging.getLogger(__name__).info(f"Loading best checkpoint from {ckpt_cb.best_model_path}")
        best_model = type(model).load_from_checkpoint(ckpt_cb.best_model_path)

        # evaluate model on validation set
        val_preds = trainer.predict(best_model, dataloaders=dm.val_dataloader())
        save_results(cfg, val_preds, dm, name="-val")
        logging.getLogger(__name__).info("Saved best-checkpoint validation predictions.")

        # evaluate model on test set
        preds = trainer.predict(best_model, datamodule=dm)

        save_results(cfg, preds, dm)
        logging.getLogger(__name__).info("Saved best-checkpoint predictions.")

