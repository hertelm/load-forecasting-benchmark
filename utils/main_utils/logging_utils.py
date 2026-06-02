import os
import logging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf
from utils.load_model import load_model

SLURM_VARS = [
    "SLURM_JOB_ID",
    "SLURM_NTASKS",
    "SLURM_CPUS_ON_NODE",
    "SLURM_NODELIST",
    "SLURM_PROCID",
]

def model_summary(model):
    """
    Summarize the model parameter count and size.
    """
    model_summary = ModelSummary(model)
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size
    return trainable_parameters, model_size

def flatten_config(d, sep="."):
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_config(v, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)

def setup_logging(output_dir: str) -> None:
    """Configure console + file logging, and dump SLURM vars if present."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    # root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)
    for var in SLURM_VARS:
        val = os.environ.get(var, "<not set>")
        logger.info(f"{var}: {val}")

def setup_wandb_logger(cfg: DictConfig, output_dir: str) -> WandbLogger:
    """
    Instantiate WandbLogger, log model summary, and write resolved config.
    Returns the logger for Trainer().
    """
    # flatten for JSON-safe config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    flat_cfg = flatten_config(cfg_dict)

    wandb_logger = WandbLogger(
        name=cfg.model.wandb_name,
        project=cfg.model.wandb_project,
        save_dir=output_dir,
        config=flat_cfg,
    )

    # push model‐size info
    trainable_params, model_size = model_summary(load_model(cfg))
    wandb_logger.experiment.config.update({
        "Trainable params": trainable_params,
        "Model size (MB)": model_size,
    })
    
    wandb_logger.experiment.define_metric("val_loss_epoch", summary="min")

    # persist the final config.yaml alongside runs
    cfg_save = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config=cfg, f=cfg_save)
    logging.getLogger(__name__).info(f"Config saved to {cfg_save}")

    return wandb_logger
