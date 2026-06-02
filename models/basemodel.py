import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from models.loss_functions import MQLoss
from typing import Any, Tuple
import numpy as np


class BaseModel(pl.LightningModule):
    """
    The base model class for all models.
    """

    def __init__(self, loss_function: str, **kwargs: Any):
        """Initialize the model with input dimensions and loss function."""
        super(BaseModel, self).__init__()
        self.loss_function = self._get_loss_function(loss_function, **kwargs)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one training step."""
        x, y, *_ = batch
        y_hat = self(batch)
        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> None:
        """Perform one validation step."""
        x, y, *_ = batch
        y_hat = self(batch)
        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> None:
        """Perform one test step."""
        x, y, *_ = batch
        y_hat = self(batch)
        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def predict_step(
        self,
        batch,
        batch_idx,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform prediction step. Optionally perform interpolation if high-resolution data is provided.
        """
        x, y, columns, *_ = batch
        y_hat = self(batch)
        y = self.trainer.datamodule.target_scaler.inverse_transform(y, columns=columns)

        y_hat = self.trainer.datamodule.target_scaler.inverse_transform(y_hat, columns=columns)
        x["past_target"] = self.trainer.datamodule.target_scaler.inverse_transform(
            x["past_target"],
            columns=columns
        )
        return y_hat, y, x, columns


    def _get_loss_function(self, loss_function: str, **kwargs: Any) -> nn.Module:
        """
        Retrieve the appropriate loss function based on the provided loss name.
        """
        loss_functions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "mqloss": MQLoss(quantiles=kwargs.get("quantiles", [0.1, 0.5, 0.9]), window_size=kwargs.get("window_size")),
        }

        if loss_function not in loss_functions:
            raise NotImplementedError(
                f"Loss function '{loss_function}' not implemented. Choose from {list(loss_functions.keys())}"
            )

        chosen_loss_function = loss_functions[loss_function]

        if not hasattr(chosen_loss_function, "outputsize"):
            chosen_loss_function.outputsize = 1

        return chosen_loss_function
