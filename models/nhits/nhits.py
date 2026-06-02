from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basemodel import BaseModel

class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # knots = knots[:,None,:]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros(
                (len(knots), self.forecast_size), device=knots.device
            )
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSBlock(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        feature_lookback: int,
        feature_lookahead: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        # pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = (
            pooled_hist_size
            + hist_input_size * feature_lookback
            + futr_input_size * (feature_lookback + feature_lookahead)
            + stat_input_size
        )

        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension [B, L] → [B, 1, L]
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y) # → [B, 1, L_p], where L_p = ceil(L / k)
        insample_y = insample_y.squeeze(1) # → [B, L_p]

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            # hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C_h] → [B, C_h, L]
            # hist_exog = self.pooling_layer(hist_exog) # → [B, C_h, L_p]
            # hist_exog = hist_exog.permute(0, 2, 1)  # → [B, L_p, C_h]
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            ) # → [B, L_p * C_h] concat → [B, L_p + L_p*C_h]

        if self.futr_input_size > 0:
            # futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            # futr_exog = self.pooling_layer(futr_exog)
            # futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class NHITS(BaseModel):
    """NHITS

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.

    **References:**<br>
    -[Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza,
    Max Mergenthaler-Canseco, Artur Dubrawski (2023). "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting".
    Accepted at the Thirty-Seventh AAAI Conference on Artificial Intelligence.](https://arxiv.org/abs/2201.12886)
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        futr_feature_size,
        hist_feature_size,
        stat_feature_size,
        feature_lookback,
        feature_lookahead,
        loss_function,    
        optimizer,
        lr,
        lr_scheduler,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta=0.0,
        activation="ReLU",

        **kwargs,
    ):
        super(NHITS, self).__init__(
            loss_function=loss_function,
            **kwargs,
        )
        self.save_hyperparameters()

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

        self.pred_len = pred_len

        # Architecture
        blocks = self.create_stack(
            h=pred_len,
            input_size=seq_len,
            futr_input_size=futr_feature_size,
            hist_input_size=hist_feature_size,
            stat_input_size=stat_feature_size,
            feature_lookback=feature_lookback,
            feature_lookahead=feature_lookahead,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(
        self,
        h,
        input_size,
        futr_input_size,
        hist_input_size,
        stat_input_size,
        feature_lookback,
        feature_lookahead,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + self.loss_function.outputsize * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.loss_function.outputsize,
                    interpolation_mode=interpolation_mode,
                )

                nbeats_block = NHITSBlock(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    feature_lookback=feature_lookback,
                    feature_lookahead=feature_lookahead,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, batch):

        # Parse windows_batch
        # insample_y = windows_batch["insample_y"].squeeze(-1).contiguous()
        # insample_mask = windows_batch["insample_mask"].squeeze(-1).contiguous()
        # futr_exog = windows_batch["futr_exog"]
        # hist_exog = windows_batch["hist_exog"]
        # stat_exog = windows_batch["stat_exog"]
        
        x_dict, *_ = batch

        # unpack exogenous / static; shapes shown for non-empty
        stat_exog = x_dict.get("static", None)            # [B, S] or []
        hist_exog = x_dict.get("past_features", None)     # [B, H_hist, F_p] or []
        futr_exog = x_dict.get("future_features", None)   # [B, H_fut, F_f] or []

        # turn empty-feature tensors into None
        if stat_exog is not None and stat_exog.shape[-1] == 0:
            stat_exog = None
        if hist_exog is not None and hist_exog.shape[-1] == 0:
            hist_exog = None
        if futr_exog is not None and futr_exog.shape[-1] == 0:
            futr_exog = None    

        # core input sequence
        x = x_dict["past_target"].squeeze(-1)  # [B, L, 1] -> [B, L]
        
        # insample
        residuals = x.flip(dims=(-1,))  # backcast init
        # insample_mask = insample_mask.flip(dims=(-1,))

        forecast = x[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.pred_len, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) # * insample_mask
            forecast = forecast + block_forecast

            # if self.decompose_forecast:
            #     block_forecasts.append(block_forecast)

        # if self.decompose_forecast:
        #     # (n_batch, n_blocks, h, output_size)
        #     block_forecasts = torch.stack(block_forecasts)
        #     block_forecasts = block_forecasts.permute(1, 0, 2, 3)
        #     block_forecasts = block_forecasts.squeeze(-1)  # univariate output
        #     return block_forecasts
        # else:
        return forecast
        
    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Define scheduler 
        if self.lr_scheduler == 'no_scheduler':
            return optimizer
        elif self.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        
            # Return both optimizer and scheduler
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # or 'step' for step-wise decay
                    'frequency': 1  # How often to apply the scheduler (every epoch in this case)
                }
            }