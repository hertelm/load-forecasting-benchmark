"""
This is a modified version of TFT, originally from https://github.com/Nixtla/neuralforecast, licensed under the Apache License 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# import wandb

from ..basemodel import BaseModel

from models.TFT.encoder import TemporalCovariateEncoder, StaticCovariateEncoder
from models.TFT.decoder import TemporalFusionDecoder
from models.TFT.embedding import TFTEmbedding


class TFT(BaseModel):
    """TFT

    The Temporal Fusion Transformer architecture (TFT) is an Sequence-to-Sequence
    model that combines static, historic and future available data to predict an
    univariate target. The method combines gating layers, an LSTM recurrent encoder,
    with and interpretable multi-head attention layer and a multi-step forecasting
    strategy decoder.

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `stat_exog_list`: str list, static continuous columns.<br>
    `hist_exog_list`: str list, historic continuous columns.<br>
    `futr_exog_list`: str list, future continuous columns.<br>
    `hidden_size`: int, units of embeddings and encoders.<br>
    `dropout`: float (0, 1), dropout of inputs VSNs.<br>
    `n_head`: int=4, number of attention heads in temporal fusion decoder.<br>
    `attn_dropout`: float (0, 1), dropout of fusion decoder's attention layer.<br>
    `shared_weights`: bool, If True, all blocks within each stack will share parameters. <br>
    `activation`: str, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int, number of different series in each batch.<br>
    `windows_batch_size`: int=None, windows sampled from rolled data, default uses all.<br>
    `inference_windows_batch_size`: int=-1, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random seed initialization for replicability.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    - [Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister,
    "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting"](https://www.sciencedirect.com/science/article/pii/S0169207021000637)
    """

    # Class attributes
    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        h: int = 96,
        input_size: int = 168,
        tgt_size: int = 1,
        # stat_exog_list=None,
        # hist_exog_list=None,
        # futr_exog_list=None,
        static_size=None,
        past_feature_size=None,
        future_feature_size=None,
        hidden_size: int = 128,
        n_head: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        loss_function='mse',
        # valid_loss=None,
        # max_steps: int = 1000,
        learning_rate: float = 1e-3,
        # num_lr_decays: int = -1,
        # early_stop_patience_steps: int = -1,
        # val_check_steps: int = 100,
        # batch_size: int = 32,
        # valid_batch_size: Optional[int] = None,
        # windows_batch_size: int = 1024,
        # inference_windows_batch_size: int = 1024,
        # start_padding_enabled=False,
        # step_size: int = 1,
        # scaler_type: str = "robust",
        # num_workers_loader=0,
        # drop_last_loader=False,
        # random_seed: int = 1,
        **kwargs
    ):
        # Inherit BaseWindows class
        super(TFT, self).__init__(
            input_dim=input_size,
            # h=h,
            # input_size=input_size,
            loss_function=loss_function,
            # valid_loss=valid_loss,
            # max_steps=max_steps,
            # learning_rate=learning_rate,
            # num_lr_decays=num_lr_decays,
            # early_stop_patience_steps=early_stop_patience_steps,
            # val_check_steps=val_check_steps,
            # batch_size=batch_size,
            # valid_batch_size=valid_batch_size,
            # windows_batch_size=windows_batch_size,
            # inference_windows_batch_size=inference_windows_batch_size,
            # start_padding_enabled=start_padding_enabled,
            # step_size=step_size,
            # scaler_type=scaler_type,
            # num_workers_loader=num_workers_loader,
            # drop_last_loader=drop_last_loader,
            # random_seed=random_seed,
            **kwargs
        )
        self.save_hyperparameters()

        self.example_length = input_size + h

        self.learning_rate = learning_rate
        self.input_size = input_size

        # Parse lists hyperparameters
        # self.stat_exog_list = [] if stat_exog_list is None else stat_exog_list
        # self.hist_exog_list = [] if hist_exog_list is None else hist_exog_list
        # self.futr_exog_list = [] if futr_exog_list is None else futr_exog_list

        self.futr_input_size = max(future_feature_size, 1)
        self.hist_input_size = past_feature_size
        self.static_size = static_size

        # stat_input_size = len(self.stat_exog_list)
        # futr_input_size = max(len(self.futr_exog_list), 1)
        # hist_input_size = len(self.hist_exog_list)

        num_historic_vars = self.futr_input_size + self.hist_input_size + tgt_size

        # ------------------------------- Encoders -----------------------------#
        self.embedding = TFTEmbedding(
            hidden_size=hidden_size,
            stat_input_size=self.static_size,
            futr_input_size=self.futr_input_size,
            hist_input_size=self.hist_input_size,
            tgt_size=tgt_size,
        )

        self.static_encoder = StaticCovariateEncoder(
            hidden_size=hidden_size, num_static_vars=static_size, dropout=dropout
        )

        self.temporal_encoder = TemporalCovariateEncoder(
            hidden_size=hidden_size,
            num_historic_vars=num_historic_vars,
            num_future_vars=self.futr_input_size,
            dropout=dropout,
        )

        # ------------------------------ Decoders -----------------------------#
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            n_head=n_head,
            hidden_size=hidden_size,
            example_length=self.example_length,
            encoder_length=self.input_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )

        # Adapter with Loss dependent dimensions
        self.output_adapter = nn.Linear(
            in_features=hidden_size, out_features=self.loss_function.outputsize
        )

    def forward(self, batch):
        x, *_ = batch
        stat_exog = x['static']
        if stat_exog.shape[-1] == 0:
            stat_exog = None
        hist_exog = x['past_features']
        if hist_exog.shape[-1] == 0:
            hist_exog = None
        futr_exog = x['future_features']
        if futr_exog.shape[-1] == 0:
            futr_exog = None
        x = x['past_target']

        x = x.squeeze(-1)
        # Parsiw windows_batch
        y_insample = x[:, :, None]
        # windows_batch["insample_y"][:, :, None]  # <- [B,T,1]
        # futr_exog = windows_batch["futr_exog"]
        # hist_exog = windows_batch["hist_exog"]
        # stat_exog = windows_batch["stat_exog"]

        if futr_exog is None:
            futr_exog = y_insample[:, [-1]]
            futr_exog = futr_exog.repeat(1, self.example_length, 1)


        s_inp, k_inp, o_inp, t_observed_tgt = self.embedding(
            target_inp=y_insample,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
        )

        # -------------------------------- Inputs ------------------------------#
        # Static context
        if s_inp is not None:
            cs, ce, ch, cc = self.static_encoder(s_inp)
            ch, cc = ch.unsqueeze(0), cc.unsqueeze(0)  # LSTM initial states
        else:
            # If None add zeros
            batch_size, example_length, target_size, hidden_size = t_observed_tgt.shape
            cs = torch.zeros(size=(batch_size, hidden_size)).to(y_insample.device)
            ce = torch.zeros(size=(batch_size, hidden_size)).to(y_insample.device)
            ch = torch.zeros(size=(1, batch_size, hidden_size)).to(y_insample.device)
            cc = torch.zeros(size=(1, batch_size, hidden_size)).to(y_insample.device)

        # Historical inputs
        _historical_inputs = [
            k_inp[:, : self.input_size, :],
            t_observed_tgt[:, : self.input_size, :],
        ]
        if o_inp is not None:
            _historical_inputs.insert(0, o_inp[:, : self.input_size, :])
        historical_inputs = torch.cat(_historical_inputs, dim=-2)

        # Future inputs
        future_inputs = k_inp[:, self.input_size :]

        # ---------------------------- Encode/Decode ---------------------------#
        # Embeddings + VSN + LSTM encoders
        temporal_features = self.temporal_encoder(
            historical_inputs=historical_inputs,
            future_inputs=future_inputs,
            cs=cs,
            ch=ch,
            cc=cc,
        )

        # Static enrichment, Attention and decoders
        temporal_features = self.temporal_fusion_decoder(
            temporal_features=temporal_features, ce=ce
        )

        # Adapt output to loss
        y_hat = self.output_adapter(temporal_features)
        # y_hat = self.loss.domain_map(y_hat)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
