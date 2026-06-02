from typing import Union, List, Iterable

import torch
import torch.nn as nn
import math
import numpy as np

from ..basemodel import BaseModel


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    without dropout
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1), :]
        return x


class Transformer(BaseModel):
    def __init__(self,
                 input_length: int,
                 target_length: int,
                 architecture: str,
                 past_feature_size: int,
                 future_feature_size: int,
                 static_feature_size: int,
                 patch_size: int,
                 conv_kernel_size: Union[Iterable[int], int],
                 conv_max_pooling: int,
                 lstm_layers: int,
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 attention: str,
                 max_pooling: int,
                 dense_units: Union[Iterable[int], int],
                 num_dense_layers: int,
                 dropout: float,
                 optimizer: str,
                 lr: float,
                 loss_function: str,
                 **kwargs):
        super(Transformer, self).__init__(
            input_dim=input_length,
            loss_function=loss_function,
            **kwargs
        )
        print("initialize Transformer model")
        self.save_hyperparameters()
        self.input_length = input_length
        self.target_length = target_length
        self.architecture = architecture
        self.encoder_input_dim = past_feature_size + future_feature_size + static_feature_size + 1
        self.decoder_input_dim = max(future_feature_size + static_feature_size, 1)
        self.patch_size = patch_size
        if not isinstance(conv_kernel_size, Iterable):
            conv_kernel_size = [conv_kernel_size]
        self.conv_kernel_size = conv_kernel_size
        self.conv_max_pooling = conv_max_pooling
        self.lstm_layers = lstm_layers
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention = attention
        self.max_pooling = max_pooling
        self.dense_units = dense_units
        self.num_dense_layers = num_dense_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self._init_model()

    def _init_model(self):
        self.encoder_embedding = nn.Sequential()
        for i, kernel_size in enumerate(self.conv_kernel_size):
            in_channels = self.encoder_input_dim * self.patch_size if i == 0 else self.d_model
            enc_conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.d_model,
                kernel_size=kernel_size,
                padding='same'
            )
            self.encoder_embedding.append(enc_conv_layer)
            self.encoder_embedding.append(nn.ReLU())
            if self.conv_max_pooling > 1:
                self.encoder_embedding.append(nn.MaxPool1d(kernel_size=self.conv_max_pooling, ceil_mode=True))
        if self.architecture != "encoder":
            self.decoder_embedding = nn.Sequential()
            for i, kernel_size in enumerate(self.conv_kernel_size):
                in_channels = self.decoder_input_dim if i == 0 else self.d_model
                dec_conv_layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.d_model,
                    kernel_size=kernel_size,
                    padding='same'
                )
                self.decoder_embedding.append(dec_conv_layer)
                self.decoder_embedding.append(nn.ReLU())
        if self.lstm_layers > 0:
            self.encoder_lstm = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=True,
                proj_size=self.d_model // 2,
                dropout=self.dropout
            )
            if self.architecture != "encoder":
                self.decoder_lstm = nn.LSTM(
                    input_size=self.d_model,
                    hidden_size=self.d_model,
                    num_layers=self.lstm_layers,
                    batch_first=True,
                    bidirectional=True,
                    proj_size=self.d_model // 2,
                    dropout=self.dropout
                )
        self.positional_encoding = PositionalEncoding(d_model=self.d_model,
                                                      max_len=self.input_length + self.target_length)
        self.encoder_layers = nn.ModuleList()
        for _layer in range(self.num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_model,
                dropout=self.dropout,
                batch_first=True
            )
            self.encoder_layers.append(encoder_layer)
            layer_norm = nn.LayerNorm(self.d_model)
            self.encoder_layers.append(layer_norm)
            if self.max_pooling > 1:
                self.encoder_layers.append(nn.MaxPool1d(kernel_size=self.max_pooling, ceil_mode=True))
        if self.architecture != "encoder" and self.num_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_model,
                dropout=self.dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers, layer_norm)
        self.dense_layers, hidden_dim = self._create_dense_layers()
        target_dim = self.target_length if self.architecture == "encoder" else 1
        self.output_layer = nn.Linear(hidden_dim, target_dim)  # TODO: adapt the number of output units for quantile loss
        if self.attention == 'sparse':
            sparse_attention_matrix = self._get_sparse_attention_matrix()
            self.encoder_mask = nn.Buffer(sparse_attention_matrix[:self.input_length, :self.input_length])
            self.decoder_mask = nn.Buffer(sparse_attention_matrix[:self.target_length, :self.target_length])
        else:
            self.encoder_mask = self.decoder_mask = None

    def _create_dense_layers(self):
        dense_layers = nn.ModuleList()
        hidden_dim = self._get_decoder_output_dim()
        if isinstance(self.dense_units, Iterable):
            dense_units = self.dense_units
        else:
            dense_units = [self.dense_units] * self.num_dense_layers
        for i, units in enumerate(dense_units):
            dense_layers.append(nn.Linear(hidden_dim, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(self.dropout))
            hidden_dim = units
        return dense_layers, hidden_dim

    def _get_sparse_attention_matrix(self):
        mask_len = max(self.input_length, self.target_length)
        mask = torch.Tensor(mask_len, mask_len).fill_(float('-inf'))
        for i in range(0, mask_len):
            mask[i, i] = 0
            if i > 0:
                n_prev = min(i, mask_len)
                for j in range(0, int(np.log2(n_prev)) + 1):
                    mask[i, i - 2**j] = 0
        return mask

    def _prepare_encoder_decoder_input(self, x):
        batch_size = x["past_target"].shape[0]
        max_len = max(self.input_length, self.target_length)
        static_expanded = x["static"][:, None, :].expand(-1, max_len, -1)
        if x["future_features"].shape[1] == 0:
            encoder_input = x["past_target"]
            decoder_input = torch.zeros(size=(batch_size, self.target_length, 1), device=x["past_target"].device)
        else:
            encoder_input = torch.concat(
                (
                    x["past_target"],
                    x["future_features"][:, :self.input_length, :],
                    static_expanded[:, :self.input_length, :]
                ),  # TODO: add past features
                dim=2
            )
            decoder_input = torch.concat(
                (
                    x["future_features"][:, -self.target_length:, :],
                    static_expanded[:, :self.target_length, :]
                ),
                dim=2
            )
        if self.architecture == "encoder":
            padded_decoder_input = torch.zeros(size=(batch_size, self.target_length, decoder_input.shape[2] + 1), device=decoder_input.device)
            padded_decoder_input[:, :, -decoder_input.shape[2]:] = decoder_input
            encoder_input = torch.concat((encoder_input, padded_decoder_input), dim=1)
            decoder_input = None
        if self.patch_size > 1:
            encoder_input = encoder_input.reshape(batch_size, encoder_input.shape[1] // self.patch_size, -1)
        return encoder_input, decoder_input

    def _get_decoder_output_dim(self):
        if self.architecture == "encoder":
            seq_len = (self.input_length + self.target_length) // self.patch_size
            if self.conv_max_pooling > 1:
                for _ in range(len(self.conv_kernel_size)):
                    seq_len = math.ceil(seq_len / self.conv_max_pooling)
            if self.max_pooling > 1:
                for _ in range(self.num_layers):
                    seq_len = math.ceil(seq_len / self.max_pooling)
            vector_dim = self.d_model if len(self.conv_kernel_size) > 0 else self.encoder_input_dim
            return seq_len * vector_dim
        else:
            return self.d_model

    def forward(self, batch):
        x, *_ = batch
        enc_in, dec_in = self._prepare_encoder_decoder_input(x)
        #print(f"enc_in shape: {enc_in.shape}, dec_in shape: {dec_in.shape if dec_in else dec_in}")
        enc_out, (enc_h, enc_c) = self._run_encoder(enc_in)
        #print(f"enc_out shape: {enc_out.shape}")
        if self.architecture == "encoder":
            dec_out = enc_out.flatten(start_dim=1)
        else:
            dec_out = self._run_decoder(enc_out, dec_in, enc_h, enc_c)
        #print(f"dec_out shape: {dec_out.shape}")
        for layer in self.dense_layers:
            dec_out = layer(dec_out)
        y_pred = self.output_layer(dec_out)
        return y_pred

    def _run_encoder(self, enc_in):
        enc_in = enc_in.permute(0, 2, 1)
        enc_out = self.encoder_embedding(enc_in)
        enc_out = enc_out.permute(0, 2, 1)
        if self.lstm_layers > 0:
            enc_out, (enc_h, enc_c) = self.encoder_lstm(enc_out)
        else:
            enc_h = enc_c = None
        if self.num_layers > 0:
            enc_out = self.positional_encoding(enc_out)
        for layer in self.encoder_layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                in_len = enc_out.shape[1]
                mask = self.encoder_mask[:in_len, :in_len] if self.attention == 'sparse' else None
                enc_out = layer(enc_out, src_mask=mask)
            elif isinstance(layer, nn.MaxPool1d):
                enc_out = enc_out.permute(0, 2, 1)
                enc_out = layer(enc_out)
                enc_out = enc_out.permute(0, 2, 1)
            else:
                enc_out = layer(enc_out)
        #enc_out = self.encoder(enc_out, mask=self.encoder_mask)
        return enc_out, (enc_h, enc_c)

    def _run_decoder(self, enc_out, dec_in, enc_h, enc_c):
        dec_in = dec_in.permute(0, 2, 1)
        dec_out = self.decoder_embedding(dec_in)
        dec_out = dec_out.permute(0, 2, 1)
        if self.lstm_layers > 0:
            dec_out, _ = self.decoder_lstm(dec_out, (enc_h, enc_c))
        if self.num_layers > 0:
            dec_out = self.positional_encoding(dec_out)
            dec_out = self.decoder(dec_out, memory=enc_out, tgt_mask=self.decoder_mask)
        return dec_out

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
