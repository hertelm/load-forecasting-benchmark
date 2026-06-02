"""
This is a modified version of TFT, originally from https://github.com/Nixtla/neuralforecast, licensed under the Apache License 2.0.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch.nn import LayerNorm
from models.TFT.modules import GRN, GLU, VariableSelectionNetwork


class TemporalCovariateEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_historic_vars,
            num_future_vars,
            dropout
        ):
        super(TemporalCovariateEncoder, self).__init__()

        self.history_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size,
            num_inputs=num_historic_vars,
            dropout=dropout
        )
        self.history_encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        self.future_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size,
            num_inputs=num_future_vars,
            dropout=dropout
        )
        self.future_encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        # Shared Gated-Skip Connection
        self.input_gate = GLU(hidden_size, hidden_size)
        self.input_gate_ln = LayerNorm(hidden_size, eps=1e-3)

    def forward(self, historical_inputs, future_inputs, cs, ch, cc):
        # [N,X_in,L] -> [N,hidden_size,L]
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))

        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)
        # torch.cuda.synchronize() # this call gives prf boost for unknown reasons

        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)
        return temporal_features

class StaticCovariateEncoder(nn.Module):
    def __init__(self, hidden_size, num_static_vars, dropout):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            hidden_size=hidden_size, num_inputs=num_static_vars, dropout=dropout
        )
        self.context_grns = nn.ModuleList(
            [
                GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
                for _ in range(4)
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        cs, ce, ch, cc = tuple(m(variable_ctx) for m in self.context_grns)

        return cs, ce, ch, cc
