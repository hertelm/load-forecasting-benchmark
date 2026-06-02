import torch.nn as nn
from torch.nn import LayerNorm
from models.TFT.modules import GRN, GLU, InterpretableMultiHeadAttention

class TemporalFusionDecoder(nn.Module):
    def __init__(
        self, n_head, hidden_size, example_length, encoder_length, attn_dropout, dropout
    ):
        super(TemporalFusionDecoder, self).__init__()
        self.encoder_length = encoder_length

        # ------------- Encoder-Decoder Attention --------------#
        self.enrichment_grn = GRN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            context_hidden_size=hidden_size,
            dropout=dropout,
        )
        self.attention = InterpretableMultiHeadAttention(
            n_head=n_head,
            hidden_size=hidden_size,
            example_length=example_length,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.attention_gate = GLU(hidden_size, hidden_size)
        self.attention_ln = LayerNorm(normalized_shape=hidden_size, eps=1e-3)

        self.positionwise_grn = GRN(
            input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
        )

        # ---------------------- Decoder -----------------------#
        self.decoder_gate = GLU(hidden_size, hidden_size)
        self.decoder_ln = LayerNorm(normalized_shape=hidden_size, eps=1e-3)

    def forward(self, temporal_features, ce):
        # ------------- Encoder-Decoder Attention --------------#
        # Static enrichment
        enriched = self.enrichment_grn(temporal_features, c=ce)

        # Temporal self attention
        x, _ = self.attention(enriched, mask_future_timesteps=True)

        # Don't compute historical quantiles
        x = x[:, self.encoder_length :, :]
        temporal_features = temporal_features[:, self.encoder_length :, :]
        enriched = enriched[:, self.encoder_length :, :]

        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        # ---------------------- Decoder ----------------------#
        # Final skip connection
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)

        return x
