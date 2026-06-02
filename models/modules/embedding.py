import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, L, N) -> (B, N, L)
        x = self.value_embedding(x) # (B, N, L) -> (B, N, d_model)
        # x: [Batch Variate d_model]
        return self.dropout(x)

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False) # mapping from patch_len to d_model
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_vars, seq_len).

        Returns:
            - Embedded tensor of shape (batch_size * n_vars, num_patches + 1, d_model)
            - The number of variables n_vars
        """
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1)) # (B, n_vars, 1, d_model)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) # (B, n_vars, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # (B * n_vars, num_patches, patch_len)
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) # (B * n_vars, num_patches, d_model)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # (B, n_vars, num_patches, d_model)
        x = torch.cat([x, glb], dim=2) # (B, n_vars, num_patches + 1, d_model)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # Flatten back to (B * n_vars, num_patches + 1, d_model)
        return self.dropout(x), n_vars
