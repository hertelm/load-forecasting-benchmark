import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, use_torch_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.use_torch_attention = use_torch_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):

        if self.use_torch_attention:
            # scaled_dot_product_attention expects shape (B, num_heads, L, d)
            # Inputs are (B, L, num_heads, d) so we transpose:
            queries = queries.transpose(1, 2)  # (B, H, L, d)
            keys = keys.transpose(1, 2)        # (B, H, S, d)
            values = values.transpose(1, 2)      # (B, H, S, d)

            # Determine if we are doing causal (e.g. autoregressive) attention.
            # When attn_mask is None and mask_flag is True, we set is_causal=True.
            is_causal = self.mask_flag and (attn_mask is None)

            # ensure attention mask is of the form expected by the built-in function:
            # (either a 2D mask of shape (L, S) or a 4D mask of shape (B, num_heads, L, S)).
            if attn_mask is not None:
                if attn_mask.dim() == 3:  # e.g. (B, L, S)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1, -1)
                # Otherwise, assume it is already in a supported shape.

            # scaled_dot_product_attention automatically applies the 1/sqrt(d_k) scaling and applies dropout
            attn_out = F.multi_head_attention_forward(
                queries, keys, values,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p,
                is_causal=is_causal
            )
            # attn_out has shape (B, H, L, d); transpose back to (B, L, H, d)
            out = attn_out.transpose(1, 2).contiguous()

            # built-in function does not return the attention weights
            return out, None
        
        else:
            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            scale = self.scale or 1. / sqrt(E)

            scores = torch.einsum("blhe,bshe->bhls", queries, keys)

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

            if self.output_attention:
                return V.contiguous(), A
            else:
                return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn