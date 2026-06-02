import torch
from torch import nn, Tensor
from typing import Optional

class TFTEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        stat_input_size,
        futr_input_size,
        hist_input_size,
        tgt_size
    ):
        super().__init__()
        # There are 4 types of input:
        # 1. Static continuous
        # 2. Temporal known a priori continuous
        # 3. Temporal observed continuous
        # 4. Temporal observed targets (time series obseved so far)

        self.hidden_size = hidden_size

        self.stat_input_size = stat_input_size
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.tgt_size = tgt_size

        # Instantiate Continuous Embeddings if size is not None
        for attr, size in [
            ("stat_exog_embedding", stat_input_size),
            ("futr_exog_embedding", futr_input_size),
            ("hist_exog_embedding", hist_input_size),
            ("tgt_embedding", tgt_size),
        ]:
            if size:
                vectors = nn.Parameter(torch.Tensor(size, hidden_size))
                bias = nn.Parameter(torch.zeros(size, hidden_size))
                torch.nn.init.xavier_normal_(vectors)
                setattr(self, attr + "_vectors", vectors)
                setattr(self, attr + "_bias", bias)
            else:
                setattr(self, attr + "_vectors", None)
                setattr(self, attr + "_bias", None)

    def _apply_embedding(
        self,
        cont: Optional[Tensor],
        cont_emb: Tensor,
        cont_bias: Tensor,
    ):
        if cont is not None:
            # the line below is equivalent to following einsums
            # e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            # e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
            e_cont = e_cont + cont_bias
            return e_cont

        return None

    def forward(
            self,
            target_inp,
            stat_exog=None,
            futr_exog=None,
            hist_exog=None
            ):
        # temporal/static categorical/continuous known/observed input
        # tries to get input, if fails returns None

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        stat_exog = stat_exog[:, :] if stat_exog is not None else None

        s_inp = self._apply_embedding(
            cont=stat_exog,
            cont_emb=self.stat_exog_embedding_vectors,
            cont_bias=self.stat_exog_embedding_bias,
        )
        k_inp = self._apply_embedding(
            cont=futr_exog,
            cont_emb=self.futr_exog_embedding_vectors,
            cont_bias=self.futr_exog_embedding_bias,
        )
        o_inp = self._apply_embedding(
            cont=hist_exog,
            cont_emb=self.hist_exog_embedding_vectors,
            cont_bias=self.hist_exog_embedding_bias,
        )

        # Temporal observed targets
        # t_observed_tgt = torch.einsum('btf,fh->btfh',
        #                               target_inp, self.tgt_embedding_vectors)
        target_inp = torch.matmul(
            target_inp.unsqueeze(3).unsqueeze(4),
            self.tgt_embedding_vectors.unsqueeze(1),
        ).squeeze(3)
        target_inp = target_inp + self.tgt_embedding_bias

        return s_inp, k_inp, o_inp, target_inp
