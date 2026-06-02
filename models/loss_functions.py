import torch
import torch.nn as nn

class MQLoss(nn.Module):
    """
    Multi-Quantile loss
    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.
    """

    def __init__(self, quantiles=None, window_size=None):
        super(MQLoss, self).__init__()
        if quantiles is None or not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles should be a list of values between 0 and 1.")
        qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        # self.register_buffer("quantiles", torch.tensor(quantiles).view(1, 1, -1))
        self.outputsize = len(quantiles)
        self.window_size = window_size

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ):
        """
        Parameters:
        `y`: tensor, Actual values. Dimension: (batch, seq_len).
        `y_hat`: tensor, Predicted values. Dimension: (batch, seq_len, number_of_quantiles/outputsize).

        Returns:
        `mqloss`: tensor (single value).
        """

        if y_hat.shape[-1] != self.outputsize:
            raise ValueError(
                f"Number of quantiles should be equal to the output size of the model. Expected {self.outputsize} quantiles, got {y_hat.shape[-1]}."
            )

        if self.window_size is not None:
            y_hat = y_hat.repeat_interleave(self.window_size, dim=-2)

        if y.shape[-1] != y_hat.shape[-2]:
            raise ValueError(
                f"y and y_hat must have the same sequence length. Got {y.shape} for y and {y_hat.shape} for y_hat."
            )

        error = y_hat - y.unsqueeze(-1)
        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))
        losses = (1 / len(self.quantiles)) * (
            self.quantiles * sq + (1 - self.quantiles) * s1_q
        )
        loss = torch.mean(losses)

        return loss
