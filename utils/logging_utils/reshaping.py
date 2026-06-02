import torch
import pandas as pd


def reshape_for_quantiles(config, y_hat, y):
    """
    Reshape the y_hat and y arrays for quantile loss.

    Arguments:
        config: Configuration object containing model and sample information.
        y_hat, y: Predictions and true values to reshape.

    Returns:
        Reshaped y_hat and y arrays.
    """
    y_hat = torch.cat(y_hat).reshape(-1, config.sample.target_length, len(config.model.quantiles))

    if config.sample.window_size is not None:
        y_hat = y_hat.repeat_interleave(config.sample.window_size, dim=1)

    y = torch.cat(y).reshape(-1, config.sample.target_length * config.sample.window_size)

    y_hat_median = pd.DataFrame(y_hat[:, :, len(config.model.quantiles) // 2].numpy())
    y_df = pd.DataFrame(y.numpy())

    return y_hat, y, y_hat_median, y_df


def reshape_for_classic_loss(config, y_hat, y):
    """
    Reshape the y_hat and y arrays for classic loss functions.

    Arguments:
        config: Configuration object containing model and sample information.
        y_hat, y: Predictions and true values to reshape.

    Returns:
        Reshaped y_hat and y arrays as Pandas DataFrames.
    """
    y_hat = pd.DataFrame(
        torch.cat(y_hat)
        .reshape(-1, config.sample.target_length)
        .numpy()
    )
    y = pd.DataFrame(
        torch.cat(y)
        .reshape(-1, config.sample.target_length)
        .numpy()
    )
    return y_hat, y
