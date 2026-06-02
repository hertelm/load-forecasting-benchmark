import numpy as np
import pandas as pd
import torch
import plotly.io as pio
import wandb
from PIL import Image
import io

from utils.logging_utils.plotting import (
    sample_plot,
    quantile_plot,
    multiline_plot,
    bar_plot,
)


def log_deterministic_metrics(config, y, y_hat, columns, name, dm):
    """
    Calculate and log RMSE, MAE, SMAPE, p90, p10, and other deterministic metrics.

    Arguments:
        config: Configuration object containing model information.
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
        dm: data module
    """
    metrics = {
        "RMSE": lambda y, y_hat: np.sqrt(np.mean((y_hat - y) ** 2, axis=0)),
        "MAE": lambda y, y_hat: np.mean(np.abs(y_hat - y), axis=0),
        "MAPE": lambda y, y_hat: np.mean(np.abs((y - y_hat) / y), axis=0),
        "SMAPE": lambda y, y_hat: np.mean(
            2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)), axis=0
        ),
        "p90": lambda y, y_hat: np.mean(
            np.maximum(0.9 * (y - y_hat), 0.1 * (y_hat - y)), axis=0
        ),
        "p10": lambda y, y_hat: np.mean(
            np.maximum(0.1 * (y - y_hat), 0.9 * (y_hat - y)), axis=0
        ),
    }

    for metric_name, metric_fn in metrics.items():
        log_metric_over_time(metric_name, metric_fn(y, y_hat), name)

    #log_nadir_error(y, y_hat, name)
    #log_rocof_error(y, y_hat, name)
    log_sample_mae(config, y, y_hat, name)
    log_sample_mse(config, y, y_hat, name)
    log_normalized_metrics(config, y, y_hat, columns, name)
    log_standardized_metrics(config, y, y_hat, columns, name, dm)


def log_normalized_metrics(config, y, y_hat, columns, name):
    y = np.array(y)
    y_hat = np.array(y_hat)
    columns = np.concat(columns)
    nmae_values = []
    nmse_values = []
    nrmse_values = []
    for i in range(config.dataset.num_time_series):
        indices = columns == i
        y_ts = y[indices]
        y_hat_ts = y_hat[indices]
        y_mean_abs = np.mean(np.abs(y_ts))
        residuals = y_hat_ts - y_ts
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        nmae_values.append(mae / y_mean_abs)
        nmse_values.append(mse / y_mean_abs)
        nrmse_values.append(rmse / y_mean_abs)
        #print(i, y_mean_abs, mae, mse, nmae_values[-1], nmse_values[-1])
    wandb.log({
        f"nMAE{name}": np.mean(nmae_values),
        f"nMSE{name}": np.mean(nmse_values),
        f"nRMSE{name}": np.mean(nrmse_values)}
    )


def log_standardized_metrics(config, y, y_hat, columns, name, dm):
    y = np.array(y)
    y_hat = np.array(y_hat)
    columns = np.concat(columns)
    smae_values = []
    smse_values = []
    for i in range(config.dataset.num_time_series):
        mean = dm.target_scaler.mean[i]
        sigma = dm.target_scaler.std[i]
        indices = columns == i
        y_ts = (y[indices] - mean) / sigma
        y_hat_ts = (y_hat[indices] - mean) / sigma
        residuals = y_hat_ts - y_ts
        smae = np.mean(np.abs(residuals))
        smse = np.mean(residuals ** 2)
        smae_values.append(smae)
        smse_values.append(smse)
    wandb.log({
        f"sMAE{name}": np.mean(smae_values),
        f"sMSE{name}": np.mean(smse_values)}
    )


def mqloss_batched(y, y_hat, quantiles_list, batch_size=1024):
    device = y.device

    quantiles = torch.tensor(
            quantiles_list, device=device, dtype=y.dtype
        )
    num_samples = y.shape[0]

    sample_metric_list = []

    time_metric_sum = None  
    total_samples = 0

    for start in range(0, num_samples, batch_size):

        end = min(start + batch_size, num_samples)
        y_batch = y[start:end]             # (batch, time)
        y_hat_batch = y_hat[start:end]       # (batch, time, num_quantiles)
        
        error = y_hat_batch - y_batch.unsqueeze(-1)  # (batch, time, num_quantiles)
        
        sq = torch.clamp(-error, min=0)    # (batch, time, num_quantiles)
        s1_q = torch.clamp(error, min=0)   # (batch, time, num_quantiles)
        
        losses = (quantiles * sq + (1 - quantiles) * s1_q) / quantiles.shape[-1]
        
        losses = torch.mean(losses, dim=2)  # (batch, time)
        
        sample_metric_batch = torch.mean(losses, dim=1)  # (batch,)
        sample_metric_list.append(sample_metric_batch)
        
        batch_size_actual = end - start
        batch_time_loss = torch.mean(losses, dim=0)  # (time,)
        if time_metric_sum is None:
            time_metric_sum = batch_time_loss * batch_size_actual
        else:
            time_metric_sum += batch_time_loss * batch_size_actual
        
        total_samples += batch_size_actual
        
    mq_sample = torch.cat(sample_metric_list, dim=0)  # shape: (N,)
    
    mq_time = time_metric_sum / total_samples  # shape: (time,)
    mq = torch.mean(mq_time)
    return mq_sample.cpu().detach().numpy(), mq_time.cpu().detach().numpy(), mq.cpu().detach().numpy()

def log_quantile_metrics(config, y, y_hat, name, timestamps):
    """
    Log quantile metrics using WandB.

    Arguments:
        config: Configuration object containing model and sample information.
        y, y_hat: True values and quantile predictions.
        name: Name of the model or experiment for logging.
    """
    device = "cpu"
    
    y = y.to(device)
    y_hat = y_hat.to(device)

    mq_sample, mq_time, mq = mqloss_batched(y, y_hat, config.model.quantiles, batch_size=10000)

    # calibration curve plot
    coverage = [y_hat[:, :, i].ge(y).float().mean().item() for i in range(len(config.model.quantiles))]

    table = wandb.Table(
        data=[[q, c] for q, c in zip(config.model.quantiles, coverage)],
        columns=["tcoverage", f"ocoverage{name}"],
    )
    wandb.log(
        {
            f"coverage_plot": wandb.plot.line(
                table, "tcoverage", f"ocoverage{name}", title=f"Coverage {name}"
            )
        }
    )

    # log sharpness and coverage
    sharpness = compute_sharpness(y_hat, config.model.quantiles)
    wandb.log(sharpness)
    interval_coverage = compute_coverage(y, y_hat, config.model.quantiles)
    wandb.log(interval_coverage)


    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(mq_sample)],
        columns=["sample", f"sample_mq{name}"],
    )
    wandb.log(
        {
            f"sample_mq_plot": wandb.plot.line(
                table, "sample", f"sample_mq{name}", title=f"Sample MQ {name}"
            )
        }
    )

    table = wandb.Table(
            data=[[i, r] for i, r in enumerate(mq_time)],
            columns=["time", "MQLoss{}".format(name)],
        )
    wandb.log(
            {
                "mq_plot": wandb.plot.line(
                    table, "time", "MQLoss{}".format(name), title="MQLoss{}".format(name)
                )
            }
        )
    wandb.log({f"MQLoss{name}": mq})

    # Log best and worst samples
    best_samples = np.argsort(mq_sample)[:5]
    average_samples = np.argsort(mq_sample)[len(mq_sample)//2-2:len(mq_sample)//2+3]
    worst_samples = np.argsort(mq_sample)[-5:]

    for i in range(5):
        log_quantile_sample(
            i,
            category="best",
            name=name,
            sample_idx=best_samples[i],
            y=y,
            y_hat=y_hat,
            config=config,
            timestamps=timestamps,
            mode=config.example_plot_mode,          # or "image"
            downsample=True,      # toggle to reduce series length
        )
        log_quantile_sample(
            i,
            category="average",
            name=name,
            sample_idx=average_samples[i],
            y=y,
            y_hat=y_hat,
            config=config,
            timestamps=timestamps,
            mode=config.example_plot_mode,
            downsample=True,
        )
        log_quantile_sample(
            i,
            category="worst",
            name=name,
            sample_idx=worst_samples[i],
            y=y,
            y_hat=y_hat,
            config=config,
            timestamps=timestamps,
            mode=config.example_plot_mode,
            downsample=True,
        )


    if timestamps is not None:
        # Convert datetime array to weekday and hour arrays
        # only use first timestamp of each sample
        first_timestamps = np.array([ts[0].tz_convert(None) for ts in timestamps])

        weekdays = (first_timestamps.astype("datetime64[D]").astype(int) + 3) % 7 # +3 shift as 1970-01-01 was a Thursday
        hours = first_timestamps.astype("datetime64[h]").astype(int) % 24

        # window_losses = losses.reshape(losses.shape[0], -1, config.sample.window_size)

        # Compute mean using NumPy for each weekday (0=Monday, 6=Sunday)
        unique_weekdays = np.sort(np.unique(weekdays))
        wd_dict = {0: "01 Monday", 1: "02 Tuesday", 2: "03 Wednesday", 3: "04 Thursday", 4: "05 Friday", 5: "06 Saturday", 6: "07 Sunday"}
        mean_by_weekday = {wd_dict[wd]: mq_sample[weekdays == wd].mean().item() for wd in unique_weekdays}

        # Compute mean using NumPy for each hour of the day
        unique_hours = np.sort(np.unique(hours))
        mean_by_hour = {hr: mq_sample[hours == hr].mean().item() for hr in unique_hours}

        # Compute mean using NumPy for weekday and hour
        mean_by_weekday_hour = {
            (wd_dict[wd], hr): mq_sample[(weekdays == wd) & (hours == hr)].mean().item()
            for wd in unique_weekdays
            for hr in unique_hours
        }


        # Log mean losses by weekday and hour
        table = wandb.Table(
            data=[[wd, mean] for wd, mean in mean_by_weekday.items()],
            columns=["weekday", "mean_loss"],
        )
        wandb.log(
            {
                "mean_loss_by_weekday": wandb.plot.bar(
                    table, "weekday", "mean_loss", title="Mean Loss by Weekday"
                )
            }
        )

        table = wandb.Table(
            data=[[hr, mean] for hr, mean in mean_by_hour.items()],
            columns=["hour", "mean_loss"],
        )
        wandb.log(
            {
                "mean_loss_by_hour": wandb.plot.bar(
                    table, "hour", "mean_loss", title="Mean Loss by Hour"
                )
            }
        )

        table = wandb.Table(
            data=[[i, mean] for i, mean in enumerate(mean_by_weekday_hour.values())],
            columns=["weekday hour", "mean_loss"],
        )
        wandb.log(
            {
                "mean_loss_by_weekday_hour": wandb.plot.line(
                    table, "weekday hour", "mean_loss", title="Mean Loss by Weekday and Hour"
                )
            }
        )

def build_timestamps(idx, timestamps, target_length):
    if timestamps is None:
        return None
    return pd.date_range(
        start=pd.to_datetime(timestamps[idx][0]),
        end=pd.to_datetime(timestamps[idx][1]),
        periods=target_length
    )


def log_quantile_sample(i, category, name, sample_idx, y, y_hat, config, timestamps,
                        mode="html", downsample=False):
    # Prepare data / timestamps
    ts = build_timestamps(sample_idx, timestamps, config.sample.target_length)
    y_series = y[sample_idx, :]
    y_hat_series = y_hat[sample_idx, :, :]

    fig = quantile_plot(
        y_series,
        y_hat_series,
        config.model.quantiles,
        timestamps=ts,
    )

    key_base = f"{category}_samples_{i}_q{name}"
    if mode == "html":
        # Use CDN to avoid embedding plotly.js runtime
        html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        wandb.log({key_base: wandb.Html(html)})
    elif mode == "image":
        # pio.kaleido.scope.chromium_args += ("--single-process",) 
        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        wandb.log({
            f"{key_base}_static": wandb.Image(
                img,
                caption=f"{category.capitalize()} sample {sample_idx} quantile plot"
            )
        })
    else:
        raise ValueError(f"Unknown mode: {mode}")

def log_metric_over_time(metric_name, metric_values, name):
    """
    Log a metric over time with WandB.

    Arguments:
        metric_name: Name of the metric (RMSE, MAE, etc.).
        metric_values: Calculated metric values.
        name: Name of the model or experiment for logging.
    """
    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(metric_values)],
        columns=["time", f"{metric_name}{name}"],
    )
    wandb.log(
        {
            f"{metric_name}_plot": wandb.plot.line(
                table, "time", f"{metric_name}{name}", title=f"{metric_name} {name}"
            )
        }
    )
    wandb.log({f"{metric_name}{name}": np.mean(metric_values)})


def log_nadir_error(y, y_hat, name):
    """
    Log the nadir error metric using WandB.

    Arguments:
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
    """
    nadir_y = y.values[np.arange(len(y)), np.argmax(np.abs(y.values), axis=1)]
    nadir_y_hat = y_hat.values[
        np.arange(len(y_hat)), np.argmax(np.abs(y_hat.values), axis=1)
    ]
    nadir_error = np.abs(nadir_y - nadir_y_hat)

    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(nadir_error)],
        columns=["sample", f"nadir_error{name}"],
    )
    wandb.log(
        {
            f"nadir_error_plot": wandb.plot.histogram(
                table, f"nadir_error{name}", title=f"Nadir {name}"
            )
        }
    )
    wandb.log({f"nadir_error{name}": np.mean(nadir_error)})


def log_rocof_error(y, y_hat, name):
    """
    Log the rate of change of frequency (RoCoF) error using WandB.

    Arguments:
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
    """
    rocof_y = y.diff(axis=1).T.rolling(30, center=True).mean().T.abs().max(axis=1)
    rocof_y_hat = (
        y_hat.diff(axis=1).T.rolling(30, center=True).mean().T.abs().max(axis=1)
    )
    rocof_error = np.abs(rocof_y - rocof_y_hat)

    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(rocof_error)],
        columns=["sample", f"rocof_error{name}"],
    )
    wandb.log(
        {
            f"rocof_error_plot": wandb.plot.histogram(
                table, f"rocof_error{name}", title=f"RoCoF {name}"
            )
        }
    )
    wandb.log({f"rocof_error{name}": np.mean(rocof_error)})


def log_sample_mae(config, y, y_hat, name):
    """
    Log sample-wise MAE.

    Arguments:
        config: Configuration object.
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
    """
    sample_mae = np.mean(np.abs((y_hat - y)), axis=1)

    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(sample_mae)],
        columns=["sample", f"sample_mae{name}"],
    )
    wandb.log(
        {
            f"sample_mae_plot{name}": wandb.plot.line(
                table, "sample", f"sample_mae{name}", title=f"Sample MAE {name}"
            )
        }
    )


def log_sample_mse(config, y, y_hat, name):
    """
    Log sample-wise MSE and identify best and worst samples.

    Arguments:
        config: Configuration object.
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
    """
    sample_mse = np.mean((y_hat - y) ** 2, axis=1)

    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(sample_mse)],
        columns=["sample", f"sample_mse{name}"],
    )
    wandb.log(
        {
            f"sample_mse_plot{name}": wandb.plot.line(
                table, "sample", f"sample_mse{name}", title=f"Sample MSE {name}"
            )
        }
    )

    # Log best and worst samples
    best_samples = np.argsort(sample_mse.values)[:5]
    average_samples = np.argsort(sample_mse.values)[len(sample_mse)//2-2:len(sample_mse)//2+3]
    worst_samples = np.argsort(sample_mse.values)[-5:]

    for i in range(5):
        wandb.log(
            {
                f"best_samples_{i}{name}": sample_plot(
                    xs=np.arange(len(y.iloc[best_samples[i]].values)),
                    ys=[y.iloc[best_samples[i]], y_hat.iloc[best_samples[i]]],
                    keys=["y", "y_hat"],
                    title=f"Best samples {name}",
                )
            }
        )
        wandb.log(
            {
                f"average_samples_{i}{name}": sample_plot(
                    xs=np.arange(len(y.iloc[average_samples[i]].values)),
                    ys=[y.iloc[average_samples[i]], y_hat.iloc[average_samples[i]]],
                    keys=["y", "y_hat"],
                    title=f"Average samples {name}",
                )
            }
        )
        wandb.log(
            {
                f"worst_samples_{i}{name}": sample_plot(
                    xs=np.arange(len(y.iloc[worst_samples[i]].values)),
                    ys=[y.iloc[worst_samples[i]], y_hat.iloc[worst_samples[i]]],
                    keys=["y", "y_hat"],
                    title=f"Worst samples {name}",
                )
            }
        )

def compute_sharpness(y_hat, quantiles):
    """
    Compute sharpness metrics (mean, median, std) for different quantile intervals.

    Parameters:
    - y_hat: shape (N, L, Q), predicted quantiles for each sample.
    - quantiles: List of quantiles used in the model (length Q, must be sorted).

    Returns:
    - sharpness_dict: Dictionary containing mean, median, and std of sharpness for each interval.
    """
    quantiles = np.array(quantiles)  # Ensure quantiles are an array
    sharpness_dict = {}

    # Iterate over symmetric quantile pairs
    for i in range(len(quantiles) // 2):  # Only iterate over lower-half indices
        low_idx = i
        high_idx = -(i + 1)
        
        sharpness_values = y_hat[:, :, high_idx] - y_hat[:, :, low_idx]  # Vectorized interval width

        # Store mean, median, and std of sharpness
        key_prefix = f"sharpness_{quantiles[low_idx]*100:.0f}-{quantiles[high_idx]*100:.0f}"
        sharpness_dict[f"{key_prefix}_mean"] = sharpness_values.mean()
        sharpness_dict[f"{key_prefix}_median"] = np.median(sharpness_values)
        sharpness_dict[f"{key_prefix}_std"] = sharpness_values.std()

    return sharpness_dict

def compute_coverage(y_true, y_hat, quantiles):
    """
    Compute empirical coverage for different quantile intervals.

    Parameters:
    - y_true: ndarray of shape (N, L), true target values.
    - y_hat: ndarray of shape (N, L, Q), predicted quantiles.
    - quantiles: List of quantiles used in the model (length Q, must be sorted).

    Returns:
    - coverage_dict: Dictionary containing empirical coverage for each interval.
    """
    quantiles = np.array(quantiles)  # Ensure quantiles are an array
    coverage_dict = {}

    # Iterate over symmetric quantile pairs
    for i in range(len(quantiles) // 2):  # Only iterate over lower-half indices
        low_idx = i
        high_idx = -(i + 1)

        lower_bound = y_hat[:, :, low_idx]
        upper_bound = y_hat[:, :, high_idx]

        # Compute how often the true y falls within the predicted interval
        coverage_values = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = coverage_values.float().mean() # Percentage of samples covered

        key_prefix = f"coverage_{quantiles[low_idx]*100:.0f}-{quantiles[high_idx]*100:.0f}"
        coverage_dict[key_prefix] = coverage

    return coverage_dict