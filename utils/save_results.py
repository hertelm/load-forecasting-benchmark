from utils.saving_utils.data_handling import extract_predictions, save_predictions
from utils.logging_utils.reshaping import reshape_for_quantiles, reshape_for_classic_loss
from utils.logging_utils.metrics import log_deterministic_metrics, log_quantile_metrics


def save_results(config, prediction, dm, name="", timestamps=None):
    """
    Save model predictions and metrics.

    Arguments:
        config: Configuration object containing model, sample, and output settings.
        prediction: List of tuples where each tuple contains (y_hat, y) and possibly more.
        dm: data module
        name: used as a suffix in logging to wandb (e.g. "-end" if training end is logged in addition to best model at checkpoint)
    """
    y_hat, y, columns = extract_predictions(prediction)

    # Reshape tensors according to the loss function
    if config.model.loss_function == "mqloss":
        y_hat, y, y_hat_median, y_df = reshape_for_quantiles(config, y_hat, y)
    else:
        y_hat, y = reshape_for_classic_loss(config, y_hat, y)
    
    # Save predictions
    if config.save_predictions:
        save_predictions(config, y_hat, y, name)
    
    # Calculate and log metrics
    if config.model.loss_function == "mqloss":
        log_quantile_metrics(config, y, y_hat, name, timestamps)
        log_deterministic_metrics(config, y_df, y_hat_median, columns, name, dm)
    else:
        log_deterministic_metrics(config, y, y_hat, columns, name, dm)
