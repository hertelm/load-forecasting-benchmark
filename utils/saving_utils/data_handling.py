import os
import pickle
from datetime import datetime

def extract_predictions(prediction):
    """
    Extract the predictions (y_hat, y) from the prediction list.

    Arguments:
        prediction: List of tuples containing predictions.

    Returns:
        y_hat, y: Extracted predictions and true values.
    """
    y_hat = [pred[0] for pred in prediction]
    y = [pred[1] for pred in prediction]
    columns = [pred[3] for pred in prediction]
    return y_hat, y, columns


def save_predictions(config, y_hat, y, name=""):
    """
    Save y_hat and y predictions to files.

    Arguments:
        config: Configuration object containing the output directory.
        y_hat, y: Predictions and true values to save.
    """
    output_dir = os.path.join(
        config.output_directory,
        datetime.today().strftime("%Y-%m-%d"),
        config.model.wandb_name,
    ).replace(":", "")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"y_hat{name}.pkl"), "wb") as f:
        pickle.dump(y_hat, f)
    with open(os.path.join(output_dir, f"y{name}.pkl"), "wb") as f:
        pickle.dump(y, f)
