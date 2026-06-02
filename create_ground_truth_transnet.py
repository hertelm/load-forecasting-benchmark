import pandas as pd
from datetime import datetime
import numpy as np


if __name__ == "__main__":
    path = "_data/transnet_bw/TransnetBW_Total_Load_1h.csv"

    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31, 23, 59)

    data = pd.read_csv(path, parse_dates=[0], index_col=0)["Actual_Total_Load"]
    data = data[start:end]

    timestamps = data.index[:-95]
    ids = ["TransnetBW"] * len(timestamps)

    gt_index = pd.MultiIndex.from_arrays([ids, timestamps], names=["ID", "Timestamp"])

    ground_truth = {}

    for h in range(96):
        ground_truth[h] = data.iloc[h:h + len(gt_index)].values

    ground_truth_df = pd.DataFrame(ground_truth, index=gt_index)
    print(ground_truth_df)

    ground_truth_df.to_csv("results/transnet_bw/ground_truth.csv")
