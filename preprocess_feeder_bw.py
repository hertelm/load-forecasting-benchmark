import pandas as pd


if __name__ == "__main__":
    load_file = "_data/feeder_bw/feeder_load.csv"
    features_file = "_data/feeder_bw/feeder_features.csv"

    load_df = pd.read_csv(load_file, parse_dates=[0], index_col=0)
    load_df.to_pickle(load_file[:-4] + ".pkl")

    features_df = pd.read_csv(features_file, parse_dates=[1], index_col=[0, 1])
    features_df.to_pickle(features_file[:-4] + ".pkl")
