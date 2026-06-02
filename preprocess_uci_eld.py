import pandas as pd


if __name__ == "__main__":
    load_file = "_data/uci_eld/uci_eld.csv"
    features_file = "_data/uci_eld/uci_eld_features.csv"

    load_df = pd.read_csv(load_file, parse_dates=[0], index_col=0)
    print(load_df)
    load_df.to_pickle(load_file[:-4] + ".pkl")

    features_df = pd.read_csv(features_file, parse_dates=[0], index_col=0)
    print(features_df)
    features_df.to_pickle(features_file[:-4] + ".pkl")
