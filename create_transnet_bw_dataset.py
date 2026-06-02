import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import holidays


if __name__ == "__main__":
    #END_DATE = "2024-12-31 23:00:00"
    END_DATE = "2025-12-31 23:00:00"
    WEATHER_FEATURES = [
        ("Air_Temperature_2m_1h.csv", "Air_Temperature_2m"),
        ("Global_Horizontal_Irradiance_1h.csv", "Global_Horizontal_Irradiance"),
        ("Total_Precipitation_1h.csv", "Total_Precipitation"),
        ("Wind_Speed_10m_1h.csv", "Wind_Speed_10m"),
    ]

    path = "_data/transnet_bw/TransnetBW_Total_Load_1h.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    time_series = df["Actual_Total_Load"]
    print(time_series)

    time_series = time_series[:END_DATE]
    print(time_series)

    out_path = "_data/transnet_bw/TransnetBW_Actual_Total_Load.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(time_series, f)
    print(f"Saved to {out_path}")

    feature_df = pd.DataFrame(index=time_series.index)
    for file_name, col_name in WEATHER_FEATURES:
        weather_path = "_data/transnet_bw/" + file_name
        weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)
        #weather_df = weather_df[:END_DATE]
        #print(weather_df)
        feature_df[col_name] = np.mean(weather_df.values, axis=1)[:len(time_series)]
    bw_holidays = holidays.Germany(state="BW")
    feature_df["Holiday"] = [1 if x in bw_holidays else 0 for x in feature_df.index]
    print(feature_df)

    #plt.plot(feature_df["Global_Horizontal_Irradiance"] * 10)
    #plt.show()

    features_out_path = "_data/transnet_bw/TransnetBW_Features.pkl"
    with open(features_out_path, "wb") as f:
        pickle.dump(feature_df, f)
    print(f"Saved to {features_out_path}")

    index_out_path = "_data/transnet_bw/TransnetBW_Actual_Total_Load_index.pkl"
    empty_df = pd.DataFrame(index=time_series.index)  # create an empty DataFrame with the same index
    with open(index_out_path, "wb") as f:
        pickle.dump(empty_df, f)
    print(f"Saved to {index_out_path}")
