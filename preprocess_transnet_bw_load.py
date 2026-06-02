from pathlib import Path
import pandas as pd
from pandas.api.types import is_float_dtype


if __name__ == "__main__":
    DIR = Path("_data/transnet_bw")
    FILE_PREFIX = "GUI_TOTAL_LOAD_DAYAHEAD"

    START = 2015

    QUARTER_HOURLY = False

    OUTPUT_DIR = Path("_data/transnet_bw")
    OUTPUT_FILE_NAME = Path(f"TransnetBW_Total_Load{'_1h' if not QUARTER_HOURLY else '_15min'}.csv")
    # change the names of the columns (shorten them a bit and remove the blanks)
    COLUMNS = {
        "Day-ahead Total Load Forecast (MW)": "DA_Total_Load_Forecast",
        "Actual Total Load (MW)": "Actual_Total_Load"
    }

    csvs = []
    for item in DIR.iterdir():
        if item.name.endswith(".csv") \
                and item.name.startswith(FILE_PREFIX):
            year = int(item.name[32:36])
            if year >= START:
                csvs.append((year, item))

    csvs = sorted(csvs)

    output_df = pd.DataFrame()
    for _, item in csvs:
        print(item)
        df = pd.read_csv(item)
        df["Timestamp"] = pd.to_datetime(
            list(map(
                lambda x: x[:16], df["MTU (CET/CEST)"]
            )),
            format="%d/%m/%Y %H:%M"
        )
        df = df.set_index("Timestamp", drop=True)
        # time switch october
        df = df[~ df.index.duplicated(keep="first")]

        for column, target_column in COLUMNS.items():
            if not is_float_dtype(df[column]):
                df[target_column] = list(map(lambda x: float(x) if x != "-" else 0.0, df[column]))
            else:
                df[target_column] = df[column]
        df = df[[col for _, col in COLUMNS.items()]]

        if QUARTER_HOURLY:
            out_df = pd.DataFrame(index=pd.date_range(
                df.index[0], df.index[-1], freq="15min"
            ))
            out_df = out_df.join(df, how="left")
            # print(out_df[out_df[COLUMNS].isna()].index)
            out_df = out_df.interpolate("linear")
            # print(out_df[out_df[COLUMNS].isna()].index)
        else:
            out_df = pd.DataFrame(index=pd.date_range(
                df.index[0].replace(minute=0), df.index[-1].replace(minute=0),
                freq="1h"
            ))
            out_df = out_df.join(
                df.groupby(pd.Grouper(freq="h")).mean()
            )
            out_df = out_df.interpolate("linear")

        if output_df.empty:
            output_df = out_df
        else:
            output_df = pd.concat([output_df, out_df])

    output_path = OUTPUT_DIR / OUTPUT_FILE_NAME
    output_df.to_csv(
        output_path,
        index=True,
        index_label="Timestamp"
    )
    print(f"Timeseries written to:", output_path)