import datetime
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from omegaconf import DictConfig
import time

from dataset.time_feature_utils import dyn_time_features, static_time_features
from dataset.scaler import StandardScaler, MinMaxScaler, IdentityScaler

from rich.progress import Progress

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df,
        history_length: int,     
        forecast_length: int, 
        stride: int,
        horizon_window_size: int,
        resolution_factor=None,
        feature_lookback=None,
        feature_lookahead=None,
        static_time_dict=None,
        future_features=None,
        past_features=None,
        static_features=None,
        static_features_dict=None,
        drop_nan_windows=True,
    ):
        self.data_np       = df.to_numpy() 
        # self.timestamps    = df.index.to_numpy()      # shape (T,)
        self.timestamps    = df.index      # shape (T,)
        self.x_size = history_length
        self.y_size   = forecast_length * horizon_window_size
        self.horizon_window_size = horizon_window_size
        self.window_size   = self.x_size + self.y_size
        self.stride         = stride

        # Store feature arrays/configs
        if future_features is not None and not isinstance(future_features, list):
            future_features = [future_features]
        if past_features is not None and not isinstance(past_features, list):
            past_features = [past_features]
        self.future_np     = np.array(future_features) if future_features is not None else None
        self.past_np       = np.array(past_features) if past_features  is not None else None
        self.static_df     = static_features
        self.static_dict   = static_features_dict or {}
        self.static_time_dict = static_time_dict or {}

        # For past/future indexing
        self.resolution_factor = resolution_factor or 1
        self.lookback          = feature_lookback or 0
        self.lookahead         = feature_lookahead or 0

        # For multiple time series
        self.n_time_series = len(df.columns)

        # Figure out all candidate window starts
        L = len(self.data_np)
        W = self.window_size
        if L < W:
            raise ValueError(f"Data length {L} < window size {W}")
        total_windows = (L - W) // self.stride + 1

        all_starts = []
        for column in range(self.n_time_series):
            starts = np.arange(0, total_windows * self.stride, self.stride)
            col_lst = np.ones_like(starts) * column
            all_starts.extend(list(zip(col_lst, starts)))

        # Drop windows containing NaNs in input or target
        logger.info(f"Total candidate windows: {len(all_starts)}")
        if drop_nan_windows:
            is_valid = [True] * len(all_starts)
            with Progress() as progress:
                task = progress.add_task("[cyan]Checking NaNs in windows...", total=len(all_starts))
                for i, (c, s) in enumerate(all_starts):
                    # main series window
                    x = self.data_np[s : s + self.x_size, c : c + 1]
                    y = self.data_np[s + self.x_size : s + W, c : c + 1]

                    # past/future features window
                    checks = [x, y]
                    if self.past_np is not None:
                        feat_col = 0 if len(self.past_np) == 1 else c
                        pf = self._extract_past_future(self.past_np[feat_col], s)
                        checks.append(pf)
                    if self.future_np is not None:
                        feat_col = 0 if len(self.future_np) == 1 else c
                        ff = self._extract_past_future(self.future_np[feat_col], s)
                        checks.append(ff)

                    # static features (if any) at this window’s end
                    if self.static_df is not None or self.static_time_dict:
                        ts = self.timestamps[s + self.x_size]
                        static_parts = []
                        if self.static_time_dict:
                            st, _ = static_time_features([ts], self.static_time_dict)
                            static_parts.append(st.squeeze(0).numpy())
                        if self.static_df is not None:
                            sf = self._extract_static_df(ts).numpy()
                            static_parts.append(sf)
                        if static_parts:
                            checks.append(np.concatenate(static_parts, axis=-1))

                    # drop if **any** array has NaNs
                    if any(np.isnan(arr).any() for arr in checks):
                        is_valid[i] = False

                    progress.advance(task)
            valid = [all_starts[i] for i in range(len(all_starts)) if is_valid[i]]
            self.starts = np.array(valid, dtype=int)
        else:
            self.starts = np.array(all_starts, dtype=int)

        logger.info(f"Valid windows after NaN check: {len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the given index,
        extracting a history window, a forecast window, plus any exogenous
        and static features.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple (inp, out)
            inp : dict of Tensors
                Contains:
                  • past_target     — Tensor of shape (x_size, num_targets)
                  • past_features   — Tensor of shape (feature_lookback, num_past_features),
                                      or empty Tensor if no past features
                  • future_features — Tensor of shape (feature_lookback + feature_lookahead,
                                      num_future_features), or empty Tensor if no future features
                  • static          — Tensor of shape (num_static_features),
                                      or empty Tensor if no static features

            out : Tensor
                Tensor of shape (y_size, num_targets) containing the
                future target values to predict.

        Examples
        --------
        # Per‐sample
        inp = {
            "past_target":     Tensor[x_size, num_targets],
            "past_features":   Tensor[feature_lookback, num_past_features],
            "future_features": Tensor[feature_lookback+feature_lookahead, num_future_features],
            "static":          Tensor[num_static_features]
        }
        out = Tensor[y_size, num_targets]

        # Batched (DataLoader)
        inp = {
            "past_target":     Tensor[batch_size, x_size, num_targets],
            "past_features":   Tensor[batch_size, feature_lookback, num_past_features],
            "future_features": Tensor[batch_size, feature_lookback+feature_lookahead, num_future_features],
            "static":          Tensor[batch_size, num_static_features]
        }
        out = Tensor[batch_size, y_size, num_targets]
        """
        # Compute slice bounds
        c, s = self.starts[idx]
        e = s + self.window_size

        # Core input/target
        x_np = self.data_np[s : s + self.x_size, c : (c + 1)]
        y_np = self.data_np[s + self.x_size : e, c : (c + 1)]
        inp  = {"past_target": torch.from_numpy(x_np).float()}

        # past/future features
        feat_c = 0 if self.future_np is None or len(self.future_np) == 1 else c
        pf = self._extract_past_future(self.past_np[feat_c],  s, lookahead=0)  if self.past_np  is not None else None
        ff = self._extract_past_future(self.future_np[feat_c], s)  if self.future_np is not None else None
        inp["past_features"]   = torch.from_numpy(pf).float() if pf is not None else torch.empty(0)
        inp["future_features"] = torch.from_numpy(ff).float() if ff is not None else torch.empty(0)

        # static time + static DataFrame features
        parts = []
        # static‐time features at the **last** timestamp of the window
        ts_last = self.timestamps[s + self.x_size]
        if self.static_time_dict:
            # returns (tensor, names)
            st, _ = static_time_features([ts_last], self.static_time_dict)
            parts.append(st.squeeze(0))
        # static_features_df: ffill/bfill columns
        if self.static_df is not None:
            sf = self._extract_static_df(ts_last)
            parts.append(sf)
        inp["static"] = torch.cat(parts, dim=-1) if parts else torch.empty(0)

        # final target tensor
        out = torch.from_numpy(y_np).float()
        return inp, out, c

    def _extract_past_future(self, arr: np.ndarray, start: int, lookback: int = None, lookahead: int = None) -> np.ndarray:

        lb = lookback  if lookback  is not None else self.lookback
        la = lookahead if lookahead is not None else self.lookahead
        idx_center = (start + self.x_size) // self.resolution_factor
        past_idxs   = idx_center - np.arange(lb, 0, -1)
        future_idxs = idx_center + np.arange(la)
        all_idxs    = np.concatenate([past_idxs, future_idxs], axis=0)
        # gather with NaN‐padding
        out = np.full((len(all_idxs), arr.shape[1]), np.nan, dtype=float)
        valid = (all_idxs >= 0) & (all_idxs < len(arr))
        out[valid] = arr[all_idxs[valid]]
        return out

    def _extract_static_df(
        self,
        ts: pd.Timestamp
    ) -> torch.Tensor:
        """
        For each column in self.static_df, look up in the coarser
        series the span that covers ts (i.e. the last index ≤ ts),
        then either take that value (“cover” mode) or the one immediately
        before it (“prev” mode).
        static_features_dict maps column → mode, where mode ∈ {'prev','cover'}.
        """
        vals = []
        for col, mode in self.static_dict.items():
            series = self.static_df[col]
            pos_cover = series.index.get_indexer([ts], method="ffill")[0]

            if mode == "cover":
                val = series.iat[pos_cover] if pos_cover >= 0 else np.nan

            elif mode == "prev":
                val = series.iat[pos_cover - 1] if pos_cover > 0 else np.nan

            vals.append(val)

        arr = np.array(vals, dtype=float)     
        return torch.from_numpy(arr).float()
    
    @property
    def sample_times(self):
        return list(zip(self.timestamps[self.starts + self.x_size], 
                   self.timestamps[self.starts + self.x_size + self.y_size - self.horizon_window_size]))


SCALER = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "none":     IdentityScaler,
}


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.batch_size = config.model.batch_size
        self.num_workers = getattr(config, 'num_workers', 1)
        self.pin_memory = getattr(config, 'pin_memory', False)

        self.target_scaler = SCALER[config.scalers.target]()
        if self.config.features.dyn_per_ts:
            self.feature_scaler = [SCALER[config.scalers.feature]() for i in range(self.config.dataset.num_time_series)]
        else:
            self.feature_scaler = SCALER[config.scalers.feature]()
        self.stat_feature_scaler = SCALER[config.scalers.static]()

        self.data = None
        self.dyn_features = None
        self.stat_features = None
        self.resolution_factor = 1

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
 
    def prepare_data(self):
        data_path = os.path.join(
            self.config.dataset.data_directory,
            self.config.dataset.data_name,
        )
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_pickle(data_path)

        freq = pd.infer_freq(self.data.index)
        if freq is None:
            raise ValueError("The frequency of the target data index could not be inferred. Please ensure that the index of the target data has a consistent frequency.")
        if self.data.index.tz is None:
            self.data.index = self.data.index.tz_localize(self.config.dataset.data_tz)
        elif str(self.data.index.tz) != self.config.dataset.data_tz:
            logging.warning(
                f"Target tz {self.data.index.tz} ≠ config {self.config.dataset.data_tz}"
            )

        # start, end = self.config.dataset.start, self.config.dataset.end
        start, end = self._get_feature_window()
        self.data = self.data.loc[pd.to_datetime(start) : pd.to_datetime(end)]

        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()

        logger.info(f"Data loaded with shape {self.data.shape} and index from {self.data.index[0]} to {self.data.index[-1]}")
        feat_start, feat_end = self._get_feature_window()

        logger.info(f"Feature window from {feat_start} to {feat_end}")
        self.dyn_features = self._load_and_slice(
            self.config.features.dyn,
            self.config.features.dyn_directory,
            self.config.features.dyn_file_name,
            self.config.features.dyn_tz,
            feat_start, feat_end,
            label="dynamic"
        )

        if self.dyn_features is not None and self.config.features.dyn_per_ts:
            ts_ids = self.dyn_features.index.get_level_values(0).unique()
            print(self.dyn_features)
            self.dyn_features = [self.dyn_features.loc[ts_id] for ts_id in ts_ids]

        if isinstance(self.dyn_features, list):
            logger.info(f"Dynamic features loaded with shape {np.asarray(self.dyn_features).shape}")
        else:
            logger.info(f"Dynamic features loaded with shape {self.dyn_features.shape if self.dyn_features is not None else 'None'}")
        self.stat_features = self._load_and_slice(
            self.config.features.stat,
            self.config.features.stat_directory,
            self.config.features.stat_file_name,
            self.config.features.stat_tz,
            feat_start, feat_end,
            label="static"
        )
        logger.info(f"Static features loaded with shape {self.stat_features.shape if self.stat_features is not None else 'None'}")
        """if self.dyn_features is not None:
            if len(self.data) % len(self.dyn_features) != 0:
                raise ValueError("len(data) must be multiple of len(dyn_features).")
            self.resolution_factor = len(self.data) // len(self.dyn_features)"""
        self.resolution_factor = 1

    def setup(self, stage=None):
        if self.data is None:
            self.prepare_data()

        tz = self.config.dataset.split_tz
        frequency = self.data.index[1] - self.data.index[0]
        train_end = pd.to_datetime(self.config.dataset.val_split).tz_localize(tz)
        val_start  = train_end - frequency * self.config.sample.input_length
        val_end = pd.to_datetime(self.config.dataset.test_split).tz_localize(tz)
        test_start = val_end - frequency * self.config.sample.input_length
        logger.info(f"Train split to {train_end}, validation split from {val_start} to {val_end}, test split from {test_start}")

        if stage in (None, 'fit'):
            train_df = self.data.loc[:train_end].iloc[:-1]
            val_df   = self.data.loc[val_start:val_end].iloc[:-1]

            # dynamic: split & scale
            if self.dyn_features is not None:
                if self.config.features.dyn_per_ts:
                    train_past, train_future, val_past, val_future = [], [], [], []
                    for ts_i in range(len(self.dyn_features)):
                        train_past_ts, train_future_ts = self._split_and_scale_dyn(
                            self.dyn_features[ts_i].loc[:train_end].iloc[:-1], fit=True, column=ts_i)
                        val_past_ts, val_future_ts = self._split_and_scale_dyn(
                            self.dyn_features[ts_i].loc[val_start:val_end].iloc[:-1], column=ts_i)
                        train_past.append(train_past_ts)
                        train_future.append(train_future_ts)
                        val_past.append(val_past_ts)
                        val_future.append(val_future_ts)
                    if train_past[0] is None:
                        train_past = None
                    if train_future[0] is None:
                        train_future = None
                    if val_past[0] is None:
                        val_past = None
                    if val_future[0] is None:
                        val_future = None
                else:
                    train_past, train_future = self._split_and_scale_dyn(self.dyn_features.loc[:train_end].iloc[:-1], fit=True)
                    val_past,   val_future   = self._split_and_scale_dyn(self.dyn_features.loc[val_start:val_end].iloc[:-1])
            else:
                train_past = train_future = val_past = val_future = None

            # static: scale once, keep full DataFrame
            if self.stat_features is not None:
                train_stat = self._scale_stat(self.stat_features.loc[:train_end].iloc[:-1], fit=True)
                val_stat   = self._scale_stat(self.stat_features.loc[val_start:val_end].iloc[:-1])
            else:
                train_stat = val_stat = None

            # target scaling
            self.target_scaler.fit(train_df)
            train_df = pd.DataFrame(
                self.target_scaler.transform(train_df),
                index=train_df.index,
                columns=train_df.columns,
            )
            val_df   = pd.DataFrame(
                self.target_scaler.transform(val_df),
                index=val_df.index,
                columns=val_df.columns,
            )

            # Create datasets
            self.train_dataset = self._create_dataset(
                train_df, train_past, train_future, train_stat
            )
            self.val_dataset = self._create_dataset(
                val_df, val_past, val_future, val_stat
            )
            self.training_data_preparation_end_time = time.time()


        if stage in (None, 'test', 'predict'):
            test_df = self.data.loc[test_start:]
            
            if self.dyn_features is not None:
                if self.config.features.dyn_per_ts:
                    test_past, test_future = [], []
                    for ts_i in range(len(self.dyn_features)):
                        test_past_ts, test_future_ts = self._split_and_scale_dyn(
                            self.dyn_features[ts_i].loc[test_start:], column=ts_i
                        )
                        test_past.append(test_past_ts)
                        test_future.append(test_future_ts)
                    if test_past[0] is None:
                        test_past = None
                    if test_future[0] is None:
                        test_future = None
                else:
                    test_past, test_future = self._split_and_scale_dyn(self.dyn_features.loc[test_start:])
            else:
                test_past = test_future = None

            if self.stat_features is not None:
                test_stat = self._scale_stat(self.stat_features.loc[test_start:])
            else:
                test_stat = None

            test_df = pd.DataFrame(
                self.target_scaler.transform(test_df),
                index=test_df.index,
                columns=test_df.columns,
            )
            self.test_dataset = self._create_dataset(
                test_df, test_past, test_future, test_stat,
                return_datetime=True
            )

    def _get_feature_window(self):
        tz = self.data.index.tz
        start = (pd.to_datetime(self.config.dataset.start).tz_localize(tz)
                 if self.config.dataset.start else self.data.index[0])
        end = (pd.to_datetime(self.config.dataset.end).tz_localize(tz)
               if self.config.dataset.end else self.data.index[-1])
        return start, end

    def _load_and_slice(
        self,
        feature_cfg,
        directory: str,
        filename: str,
        tz: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> Optional[pd.DataFrame]:
        if feature_cfg is None:
            return None
        path = os.path.join(directory, filename)
        df = pd.read_pickle(path)

        if isinstance(df.index, pd.MultiIndex):
            if start is not None:
                df = df[
                    df.index.get_level_values(1) >= np.datetime64(start)
                ]
            if end is not None:
                df = df[
                    df.index.get_level_values(1) <= np.datetime64(end)
                ]
        else:
            freq = pd.infer_freq(df.index)
            if freq is None:
                raise ValueError(f"{label.capitalize()} index has no consistent frequency.")
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz)
            df = df.loc[start:end]
        if not isinstance(feature_cfg, (dict, DictConfig)):
            raise ValueError(f"Expected dict or DictConfig for config.features.{label}, got {type(feature_cfg)}.")
        
        df = df[list(feature_cfg.keys())]

        return df

    def _split_and_scale_dyn(self, df_feats: pd.DataFrame, fit: bool=False, column: int=None):
        if self.dyn_features is None:
            logger.info("No dynamic features configured, returning None for past and future.")
            return None, None

        if column is None:
            scaler = self.feature_scaler
        else:
            scaler = self.feature_scaler[column]

        if fit:
            scaler.fit(df_feats)
        df_feats = pd.DataFrame(
            scaler.transform(df_feats),
            index=df_feats.index,
            columns=df_feats.columns,
        )

        past_cols   = [k for k, v in self.config.features.dyn.items() if v == 'past']
        future_cols = [k for k, v in self.config.features.dyn.items() if v == 'future']
        past   = df_feats[past_cols]   if past_cols   else None
        future = df_feats[future_cols] if future_cols else None

        if self.config.features.dyn_time_dict:
            time_feats, _ = dyn_time_features(df_feats.index, self.config.features.dyn_time_dict)
            future = pd.concat([future, time_feats], axis=1) if future is not None else time_feats

        return past, future

    def _scale_stat(self, df_stat: pd.DataFrame, fit: bool=False) -> Optional[pd.DataFrame]:
        if df_stat is None:
            return None
        if fit:
            self.stat_feature_scaler.fit(df_stat)
        return pd.DataFrame(
            self.stat_feature_scaler.transform(df_stat),
            index=df_stat.index,
            columns=df_stat.columns,
        )

    def _create_dataset(
        self,
        data: pd.DataFrame,
        past: pd.DataFrame,
        future: pd.DataFrame,
        static: pd.DataFrame,
        return_datetime: bool=False,
    ) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            df=data,
            history_length=self.config.sample.input_length,
            forecast_length=self.config.sample.target_length,
            stride=self.config.sample.step_size,
            horizon_window_size=self.config.sample.window_size,
            resolution_factor=self.resolution_factor,
            feature_lookback=self.config.sample.input_length,
            feature_lookahead=self.config.sample.target_length,
            static_time_dict=self.config.features.static_time_dict,
            past_features=past,
            future_features=future,
            static_features=static,
            static_features_dict=self.config.features.stat,
            drop_nan_windows=True,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self.test_dataloader()


    def _dataloader(self, dataset, shuffle: bool=False, batch_size=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )