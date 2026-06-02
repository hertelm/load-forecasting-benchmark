import numpy as np
import torch

class IdentityScaler:
    def fit(self, data):
        return self
    def transform(self, data, columns=None):
        return data
    def inverse_transform(self, data, columns=None):
        return data

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        if torch.is_tensor(data):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)

        self.mean = np.array(data.mean(axis=0))
        self.std = np.array(data.std(axis=0))

        self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero

    def transform(self, data, columns=None):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler: you must call .fit(...) before .transform(...)")
        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std  = torch.as_tensor(self.std,  dtype=data.dtype, device=data.device)

            if columns is not None:
                cols = columns if isinstance(columns, torch.Tensor) else torch.tensor(columns, dtype=torch.long, device=data.device)
                mean = mean[cols]
                std  = std[cols]
                
                view_shape = [len(cols)] + [1] * (data.ndim - 1)
                mean = mean.view(*view_shape)
                std  = std.view(*view_shape)

            return (data - mean) / std
        
        else:
            arr = np.asarray(data)
            if columns is not None:
                cols = np.asarray(columns, dtype=int)
                out = arr.copy()
                out[:, cols] = (out[:, cols] - self.mean[cols]) / self.std[cols]
                return out
            return (arr - self.mean) / self.std

    def inverse_transform(self, data, columns=None):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler: you must call .fit(...) before .inverse_transform(...)")

        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std  = torch.as_tensor(self.std,  dtype=data.dtype, device=data.device)

            if columns is not None:
                cols = columns if isinstance(columns, torch.Tensor) else torch.tensor(columns, dtype=torch.long, device=data.device)
                mean = mean[cols]
                std  = std[cols]
                view_shape = [len(cols)] + [1] * (data.ndim - 1)
                mean = mean.view(*view_shape)
                std  = std.view(*view_shape)

            return data * std + mean

        else:
            arr = np.asarray(data)
            if columns is not None:
                cols = np.asarray(columns, dtype=int)
                out = arr.copy()
                out[:, cols] = out[:, cols] * self.std[cols] + self.mean[cols]
                return out
            return arr * self.std + self.mean


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        if torch.is_tensor(data):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)

        self.min = arr.min(axis=0)
        self.max = arr.max(axis=0)

        data_range = self.max - self.min
        data_range[data_range == 0.0] = 1.0
        self.data_range_ = data_range


    def transform(self, data, columns=None):
        if self.min is None or self.max is None:
            raise RuntimeError("MinMaxScaler: call .fit(...) before .transform(...)")

        if torch.is_tensor(data):
            dmin   = torch.as_tensor(self.min, dtype=data.dtype, device=data.device)
            drange = torch.as_tensor(self.max - self.min, dtype=data.dtype, device=data.device)

            if columns is not None:
                cols = columns if isinstance(columns, torch.Tensor) else torch.tensor(columns, dtype=torch.long, device=data.device)
                dmin   = dmin[cols]
                drange = drange[cols]
                view_shape = [len(cols)] + [1] * (data.ndim - 1)
                dmin   = dmin.view(*view_shape)
                drange = drange.view(*view_shape)

            X_std    = (data - dmin) / drange
            return X_std

        else:
            arr = np.asarray(data)
            X = arr.copy()
            if columns is not None:
                cols = np.asarray(columns, dtype=int)
                Xc = X[:, cols]
                X[:, cols] = ((Xc - self.min[cols]) / self.data_range_[cols])
                return X
            return ((X - self.min) / self.data_range_)

    def inverse_transform(self, data, columns=None):
        if self.min is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScaler: call .fit(...) before .inverse_transform(...)")

        if torch.is_tensor(data):
            dmin   = torch.as_tensor(self.min, dtype=data.dtype, device=data.device)
            drange = torch.as_tensor(self.data_range_, dtype=data.dtype, device=data.device)

            if columns is not None:
                cols = columns if isinstance(columns, torch.Tensor) else torch.tensor(columns, dtype=torch.long, device=data.device)
                dmin   = dmin[cols]
                drange = drange[cols]
                view_shape = [len(cols)] + [1] * (data.ndim - 1)
                dmin   = dmin.view(*view_shape)
                drange = drange.view(*view_shape)

            return data * drange + dmin

        else:
            arr = np.asarray(data)
            X = arr.copy()
            if columns is not None:
                cols = np.asarray(columns, dtype=int)
                Xc = X[:, cols]
                X[:, cols] = Xc * self.data_range_[cols] + self.min[cols]
                return X
            return X * self.data_range_ + self.min