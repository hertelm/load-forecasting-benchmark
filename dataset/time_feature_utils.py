import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import holidays

def dyn_time_features(index, dyn_time_dict):

    time_dict = {
        'minute': ['m', 60],
        'hour': ['h', 24],
        'weekday': ['D', 7],
        'month': ['M', 12],
    }

    feature_list = []

    dyn_time_df = pd.DataFrame(index=index)

    for time in dyn_time_dict.keys():
        if time in time_dict:
            time_feature = getattr(index, time)
            if dyn_time_dict[time] == 'sin_cos':
                time_features = sin_cos_encoding(time_feature, time_dict[time][1])
                feature_list += [time + '_sin', time + '_cos']
                dyn_time_df[time + '_sin'] = time_features[0]
                dyn_time_df[time + '_cos'] = time_features[1]
            elif dyn_time_dict[time] == 'raw':
                feature_list += [time]
                dyn_time_df[time] = time_feature
            else:
                raise NotImplementedError(f'time feature {dyn_time_dict[time]} not implemented. Please choose from sin_cos or raw')
        elif time == 'holiday':
            holiday = getattr(holidays, dyn_time_dict[time])()
            time_feature = index.map(lambda x: float(x in holiday))
            feature_list += [time]
            dyn_time_df[time] = time_feature
        else:
            raise NotImplementedError(f'time feature {time} not implemented. Please choose from {list(time_dict.keys()) + ["holiday"]}')

    return dyn_time_df, feature_list


def static_time_features(time_values, static_time_dict):

    time_dict = {
        'hour': ['h', 24],
        'weekday': ['D', 7],
        'month': ['M', 12],
    }

    idx = pd.to_datetime(time_values)

    feature_list = []
    time_features = []

    for time in static_time_dict.keys():
        if time in time_dict:
            time_feature = torch.tensor(idx.tz_localize(None).values.astype('datetime64[{}]'.format(time_dict[time][0])).astype('int') % time_dict[time][1])
            if static_time_dict[time] == 'sin_cos':
                time_feature = sin_cos_encoding(time_feature, time_dict[time][1])
                feature_list += [time + '_sin', time + '_cos']
                time_features += [torch.stack(time_feature, dim=1)]
            elif static_time_dict[time] == 'one_hot':
                time_feature = F.one_hot(time_feature, num_classes=time_dict[time][1])
                feature_list += [time + '_' + str(i) for i in range(time_dict[time][1])]
                time_features += [time_feature]
            elif static_time_dict[time] == 'raw':
                feature_list += [time]
                time_features += [time_feature.unsqueeze(-1)]
            else:
                raise NotImplementedError(f'time feature {static_time_dict[time]} not implemented. Please choose from sin_cos or raw')
        elif time == 'holiday':
            holiday = getattr(holidays, static_time_dict[time])()
            time_feature = torch.tensor([float(pd.to_datetime(t) in holiday) for t in idx])
            feature_list += [time]
            time_features += [time_feature.unsqueeze(-1)]
        elif time == 'workday':
            holiday = getattr(holidays, static_time_dict[time])()
            time_feature = torch.tensor([float((pd.to_datetime(t) not in holiday) and (pd.to_datetime(t).weekday() < 5)) for t in idx])
            feature_list += [time]
            time_features += [time_feature.unsqueeze(-1)]
        elif time == 'tomorrow_workday':
            holiday = getattr(holidays, static_time_dict[time])()
            time_feature = torch.tensor([float((pd.to_datetime(t) + pd.Timedelta(days=1) not in holiday) and (pd.to_datetime(t).weekday() < 5)) for t in idx])
            feature_list += [time]
            time_features += [time_feature.unsqueeze(-1)]
        else:
            raise NotImplementedError(f'time feature {time} not implemented. Please choose from {list(time_dict.keys()) + ["holiday", "workday", "tomorrow_workday"]}')

    return torch.cat(time_features, dim=1), feature_list


def sin_cos_encoding(time, period_length):
    """
    Encodes a cyclical time feature using cosine and sine functions.

    :param time: the value to transform
    :param period_length: the length of the period (e.g. 24 for a hourly value)
    :return: the cyclical feature
    """
    return [
        np.sin(2 * np.pi * time / period_length),
        np.cos(2 * np.pi * time / period_length)
    ]
