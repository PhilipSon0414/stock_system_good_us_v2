"""Date-grouped chronological splits with label-window purging."""
import numpy as np
import pandas as pd
from market_clock import session_offset

def purged_split(data, fraction, horizon):
    dates = np.sort(data['date'].unique())
    if len(dates) < 4:
        raise ValueError('Not enough dates for chronological validation')
    boundary = pd.Timestamp(dates[max(1, min(len(dates)-1, int(len(dates)*fraction)))])
    cutoff = pd.Timestamp(session_offset(boundary, -horizon))
    train = data[data.date < cutoff].copy()
    valid = data[data.date >= boundary].copy()
    if train.empty or valid.empty:
        raise ValueError('Not enough history after purging label windows')
    return train, valid
