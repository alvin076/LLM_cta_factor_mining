"""Signal generation: rolling zscore -> long/short positions."""

import pandas as pd
import numpy as np


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score of a series.

    zscore[t] = (value[t] - mean[t-window:t]) / std[t-window:t]
    First `window` values will be NaN.
    """
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")

    roll = series.rolling(window)
    mean = roll.mean()
    std = roll.std()
    zscore = (series - mean) / std

    return zscore


def generate_positions(indicator: pd.Series, threshold: float) -> pd.Series:
    """
    Generate trading positions from indicator signals.

    indicator > +threshold  ->  long  (1)
    indicator < -threshold  ->  short (-1)
    otherwise               ->  flat  (0)

    三态信号，无持仓记忆：阈值内直接空仓；
    暖机期的 NaN 比较结果为 False，同样落为 0（空仓）。
    """
    threshold = abs(threshold)

    arr = np.where(indicator > threshold, 1.0,
                   np.where(indicator < -threshold, -1.0, 0.0))
    positions = pd.Series(arr, index=indicator.index)

    return positions
