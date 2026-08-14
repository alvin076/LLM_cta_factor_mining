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
    std = roll.std(ddof=0)
    zscore = (series - mean) / std.replace(0.0, np.nan)

    return zscore


def generate_positions(indicator: pd.Series, threshold: float) -> pd.Series:
    """
    Generate trading positions from indicator signals.

    indicator > +threshold  ->  long  (1)
    indicator < -threshold  ->  short (-1)
    otherwise               ->  flat  (0)

    Positions are forward-filled: once in a position, stay there
    until the opposite signal is triggered.
    """
    threshold = abs(threshold)

    raw_signal = pd.Series(0, index=indicator.index, dtype=float)
    raw_signal[indicator > threshold] = 1.0
    raw_signal[indicator < -threshold] = -1.0

    positions = raw_signal.replace(0.0, np.nan).ffill().fillna(0.0)

    return positions
