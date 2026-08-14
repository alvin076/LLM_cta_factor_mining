"""Factor library functions — vectorized, future-leak-safe.

These are injected into the safe_eval namespace so LLM-generated formulas
can use them as building blocks instead of raw pandas operations.

IMPORTANT: The first docstring line of each function is auto-extracted
into the LLM system prompt (see agent/factor_gen.py). Keep it a single
line describing what the function does, in Chinese.
"""

import numpy as np
import pandas as pd


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """移动平均。窗口内均值，刻画趋势中枢。"""
    return series.rolling(window).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    """移动标准差。窗口内波动幅度，刻画波动率水平。"""
    return series.rolling(window).std(ddof=0)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """百分位排名(0~1)。当前值在窗口内的相对位置，1=最高。"""
    def _rank(x):
        if len(x) < 2:
            return np.nan
        return float(x.rank(pct=True).iloc[-1])
    return series.rolling(window).apply(_rank, raw=False)


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """滚动求和。窗口内累计值，如成交量累计。"""
    return series.rolling(window).sum()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """滚动最大值。窗口内最高价，如近期高点。"""
    return series.rolling(window).max()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """滚动最小值。窗口内最低价，如近期低点。"""
    return series.rolling(window).min()


def corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """滚动相关系数。两序列窗口内相关性，如量价相关性。"""
    return s1.rolling(window).corr(s2)


def delay(series: pd.Series, periods: int) -> pd.Series:
    """滞后取值。取N期前的值，防止未来函数。"""
    return series.shift(periods)


def signed_power(series: pd.Series, exponent: float) -> pd.Series:
    """带符号幂变换。保留符号、压缩幅度，如对收益率开方。"""
    return np.sign(series) * np.abs(series) ** exponent


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """线性加权衰减平均。窗口内近端权重高，对近期变化更敏感。"""
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()

    def _wmean(x):
        if len(x) < window:
            return np.nan
        return np.dot(x[-window:], weights)

    return series.rolling(window).apply(_wmean, raw=True)


# All callable functions keyed by name for namespace injection
FACTOR_FUNCTIONS = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_rank": ts_rank,
    "ts_sum": ts_sum,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "corr": corr,
    "delay": delay,
    "signed_power": signed_power,
    "decay_linear": decay_linear,
}
