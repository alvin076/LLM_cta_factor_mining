"""Factor library functions — pandas port of the GP engine operator set.

Every operator from gp_factor_mining/gp_engine.py, reimplemented in
pandas (rolling-based instead of sliding_window_view) so LLM-generated
formulas can use them as building blocks.

IMPORTANT: The first docstring line of each function is auto-extracted
into the LLM system prompt (see agent/factor_gen.py). Keep it a single
line describing what the function does, in Chinese.
"""

import numpy as np
import pandas as pd


def _coerce(x, ref=None):
    """Convert scalar to float, keep Series as-is."""
    if isinstance(x, pd.Series):
        return x
    return float(np.asarray(x, dtype=np.float64).ravel()[0])


# ============================================================================
# 时序算子 (timing operators)
# ============================================================================

def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """滚动均值。窗口内平均值，刻画趋势中枢。"""
    return series.rolling(window).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    """滚动标准差(ddof=1)。窗口内波动幅度，刻画波动率水平。"""
    return series.rolling(window).std(ddof=1)


def ts_median(series: pd.Series, window: int) -> pd.Series:
    """滚动中位数。窗口内中值，对异常值稳健的趋势中枢。"""
    return series.rolling(window).median()


def ts_mad(series: pd.Series, window: int) -> pd.Series:
    """滚动中位绝对偏差。窗口内偏离中位数的幅度，稳健波动率。"""
    def _mad(x):
        med = np.median(x)
        return float(np.median(np.abs(x - med)))
    return series.rolling(window).apply(_mad, raw=True)


def ts_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """滚动相关系数。两序列窗口内线性相关性，如量价相关性。"""
    return s1.rolling(window).corr(s2)


def ts_cov(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """滚动协方差。两序列窗口内共变性，量级随波动放大。"""
    return s1.rolling(window).cov(s2)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """滚动百分位排名(0~1)。当前值在窗口内的相对位置，1=最高。"""
    def _rank(x):
        if len(x) < 2:
            return np.nan
        return float((x[:-1] < x[-1]).sum()) / (len(x) - 1)
    return series.rolling(window).apply(_rank, raw=True)


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """滚动最大值。窗口内最高值，如近期高点。"""
    return series.rolling(window).max()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """滚动最小值。窗口内最低值，如近期低点。"""
    return series.rolling(window).min()


def ts_slope(series: pd.Series, window: int) -> pd.Series:
    """滚动线性回归斜率。窗口内每期平均变化量，趋势强度与方向。"""
    def _slope(x):
        t = np.arange(len(x), dtype=np.float64)
        t_mean = t.mean()
        y_mean = x.mean()
        denom = ((t - t_mean) ** 2).sum()
        if denom == 0:
            return np.nan
        return float(((x - y_mean) * (t - t_mean)).sum() / denom)
    return series.rolling(window).apply(_slope, raw=True)


def ts_delay(series: pd.Series, periods: int) -> pd.Series:
    """滞后取值。取N期前的值，防止未来函数。"""
    return series.shift(periods)


def ts_delta(series: pd.Series, periods: int) -> pd.Series:
    """N期差分。当前值减N期前的值，刻画短期变化量。"""
    return series - series.shift(periods)


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """滚动求和。窗口内累计值，如成交量累计。"""
    return series.rolling(window).sum()


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """距窗口内最大值的时间。当前距离最高值出现时的期数。"""
    def _argmax(x):
        return float(len(x) - 1 - np.argmax(x))
    return series.rolling(window).apply(_argmax, raw=True)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """距窗口内最小值的时间。当前距离最低值出现时的期数。"""
    def _argmin(x):
        return float(len(x) - 1 - np.argmin(x))
    return series.rolling(window).apply(_argmin, raw=True)


def ts_skew(series: pd.Series, window: int) -> pd.Series:
    """滚动偏度。窗口内分布不对称性，正值=右偏(长尾在上)。"""
    def _skew(x):
        m1 = x.mean()
        m2 = ((x - m1) ** 2).mean()
        m3 = ((x - m1) ** 3).mean()
        return m3 / (m2 ** 1.5 + 1e-8)
    return series.rolling(window).apply(_skew, raw=True)


def ts_kurt(series: pd.Series, window: int) -> pd.Series:
    """滚动超额峰度。窗口内分布尖峭程度，>0=厚尾/极端值聚集。"""
    def _kurt(x):
        m1 = x.mean()
        m2 = ((x - m1) ** 2).mean()
        m4 = ((x - m1) ** 4).mean()
        return m4 / (m2 ** 2 + 1e-8) - 3.0
    return series.rolling(window).apply(_kurt, raw=True)


# ============================================================================
# 基础/条件算子 (elementwise operators)
# ============================================================================

def protected_div(x, y):
    """安全除法。分母绝对值小于1e-8时返回1.0，防止除零爆炸。"""
    x = _coerce(x)
    y = _coerce(y)
    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        idx = x.index if isinstance(x, pd.Series) else y.index
        res = np.where(np.abs(np.asarray(y, dtype=np.float64)) > 1e-8,
                       np.asarray(x, dtype=np.float64) / np.asarray(y, dtype=np.float64),
                       1.0)
        return pd.Series(res, index=idx)
    return float(x) / float(y) if abs(float(y)) > 1e-8 else 1.0


def protected_sqrt(x):
    """安全开方。负数返回0.0，非负正常开方。"""
    x = _coerce(x)
    if isinstance(x, pd.Series):
        arr = np.asarray(x, dtype=np.float64)
        return pd.Series(np.where(arr >= 0, np.sqrt(arr), 0.0), index=x.index)
    return float(np.sqrt(x)) if x >= 0 else 0.0


def protected_log(x):
    """安全对数。非正值返回0.0，正值正常取对数。"""
    x = _coerce(x)
    if isinstance(x, pd.Series):
        arr = np.asarray(x, dtype=np.float64)
        return pd.Series(np.where(arr > 0, np.log(arr), 0.0), index=x.index)
    return float(np.log(x)) if x > 0 else 0.0


def abs_op(x):
    """绝对值。保留正负号去除，衡量偏离幅度。"""
    return np.abs(x)


def if_else(cond, x, y):
    """条件选择。cond为NaN或0时取y，否则取x，向量化分支。"""
    cond_s = cond if isinstance(cond, pd.Series) else float(cond)
    x_s = x if isinstance(x, pd.Series) else float(x)
    y_s = y if isinstance(y, pd.Series) else float(y)

    if isinstance(cond_s, pd.Series) or isinstance(x_s, pd.Series) or isinstance(y_s, pd.Series):
        idx = None
        for s in (cond_s, x_s, y_s):
            if isinstance(s, pd.Series):
                idx = s.index
                break
        c = np.asarray(cond_s, dtype=np.float64)
        xa = np.asarray(x_s, dtype=np.float64)
        ya = np.asarray(y_s, dtype=np.float64)
        res = np.where(np.isnan(c) | (c == 0), ya, xa)
        return pd.Series(res, index=idx)
    return x_s if (not np.isnan(cond_s) and cond_s != 0) else y_s


def isnan(x):
    """NaN检测。输入为NaN返回1.0，否则0.0，用于掩码构造。"""
    if isinstance(x, pd.Series):
        return x.isna().astype(np.float64)
    return 1.0 if np.isnan(x) else 0.0


# All callable functions keyed by name for namespace injection
FACTOR_FUNCTIONS = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_median": ts_median,
    "ts_mad": ts_mad,
    "ts_corr": ts_corr,
    "ts_cov": ts_cov,
    "ts_rank": ts_rank,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_slope": ts_slope,
    "ts_delay": ts_delay,
    "ts_delta": ts_delta,
    "ts_sum": ts_sum,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "protected_div": protected_div,
    "protected_sqrt": protected_sqrt,
    "protected_log": protected_log,
    "abs": abs_op,
    "if_else": if_else,
    "isnan": isnan,
}
