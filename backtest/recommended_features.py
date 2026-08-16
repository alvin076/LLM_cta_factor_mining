"""Recommended high-semantic features — pandas port of data_pipeline.py.

The 20 features from gp_factor_mining/data_pipeline.py::compute_20_features,
reimplemented in pandas (.shift/.pct_change replace np.roll, no future leak)
and injected into the safe_eval namespace so LLM-generated formulas can
reference them directly, e.g. factor = POSITION * VOL_RATIO.

RECOMMENDED_FEATURE_DOCS is auto-extracted into the LLM system prompt
(see agent/factor_gen.py). Keep each description a single line in Chinese.
"""

import numpy as np
import pandas as pd


def _position(close, low, high, d=20):
    low_d = low.rolling(d).min()
    high_d = high.rolling(d).max()
    return (close - low_d) / (high_d - low_d + 1e-8)


def _vol_atr(close, high, low, d=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(d).mean()


def _hour_sin(index: pd.Index) -> pd.Series:
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(
            np.sin(2 * np.pi * index.hour.values / 24.0), index=index,
        )
    return pd.Series(np.nan, index=index)


def _weekend_flag(index: pd.Index) -> pd.Series:
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(
            (index.dayofweek.values >= 5).astype(np.float64), index=index,
        )
    return pd.Series(np.nan, index=index)


def compute_recommended_features(df: pd.DataFrame) -> dict:
    """Compute the 20 recommended features from OHLCV columns.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume.
            If index is a DatetimeIndex, HOUR_SIN / WEEKEND_FLAG are
            computed from it; otherwise they are all-NaN.

    Returns:
        dict of name -> pd.Series aligned to df.index
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]
    idx = df.index

    vol_atr = _vol_atr(close, high, low)
    amount = close * volume

    return {
        "RET_1": close.pct_change(1),
        "RET_5": close.pct_change(5),
        "ROC_10": close.pct_change(10),
        "POSITION": _position(close, low, high),
        "VOL_ATR": vol_atr,
        "ATR_CHG": vol_atr.pct_change(10),
        "SKEW": (close - low) - (high - close),
        "GAP": open_ - close.shift(1),
        "VOL_RATIO": volume / (volume.rolling(20, min_periods=1).mean() + 1e-8),
        "AMOUNT": amount,
        "AMOUNT_RATIO": amount / (amount.rolling(20, min_periods=1).mean() + 1e-8),
        "HL_RATIO": high / (low + 1e-8),
        "HL_SPREAD": (high - low) / (close + 1e-8),
        "VWAP": (high + low + close) / 3.0,
        "C2V_RATIO": close / (volume + 1.0),
        "LOG_VOL": np.log(volume + 1.0),
        "HOUR_SIN": _hour_sin(idx),
        "WEEKEND_FLAG": _weekend_flag(idx),
        "RET_24": close.pct_change(24),
        "RET_168": close.pct_change(168),
    }


RECOMMENDED_FEATURE_DOCS = {
    "RET_1": "单期收益率。最新1小时涨跌幅，动量/反转的基础量。",
    "RET_5": "5期收益率。约5小时动量，捕捉短中期趋势。",
    "ROC_10": "10期变动率。约10小时涨跌幅，中期动量指标。",
    "POSITION": "20期通道百分位。当前价在20小时高低区间中的相对位置(0~1)。",
    "VOL_ATR": "14期平均真实波幅。波动率水平，衡量市场活跃度。",
    "ATR_CHG": "波动率加速度。ATR相对10期前的变化率，波动扩张/收缩信号。",
    "SKEW": "日内阴阳实体差。实体重心偏向买方或卖方的程度。",
    "GAP": "跳空缺口。开盘价相对前收盘的缺口，情绪/流动性变化。",
    "VOL_RATIO": "量比。成交量相对20期均值的倍数，放量/缩量信号。",
    "AMOUNT": "成交额。价格与成交量的乘积，资金规模。",
    "AMOUNT_RATIO": "成交额比。成交额相对20期均值的倍数，资金活跃度。",
    "HL_RATIO": "日内振幅倍数。最高价/最低价，日内极端波动范围。",
    "HL_SPREAD": "归一化日内振幅。高低价差除以收盘价，波动烈度。",
    "VWAP": "日内典型均价。(高+低+收)/3，价格中枢参考。",
    "C2V_RATIO": "单位成交量价格。收盘价/成交量，拉升所需资金效率。",
    "LOG_VOL": "对数成交量。压缩尺度的成交量，处理长尾分布。",
    "HOUR_SIN": "小时正弦编码。日内时间相位(-1~1)，捕捉日内周期规律。",
    "WEEKEND_FLAG": "周末标志。周六周日=1，其余=0，捕捉周末效应。",
    "RET_24": "日收益率。24小时涨跌幅，日频动量。",
    "RET_168": "周收益率。168小时涨跌幅，周频动量。",
}
