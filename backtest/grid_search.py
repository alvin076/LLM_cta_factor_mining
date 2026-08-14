"""Grid search over (window, threshold) parameter space."""

import numpy as np
import pandas as pd

from .safety import safe_eval
from .signal import rolling_zscore, generate_positions
from .metrics import annualized_sharpe, trade_count, compute_roughness, sharpe_positive_ratio


def grid_search(
    df: pd.DataFrame,
    formula: str,
    windows: list = None,
    thresholds: list = None,
    commission_bps: float = 6.0,
) -> dict:
    """
    Run grid search over (window, threshold) parameter space.

    Args:
        df: OHLCV DataFrame with columns Open, High, Low, Close, Volume
        formula: pandas factor expression
        windows: list of rolling window sizes (default: range(100, 1050, 50))
        thresholds: list of threshold values (default: [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    Returns:
        dict with keys: results (list of per-combo dicts), sharpe_grid (2D array),
                        windows, thresholds, roughness (dict), sharpe_positive_ratio,
                        sharpe_mean, sharpe_max, sharpe_std
    """
    if windows is None:
        windows = list(range(100, 1050, 50))
    if thresholds is None:
        thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    factor = safe_eval(formula, df)

    if factor.isna().all():
        return _empty_result(windows, thresholds, "Factor returned all NaN")

    forward_returns = df['Close'].pct_change().shift(-1)

    results = []
    sharpe_grid = np.full((len(windows), len(thresholds)), np.nan)
    trade_grid = np.full((len(windows), len(thresholds)), np.nan)

    for i, w in enumerate(windows):
        indicator = rolling_zscore(factor, w)
        for j, th in enumerate(thresholds):
            positions = generate_positions(indicator, th)
            gross_returns = positions * forward_returns
            if commission_bps > 0:
                turnover = positions.diff().abs().fillna(0)
                commission = turnover * (commission_bps / 10000)
                strategy_returns = gross_returns - commission
            else:
                strategy_returns = gross_returns

            sharpe = annualized_sharpe(strategy_returns)
            trades = trade_count(positions)

            sharpe_grid[i, j] = sharpe
            trade_grid[i, j] = trades
            results.append({
                "window": w,
                "threshold": th,
                "sharpe": round(float(sharpe), 4),
                "n_trades": trades,
            })

    roughness_metrics = compute_roughness(sharpe_grid, windows, thresholds)

    valid_sharpes = sharpe_grid[~np.isnan(sharpe_grid)]
    sharpe_mean = float(np.nanmean(sharpe_grid))
    sharpe_std = float(np.nanstd(sharpe_grid))
    sharpe_max = float(np.nanmax(sharpe_grid))

    return {
        "results": results,
        "sharpe_grid": sharpe_grid,
        "trade_grid": trade_grid,
        "windows": windows,
        "thresholds": thresholds,
        "roughness": roughness_metrics,
        "sharpe_positive_ratio": round(sharpe_positive_ratio(sharpe_grid), 4),
        "sharpe_mean": round(sharpe_mean, 4),
        "sharpe_std": round(sharpe_std, 4),
        "sharpe_max": round(sharpe_max, 4),
    }


def _empty_result(windows, thresholds, reason):
    results = []
    sharpe_grid = np.full((len(windows), len(thresholds)), np.nan)
    for i, w in enumerate(windows):
        for j, th in enumerate(thresholds):
            results.append({
                "window": w,
                "threshold": th,
                "sharpe": 0.0,
                "n_trades": 0,
            })
    return {
        "results": results,
        "sharpe_grid": sharpe_grid,
        "trade_grid": np.zeros_like(sharpe_grid),
        "windows": windows,
        "thresholds": thresholds,
        "roughness": {"window_roughness": 0.0, "threshold_roughness": 0.0, "combined": 0.0},
        "sharpe_positive_ratio": 0.0,
        "sharpe_mean": 0.0,
        "sharpe_std": 0.0,
        "sharpe_max": 0.0,
        "error": reason,
    }
