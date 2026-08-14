"""Backtest metrics: Sharpe, trade count, roughness."""

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365 * 24


def annualized_sharpe(returns: pd.Series) -> float:
    """
    Annualized Sharpe ratio from period returns.

    Assumes 1H returns, annualizes with sqrt(365*24).
    Ignores risk-free rate (crypto context).
    """
    valid = returns.dropna()
    if len(valid) == 0 or valid.std() == 0:
        return 0.0
    return float(valid.mean() / valid.std() * np.sqrt(HOURS_PER_YEAR))


def trade_count(positions: pd.Series) -> int:
    """Count position changes (entry + exit count)."""
    changes = positions.diff().fillna(0.0)
    return int((changes != 0).sum())


def roughness(sharpe_values: np.ndarray, axis: int) -> float:
    """
    Compute roughness of a 2D Sharpe grid along one axis.

    roughness = mean absolute difference between adjacent points.

    Args:
        sharpe_values: 2D array of Sharpe ratios [n_windows, n_thresholds]
        axis: 0 for window direction (vertical neighbors), 1 for threshold direction (horizontal)

    Returns:
        Mean absolute difference between adjacent grid points
    """
    if sharpe_values.ndim != 2:
        raise ValueError(f"Expected 2D array, got {sharpe_values.ndim}D")

    diffs = np.abs(np.diff(sharpe_values, axis=axis))
    return float(np.nanmean(diffs))


def compute_roughness(
    sharpe_values: np.ndarray,
    windows: list,
    thresholds: list,
) -> dict:
    """
    Compute roughness in both directions and combined.

    Returns:
        dict with keys: window_roughness, threshold_roughness, combined
    """
    window_rough = roughness(sharpe_values, axis=0)
    threshold_rough = roughness(sharpe_values, axis=1)
    combined = (window_rough + threshold_rough) / 2.0

    return {
        "window_roughness": round(window_rough, 4),
        "threshold_roughness": round(threshold_rough, 4),
        "combined": round(combined, 4),
    }


def sharpe_positive_ratio(sharpe_values: np.ndarray) -> float:
    """Fraction of grid points with Sharpe > 0."""
    total = sharpe_values.size
    if total == 0:
        return 0.0
    return float((sharpe_values > 0).sum() / total)
