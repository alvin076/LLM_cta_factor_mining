"""Config loader — IS screening thresholds with file override."""

import json
import os

DEFAULT_CONFIG = {
    "is_sharpe_min": 1.3,
    "is_positive_ratio_min": 0.80,
    "selected_roughness_max": 0.15,
    "selected_trades_min": 50,
    "selected_trades_max": 5000,
    "min_selected_params": 3,
    "oos_sharpe_min": 1.3,
    "oos_positive_ratio_min": 0.8,
}


def load_config(path: str = "config.json") -> dict:
    """
    Load config from JSON file. Missing keys fall back to defaults.

    Args:
        path: JSON config file path

    Returns:
        merged config dict
    """
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            cfg.update(file_cfg)
        except (json.JSONDecodeError, IOError):
            pass
    return cfg
