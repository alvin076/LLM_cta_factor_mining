"""Deterministic IS screening — replaces the LLM IS judge.

Screening chain (each step rejects with a learning message):
1. sharpe_max >= is_sharpe_min
2. sharpe>0 ratio >= is_positive_ratio_min
3. select params with sharpe >= is_sharpe_min
4. selected count >= min_selected_params (no single-point luck)
5. roughness among adjacent selected params < selected_roughness_max
6. every selected param's trade count within [trades_min, trades_max]
"""

import numpy as np


def select_params(results: list, sharpe_min: float) -> list:
    """Return (window, threshold) combos with sharpe >= sharpe_min."""
    return [(r["window"], r["threshold"]) for r in results
            if r["sharpe"] >= sharpe_min]


def selected_roughness(results: list, selected: list,
                       windows: list, thresholds: list) -> dict:
    """
    Roughness among adjacent SELECTED params only.

    Adjacent pairs are counted only if BOTH are selected:
    - window direction: (w, th) vs (w+50, th)
    - threshold direction: (w, th) vs (w, th+0.2)
    """
    w_idx = {w: i for i, w in enumerate(windows)}
    t_idx = {t: j for j, t in enumerate(thresholds)}
    selected_set = set(selected)
    sharpe_lookup = {(r["window"], r["threshold"]): r["sharpe"] for r in results}

    window_diffs = []
    threshold_diffs = []

    for (w, th) in selected:
        w_next = w + (windows[1] - windows[0]) if len(windows) > 1 else None
        if w_next is not None and (w_next, th) in selected_set:
            window_diffs.append(abs(
                sharpe_lookup[(w, th)] - sharpe_lookup[(w_next, th)]
            ))

        t_next = round(th + (thresholds[1] - thresholds[0]), 2) if len(thresholds) > 1 else None
        if t_next is not None and (w, t_next) in selected_set:
            threshold_diffs.append(abs(
                sharpe_lookup[(w, th)] - sharpe_lookup[(w, t_next)]
            ))

    window_rough = float(np.mean(window_diffs)) if window_diffs else 0.0
    threshold_rough = float(np.mean(threshold_diffs)) if threshold_diffs else 0.0
    combined = (window_rough + threshold_rough) / 2.0

    return {
        "window_roughness": round(window_rough, 4),
        "threshold_roughness": round(threshold_rough, 4),
        "combined": round(combined, 4),
        "n_adjacent_pairs": len(window_diffs) + len(threshold_diffs),
    }


def screen_factor(is_result: dict, cfg: dict) -> dict:
    """
    Deterministic IS screening.

    Args:
        is_result: grid_search() result dict
        cfg: screening config from agent.config.load_config()

    Returns:
        dict with keys: passed, reason, selected_params, selected_stats
    """
    results = is_result["results"]
    windows = is_result["windows"]
    thresholds = is_result["thresholds"]

    # 1. sharpe_max >= threshold
    if is_result["sharpe_max"] < cfg["is_sharpe_min"]:
        return {
            "passed": False,
            "reason": f"IS Sharpe最大值{is_result['sharpe_max']}未达{cfg['is_sharpe_min']}",
            "selected_params": [],
            "selected_stats": None,
        }

    # 2. positive ratio
    if is_result["sharpe_positive_ratio"] < cfg["is_positive_ratio_min"]:
        return {
            "passed": False,
            "reason": (f"IS Sharpe>0占比{is_result['sharpe_positive_ratio']:.0%}"
                       f"不足{cfg['is_positive_ratio_min']:.0%}"),
            "selected_params": [],
            "selected_stats": None,
        }

    # 3. select params
    selected = select_params(results, cfg["is_sharpe_min"])

    # 4. min selected count (no single-point luck)
    if len(selected) < cfg["min_selected_params"]:
        return {
            "passed": False,
            "reason": f"入选参数仅{len(selected)}个，少于{cfg['min_selected_params']}，孤峰嫌疑",
            "selected_params": selected,
            "selected_stats": None,
        }

    # 5. roughness among adjacent selected params
    rough = selected_roughness(results, selected, windows, thresholds)
    if rough["combined"] >= cfg["selected_roughness_max"]:
        return {
            "passed": False,
            "reason": f"入选参数粗糙度{rough['combined']}过高（上限{cfg['selected_roughness_max']}）",
            "selected_params": selected,
            "selected_stats": {"roughness": rough},
        }

    # 6. trade count for every selected param
    trades_lookup = {(r["window"], r["threshold"]): r["n_trades"] for r in results}
    for (w, th) in selected:
        n = trades_lookup.get((w, th), 0)
        if not (cfg["selected_trades_min"] <= n <= cfg["selected_trades_max"]):
            return {
                "passed": False,
                "reason": f"入选参数({w},{th})交易次数{n}不合理",
                "selected_params": selected,
                "selected_stats": {"roughness": rough},
            }

    return {
        "passed": True,
        "reason": "",
        "selected_params": selected,
        "selected_stats": {
            "roughness": rough,
            "n_selected": len(selected),
            "sharpe_range": (
                min(r["sharpe"] for r in results if (r["window"], r["threshold"]) in set(selected)),
                max(r["sharpe"] for r in results if (r["window"], r["threshold"]) in set(selected)),
            ),
        },
    }


def screen_oos(oos_result: dict, selected_params: list, cfg: dict) -> dict:
    """
    Deterministic OOS screening (2 gates).

    Gate 1: at least 1 selected param with OOS Sharpe >= oos_sharpe_min
    Gate 2: OOS Sharpe>0 ratio among selected params >= oos_positive_ratio_min

    Args:
        oos_result: grid_search() result dict from OOS data
        selected_params: [(window, threshold), ...] chosen by IS screening
        cfg: screening config from agent.config.load_config()

    Returns:
        dict with keys: passed, reason, qualified, stats
    """
    oos_lookup = {(r["window"], r["threshold"]): r for r in oos_result["results"]}

    # Gate 1: at least 1 qualified param
    qualified = [
        (w, th, oos_lookup[(w, th)]["sharpe"])
        for w, th in selected_params
        if oos_lookup.get((w, th), {}).get("sharpe", 0) >= cfg["oos_sharpe_min"]
    ]
    if len(qualified) == 0:
        return {
            "passed": False,
            "reason": f"OOS无达标参数(Sharpe>={cfg['oos_sharpe_min']})",
            "qualified": [],
            "stats": None,
        }

    # Gate 2: Sharpe>0 ratio >= threshold
    pos_count = sum(1 for w, th in selected_params
                    if oos_lookup.get((w, th), {}).get("sharpe", 0) > 0)
    ratio = pos_count / len(selected_params) if selected_params else 0.0
    if ratio < cfg["oos_positive_ratio_min"]:
        return {
            "passed": False,
            "reason": (f"OOS Sharpe>0占比{ratio:.0%}"
                       f"不足{cfg['oos_positive_ratio_min']:.0%}"),
            "qualified": qualified,
            "stats": {"positive_ratio": round(ratio, 4)},
        }

    return {
        "passed": True,
        "reason": "",
        "qualified": qualified,
        "stats": {
            "positive_ratio": round(ratio, 4),
            "n_qualified": len(qualified),
            "qualified_sharpe_range": (
                min(s for _, _, s in qualified),
                max(s for _, _, s in qualified),
            ),
        },
    }
