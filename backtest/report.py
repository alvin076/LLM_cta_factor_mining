"""Format grid search results as LLM-readable reports."""

import numpy as np


def is_report(
    result: dict,
    formula: str,
    category: str,
    symbol: str = "BTCUSDT",
    max_heatmap_rows: int = 10,
) -> str:
    """
    Format IS grid search results for LLM evaluation.

    Returns a compact Chinese report string.
    """
    windows = result["windows"]
    thresholds = result["thresholds"]
    sharpe_grid = result["sharpe_grid"]
    trade_grid = result["trade_grid"]
    roughness = result["roughness"]
    results = result["results"]

    lines = []
    lines.append(f"IS 网格搜索 — {symbol}")
    lines.append(f"因子: {formula[:120]}")
    if category:
        lines.append(f"类别: {category}")

    lines.append("")
    lines.append(_format_heatmap(windows, thresholds, sharpe_grid, max_heatmap_rows))

    lines.append("")
    lines.append("参数稳定性(粗糙度, 越小越稳定):")
    lines.append(f"  窗口方向(w±50):  {roughness['window_roughness']}")
    lines.append(f"  阈值方向(th±0.2): {roughness['threshold_roughness']}")
    lines.append(f"  综合粗糙度:       {roughness['combined']}")

    lines.append("")
    lines.append("Sharpe统计:")
    lines.append(f"  均值: {result['sharpe_mean']}, 标准差: {result['sharpe_std']}, 最大值: {result['sharpe_max']}")
    lines.append(f"  Sharpe>0占比: {result['sharpe_positive_ratio']:.0%}")

    lines.append("")
    lines.append("典型交易次数 (部分):")
    sample_combos = []
    mid_w = len(windows) // 2
    sample_combos.append((0, 0))
    sample_combos.append((mid_w, len(thresholds) // 2))
    sample_combos.append((len(windows) - 1, len(thresholds) - 1))
    sample_combos.append((0, len(thresholds) - 1))
    for wi, ti in sample_combos:
        if wi < len(windows) and ti < len(thresholds):
            lines.append(f"  w={windows[wi]}, th={thresholds[ti]}: trades={int(trade_grid[wi, ti])}")

    lines.append("")
    lines.append("请列出你认为值得进入OOS测试的所有(w, th)参数组合。")
    lines.append("筛选标准: IS Sharpe>0, 交易次数>50, 参数表面平滑(粗糙度低)。")
    lines.append("用列表格式回复: [(200, 0.4), (150, 0.6), ...]")

    return "\n".join(lines)


def oos_report(
    is_result: dict,
    oos_result: dict,
    selected_params: list,
    symbol: str = "BTCUSDT",
    max_heatmap_rows: int = 10,
) -> str:
    """
    Format OOS grid search results for LLM evaluation.

    Compares IS vs OOS performance for selected parameters.
    """
    windows = oos_result["windows"]
    thresholds = oos_result["thresholds"]
    oos_sharpe_grid = oos_result["sharpe_grid"]
    is_sharpe_grid = is_result["sharpe_grid"]
    oos_roughness = oos_result["roughness"]
    is_roughness = is_result["roughness"]

    lines = []
    lines.append(f"OOS测试 — {symbol} (最后8760小时)")
    lines.append("")

    lines.append("OOS Sharpe Heatmap:")
    lines.append(_format_heatmap(windows, thresholds, oos_sharpe_grid, max_heatmap_rows))

    lines.append("")
    lines.append("OOS粗糙度 (vs IS):")
    roughness_ratio = _safe_ratio(oos_roughness["combined"], is_roughness["combined"])
    lines.append(f"  窗口方向(w±50):  {oos_roughness['window_roughness']} (IS: {is_roughness['window_roughness']})")
    lines.append(f"  阈值方向(th±0.2): {oos_roughness['threshold_roughness']} (IS: {is_roughness['threshold_roughness']})")
    lines.append(f"  综合粗糙度:       {oos_roughness['combined']} (IS: {is_roughness['combined']}, 恶化{roughness_ratio}x)")

    lines.append("")
    lines.append("选中参数在IS和OOS的对比:")

    pass_count = 0
    # Build lookup for IS Sharpe by (window, threshold)
    is_lookup = {}
    for r in is_result["results"]:
        is_lookup[(r["window"], r["threshold"])] = r

    oos_lookup = {}
    for r in oos_result["results"]:
        oos_lookup[(r["window"], r["threshold"])] = r

    oos_sharpes = []
    is_sharpes = []
    for i, (w, th) in enumerate(selected_params, 1):
        is_r = is_lookup.get((w, th))
        oos_r = oos_lookup.get((w, th))

        is_sharpe = is_r["sharpe"] if is_r else "N/A"
        oos_sharpe = oos_r["sharpe"] if oos_r else "N/A"
        oos_trades = oos_r["n_trades"] if oos_r else "N/A"

        passed = "✓" if (isinstance(oos_sharpe, (int, float)) and oos_sharpe > 0) else "✗"
        if passed == "✓":
            pass_count += 1

        if isinstance(oos_sharpe, (int, float)):
            oos_sharpes.append(oos_sharpe)
        if isinstance(is_sharpe, (int, float)):
            is_sharpes.append(is_sharpe)

        lines.append(f"  {i}. ({w}, {th}): IS={is_sharpe} → OOS={oos_sharpe}, trades={oos_trades} {passed}")

    lines.append("")
    total = len(selected_params)
    pass_rate = f"{pass_count}/{total} ({pass_count / total:.0%})" if total > 0 else "N/A"
    lines.append(f"OOS通过率(Sharpe>0): {pass_rate}")

    if oos_sharpes and is_sharpes:
        median_oos = np.median(oos_sharpes)
        median_is = np.median(is_sharpes)
        retention = _safe_ratio(median_oos, median_is)
        lines.append(f"IS→OOS Sharpe中位数保持率: {retention:.0%} (IS={median_is:.3f}, OOS={median_oos:.3f})")

    is_degradation = _safe_ratio(oos_roughness["combined"], is_roughness["combined"])
    lines.append(f"粗糙度恶化倍数: {is_degradation:.1f}x")

    lines.append("")
    lines.append("请判断该因子是否通过OOS检验。")
    lines.append("如果失败，分析原因并建议改进方向。")

    return "\n".join(lines)


def _format_heatmap(windows, thresholds, sharpe_grid, max_rows):
    limit = min(len(windows), max_rows)
    header = "        " + " ".join(f"th={t:<4.1f}" for t in thresholds)
    rows = [header]

    for i in range(limit):
        vals = " ".join(f"{sharpe_grid[i, j]:>7.2f}" if not np.isnan(sharpe_grid[i, j]) else "   N/A " for j in range(len(thresholds)))
        rows.append(f"w={windows[i]:>4}: {vals}")

    if len(windows) > max_rows:
        rows.append("  ... (已截断)")

    return "\n".join(rows)


def _safe_ratio(numer, denom):
    if denom and denom != 0:
        return round(numer / denom, 2)
    return float('inf')


def format_params_list(params: list) -> str:
    return str(params)
