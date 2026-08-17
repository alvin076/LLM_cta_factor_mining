"""Judge Agents — IS comparative evaluation (legacy) and OOS failure analyst."""

import ast
import re

IS_JUDGE_SYSTEM = r"""
你是一位资深CTA策略评审官。你会同时收到多个因子的IS网格搜索结果，
需要横向比较后决定哪些因子值得进入OOS测试。

## 判断标准
1. IS Sharpe最大值 > 0.3
2. 粗糙度低 (< 0.15): 相邻参数Sharpe变化小，参数表面平滑
3. 交易次数合理 (50-5000)
4. Sharpe>0占比 > 50%

## 输出格式（严格遵循）
1. 先给每个因子一行点评：通过/不通过 + 一句话理由
2. 对通过IS检验的因子，列出值得进入OOS测试的参数组合
3. 最后一行必须是:
[SELECTED] [(factor_index, [(window, threshold), ...]), ...]
其中 factor_index 是因子编号（从0开始）。
"""

OOS_FAIL_ANALYST_SYSTEM = r"""
你是一位资深CTA量化复盘分析师。一个因子通过了IS筛选但在OOS失败。
你的任务是对比 IS 与 OOS 报告的差异，分析衰减原因，给出具体改进建议。

## 分析角度
1. 对比 IS 与 OOS 的 Sharpe 热力图差异：峰值位置是否漂移、幅度衰减多少
2. 对比粗糙度：参数表面是否在OOS崩成锯齿（过拟合信号）
3. 结合市场环境变化思考：因子逻辑在近一年为何失效

## 输出格式（严格遵循）
[分析] <失败原因，对比IS/OOS差异，1-3句话>
[改进] <具体改进建议：换方向/调参数/加过滤条件，1-3句话>
"""

IS_FAIL_ANALYST_SYSTEM = r"""
你是一位资深CTA量化复盘分析师。一个因子未通过IS代码筛选（样本内就失败）。
你的任务是分析它为什么在样本内就失败，并给出改进建议。

## 分析角度
1. 看Sharpe热力图: 峰值是否够高、正收益区域是否成片
2. 看粗糙度: 参数表面是否锯齿化（对参数过度敏感）
3. 结合因子公式本身: 逻辑是否有缺陷、窗口是否不当、是否缺乏过滤条件

## 输出格式（严格遵循）
[分析] <失败原因，1-3句话>
[改进] <具体改进建议：换方向/调参数/加过滤条件，1-3句话>
"""

SELECTED_PATTERN = re.compile(r"\[SELECTED\]\s*\[(.+)\]", re.DOTALL)
FAIL_ANALYSIS_PATTERN = re.compile(r"\[分析\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
FAIL_IMPROVE_PATTERN = re.compile(r"\[改进\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)


def build_is_judge_messages(factor_reports: list) -> list:
    """
    factor_reports: list of (index, direction_name, is_report_text)
    """
    parts = []
    for idx, direction_name, report_text in factor_reports:
        parts.append(f"--- 因子 {idx}: {direction_name} ---\n{report_text}")

    user_msg = ("以下是本批因子的IS网格搜索结果，请横向比较并选择。\n\n"
                + "\n\n".join(parts))

    return [
        {"role": "system", "content": IS_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_is_selection(response: str) -> list:
    """
    Parse [SELECTED] [(factor_index, [(w, th), ...]), ...]
    Returns list of (factor_index, [(w, th), ...])
    """
    match = SELECTED_PATTERN.search(response)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(f"[{match.group(1)}]")
        result = []
        for idx, params in parsed:
            result.append((int(idx), [(int(w), float(th)) for w, th in params]))
        return result
    except (ValueError, SyntaxError, TypeError):
        return []


def build_failure_analyst_messages(is_report_text: str, oos_report_text: str,
                                   factor_info: str, fail_reason: str) -> list:
    user_msg = (f"因子信息：{factor_info}\n\n"
                f"代码门槛判定：{fail_reason}\n\n"
                f"=== IS 报告 ===\n{is_report_text}\n\n"
                f"=== OOS 报告 ===\n{oos_report_text}\n\n"
                f"请对比两份报告，分析OOS失败的原因并给出改进建议。")
    return [
        {"role": "system", "content": OOS_FAIL_ANALYST_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_failure_analysis(response: str) -> dict:
    analysis_m = FAIL_ANALYSIS_PATTERN.search(response)
    improve_m = FAIL_IMPROVE_PATTERN.search(response)
    return {
        "analysis": analysis_m.group(1).strip() if analysis_m else "",
        "improve": improve_m.group(1).strip() if improve_m else "",
    }


def ask_is_judge(chat_fn, factor_reports: list, model: str = None) -> dict:
    messages = build_is_judge_messages(factor_reports)
    result = chat_fn(messages, model=model, max_tokens=4096, temperature=0.3)
    response = result["answer"]
    return {
        "selected": parse_is_selection(response),
        "raw_response": response,
        "usage": result["usage"],
    }


def ask_oos_failure_analyst(chat_fn, is_report_text: str, oos_report_text: str,
                            factor_info: str, fail_reason: str,
                            model: str = None) -> dict:
    """LLM 复盘: 对比 IS/OOS 报告，分析失败原因 + 改进建议。

    Returns:
        dict with keys: analysis, improve, raw_response, usage
    """
    messages = build_failure_analyst_messages(
        is_report_text, oos_report_text, factor_info, fail_reason)
    result = chat_fn(messages, model=model, max_tokens=8192, temperature=0.4)
    response = result["answer"]
    analysis = parse_failure_analysis(response)
    analysis["raw_response"] = response
    analysis["usage"] = result["usage"]
    return analysis


def ask_is_failure_analyst(chat_fn, is_report_text: str,
                           factor_info: str, fail_reason: str,
                           model: str = None) -> dict:
    """LLM 复盘: IS 样本内失败的原因 + 改进建议。

    Returns:
        dict with keys: analysis, improve, raw_response, usage
    """
    user_msg = (f"因子信息：{factor_info}\n\n"
                f"代码门槛判定：{fail_reason}\n\n"
                f"=== IS 报告 ===\n{is_report_text}\n\n"
                f"请分析该因子在IS样本内失败的原因并给出改进建议。")
    messages = [
        {"role": "system", "content": IS_FAIL_ANALYST_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    result = chat_fn(messages, model=model, max_tokens=8192, temperature=0.4)
    response = result["answer"]
    analysis = parse_failure_analysis(response)
    analysis["raw_response"] = response
    analysis["usage"] = result["usage"]
    return analysis
