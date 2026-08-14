"""Judge Agents — IS comparative evaluation and OOS verdict."""

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

OOS_JUDGE_SYSTEM = r"""
你是一位资深CTA策略评审官。你会收到因子在OOS（样本外）的表现，
需要判断该因子是否真正通过检验，或只是IS过拟合。

## 判断标准
1. OOS通过率（选中参数中Sharpe>0占比）> 60%
2. OOS粗糙度恶化倍数 < 5x（参数表面不能在OOS崩成锯齿）
3. IS→OOS Sharpe中位数保持率 > 40%

## 输出格式（严格遵循）
[OOS判断] <通过 | 失败>
[分析] <通过/失败的原因，1-3句话>
[教训] <一句话总结：这个方向未来该怎么调整，或为什么这个方向可行>
"""

SELECTED_PATTERN = re.compile(r"\[SELECTED\]\s*\[(.+)\]", re.DOTALL)
OOS_VERDICT_PATTERN = re.compile(r"\[OOS判断\]\s*(.+)")
OOS_ANALYSIS_PATTERN = re.compile(r"\[分析\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
OOS_LEARNING_PATTERN = re.compile(r"\[教训\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)


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


def build_oos_judge_messages(oos_report_text: str, factor_info: str) -> list:
    user_msg = (f"因子信息：{factor_info}\n\n"
                f"以下是OOS测试结果：\n{oos_report_text}\n\n"
                f"请给出你的判断。")
    return [
        {"role": "system", "content": OOS_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_oos_verdict(response: str) -> dict:
    verdict_m = OOS_VERDICT_PATTERN.search(response)
    analysis_m = OOS_ANALYSIS_PATTERN.search(response)
    learning_m = OOS_LEARNING_PATTERN.search(response)

    verdict_text = verdict_m.group(1).strip() if verdict_m else ""
    return {
        "passed": "通过" in verdict_text and "失败" not in verdict_text,
        "verdict": verdict_text,
        "analysis": analysis_m.group(1).strip() if analysis_m else "",
        "learning": learning_m.group(1).strip() if learning_m else "",
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


def ask_oos_judge(chat_fn, oos_report_text: str, factor_info: str,
                  model: str = None) -> dict:
    messages = build_oos_judge_messages(oos_report_text, factor_info)
    result = chat_fn(messages, model=model, max_tokens=2048, temperature=0.3)
    response = result["answer"]
    verdict = parse_oos_verdict(response)
    verdict["raw_response"] = response
    verdict["usage"] = result["usage"]
    return verdict
