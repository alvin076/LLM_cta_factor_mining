"""Factor Generator — generates factor formulas with direction context.

System prompt includes the function library (auto-generated from
backtest/factor_lib.py) and hard complexity red lines.
"""

import re
import inspect

FACTOR_SYSTEM_TEMPLATE = r"""
你是一位拥有二十余年实战经验的资深CTA量化研究员。
你的交易生涯横跨全球所有主要市场——股票、期货、期权、加密货币、外汇、大宗商品。
你历经多轮牛熊周期、流动性危机与市场体制切换，管理过数十亿规模资金。
现在你需要为CTA策略挖掘Alpha因子。

## 数据
1小时频OHLCV数据，DataFrame字段：Open, High, Low, Close, Volume。

## 核心约束
1. 经济直觉：每个因子必须有清晰的经济学或行为金融学逻辑。
   不允许"这个模式在回测中表现好"这类纯粹数据挖掘的理由。
   必须能向非量化的人解释清楚：这个因子在捕捉什么市场行为。

2. 禁止垃圾嵌套：严禁出现类似以下的多层套娃：
   ts_mean(ts_mean(ts_mean(close/open, 10), 90), 100)
   这种无限嵌套是过拟合垃圾。最多一层平滑函数，如果确实需要，
   必须明确解释为什么是这个窗口参数。

3. 因子类型偏好：
   - 动量/反转类：趋势强度、极端收益反转、路径依赖效应
   - 波动率类：波动率聚集、杠杆效应、风险溢价
   - 成交量/流动性类：量价关系、流动性压力、供需失衡
   - 行为金融类：锚定效应、处置效应、羊群效应
   - 宏观/跨资产类：相关性破裂、波动率传导

4. 复杂度红线（生成前自我审查，违反任何一条必须重写）：
   - 因子表达式总字符数严禁超过 250 个。
   - 公式中引用的原始字段（Open/High/Low/Close/Volume）不得超过 6 个。
   - 可调窗口参数（如 ts_mean 中的 10）数量不得超过总 Token 数的一半。

## 可用函数库（白名单，只能使用以下名字）

你只能使用以下内容，白名单之外的任何名字都会直接报 NameError：
- 数据字段（必须带 df 前缀）：df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
- 以下因子函数：

{FUNCTION_LIB}

- numpy 函数（仅限这些）：np.log, np.sign, np.sqrt, np.exp, np.abs, np.clip, np.maximum, np.minimum, np.where
- 基本运算：+ - * / ** ( ) < > == 及数字常量

严禁使用：
- 裸字段名（如 Close、high，必须写 df['Close']）
- df 之外的任何未定义变量（如 ret、close）
- pandas 方法链（如 .rolling()、.pct_change()）、for 循环、lambda

调用规则：
- 窗口参数 window 建议 3~240（1小时线的3小时~10天）。

## 输出格式（严格遵循）
只输出以下两个区块，不得有任何额外内容。

[公式]
factor = <表达式，只能使用白名单内的函数和字段>

[经济逻辑]
1. 捕捉什么市场现象：
2. 为什么这个现象会持续存在（不会被套利消除）：
3. 适合/不适合的市场环境：

## 输出示例（严格模仿此格式，包括标签、空行和标点）

示例1（动量类）：
[公式]
factor = ts_mean(df['Close'], 20) / ts_std(df['Close'], 20)

[经济逻辑]
1. 捕捉什么市场现象：趋势强度与波动幅度的比值，衡量单位波动下的趋势稳定性。
2. 为什么这个现象会持续存在：信息扩散缓慢，趋势形成后正反馈资金接力。
3. 适合/不适合的市场环境：适合单边趋势市；不适合窄幅震荡市。

示例2（量价类）：
[公式]
factor = corr(df['Close'], df['Volume'], 30) * ts_rank(df['Volume'], 60)

[经济逻辑]
1. 捕捉什么市场现象：量价背离程度与成交活跃度的联合信号。
2. 为什么这个现象会持续存在：筹码结构与资金行为惯性。
3. 适合/不适合的市场环境：适合机构参与度高的时段；不适合流动性枯竭期。
"""


def build_function_lib_section() -> str:
    """
    Auto-generate the function library section from backtest/factor_lib.py.
    Single source of truth: add/remove functions there, prompt updates itself.
    """
    from backtest.factor_lib import FACTOR_FUNCTIONS

    lines = []
    for name, fn in FACTOR_FUNCTIONS.items():
        params = ", ".join(inspect.signature(fn).parameters)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  - {name}({params}): {doc}")
    return "\n".join(lines)


# Materialize the final system prompt with the live function library
FACTOR_SYSTEM = FACTOR_SYSTEM_TEMPLATE.replace("{FUNCTION_LIB}", build_function_lib_section())


def build_factor_messages(direction: dict, trajectory_context: str = "") -> list:
    user_msg = (f"研究方向：{direction['name']}。\n"
                f"核心逻辑：{direction['logic']}。\n"
                f"该方向历史尝试：\n{trajectory_context}\n"
                f"请生成一个新的Alpha因子。必须吸取历史教训，避免重复历史公式。")
    return [
        {"role": "system", "content": FACTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def build_retry_messages(direction: dict, violation_reason: str,
                         trajectory_context: str = "") -> list:
    user_msg = (f"研究方向：{direction['name']}。核心逻辑：{direction['logic']}。\n"
                f"该方向历史尝试：\n{trajectory_context}\n"
                f"你上次生成的因子被拒绝：{violation_reason}\n"
                f"请重新输出，严格模仿系统提示中的示例格式，避免重复历史公式。")
    return [
        {"role": "system", "content": FACTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def extract_factor(response: str) -> dict:
    """Parse [formula] [economic logic] from LLM output.

    Tolerant to: same-line tags, markdown fences, missing [公式] tag.
    """
    clean = response.replace("```", "")

    # 1) formula: first "factor = ..." occurrence anywhere, first line only
    formula = ""
    code_match = re.search(r"factor\s*=\s*(.+)", clean)
    if code_match:
        formula = code_match.group(1).strip()
        formula = formula.split("\n")[0].strip()
        formula = formula.strip("`").strip()

    # 2) logic: everything after [经济逻辑] tag
    logic = ""
    logic_match = re.search(r"\[经济逻辑\]\s*(.*)", clean, re.DOTALL)
    if logic_match:
        logic = logic_match.group(1).strip()

    return {"formula": formula, "logic": logic}


def smoke_test(formula: str) -> tuple:
    """
    Eval the formula on a tiny dummy DataFrame to catch
    NameError/SyntaxError before the real grid search.

    Returns (True, "") or (False, error_message).
    """
    import pandas as pd
    from backtest.safety import safe_eval

    dummy = pd.DataFrame({
        "Open": [100.0] * 30,
        "High": [101.0] * 30,
        "Low": [99.0] * 30,
        "Close": [100.5] * 30,
        "Volume": [1000.0] * 30,
    })
    try:
        result = safe_eval(formula, dummy)
        return True, ""
    except ValueError as e:
        return False, str(e)


def generate_factor(chat_fn, direction: dict, trajectory_context: str = "",
                    model: str = None, max_retries: int = 2) -> dict:
    """
    Generate a factor for a research direction, retrying on
    complexity violations.

    Args:
        chat_fn: LLM chat function
        direction: dict with name, logic
        trajectory_context: formatted history of this direction's attempts
        model: model override
        max_retries: max retries on violation/parse failure

    Returns:
        dict with keys: category (=direction name), formula, logic, direction,
                        retries, valid, violation_reason, tokens
    """
    messages = build_factor_messages(direction, trajectory_context)

    from .validators import validate_formula

    for attempt in range(max_retries + 1):
        result = chat_fn(messages, model=model, max_tokens=8192, temperature=0.8)
        response = result["answer"]
        extracted = extract_factor(response)

        if not extracted["formula"]:
            if attempt < max_retries:
                messages = build_retry_messages(
                    direction,
                    "输出格式错误（缺少 factor = 表达式），"
                    "请严格模仿系统提示中的输出示例格式",
                    trajectory_context,
                )
                continue
            return {
                "category": direction["name"], "formula": "", "logic": "",
                "direction": direction,
                "retries": attempt, "valid": False,
                "violation_reason": "公式解析失败",
                "raw_response": response,
                "tokens": result["usage"],
            }

        ok, reason = validate_formula(extracted["formula"])
        if not ok:
            # violated - retry with the reason fed back
            messages = build_retry_messages(direction, reason, trajectory_context)
            continue

        # smoke test: catch NameError / SyntaxError early
        ok_smoke, smoke_reason = smoke_test(extracted["formula"])
        if not ok_smoke:
            messages = build_retry_messages(direction, smoke_reason, trajectory_context)
            continue

        extracted["category"] = direction["name"]
        extracted["direction"] = direction
        extracted["retries"] = attempt
        extracted["valid"] = True
        extracted["violation_reason"] = ""
        extracted["tokens"] = result["usage"]
        return extracted

    return {
        "category": direction["name"],
        "formula": extracted.get("formula", ""),
        "logic": extracted.get("logic", ""),
        "direction": direction,
        "retries": max_retries,
        "valid": False,
        "violation_reason": reason,
        "raw_response": result.get("answer", ""),
        "tokens": result.get("usage", {}),
    }
