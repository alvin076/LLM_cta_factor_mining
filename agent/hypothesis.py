"""Hypothesis Agent — generates research directions.

Instead of picking one direction, the agent generates 5-8 independent,
complementary research directions informed by the research trajectory.
"""

import re

HYPOTHESIS_SYSTEM = r"""
你是一位拥有二十余年实战经验的资深CTA量化研究员。
你的任务是制定因子挖掘的"研究方向清单"，而不是直接写因子公式。

## 策略边界（硬约束）
1. 策略是单标的CTA时序策略：基于单一标的自身的历史OHLCV序列
   生成开多/开空/空仓信号，不做多资产组合。
2. 严禁横截面方向：如"资产间相对收益排序"、"多资产轮动"、
   "截面动量"、"横截面相对强弱"等。
3. 所有方向必须能只用单标的自身的 Open/High/Low/Close/Volume 表达。
4. 方向覆盖类型（全部时序视角）：趋势动量、均值回复、
   波动率状态、量价关系、行为金融锚点、微观结构买卖压力。

## 要求
1. 一次性生成 5~8 个在金融逻辑上相互独立、彼此互补的研究方向。
2. 覆盖范围必须包括但不限于：动量/反转、波动率聚集、量价背离、
   行为金融（锚定效应、处置效应）、微观结构（买卖压力）。
3. 如果提供了研究轨迹，必须基于轨迹中的教训调整方向：
   - 已失败的方向要换角度或放弃
   - 已通过的方向可以深化
   - 未探索的方向优先
4. 每个方向只写名称和 1 句核心逻辑，不要展开代码。

## 输出格式（严格遵循）
[方向1] <名称> | <核心逻辑一句>
[方向2] <名称> | <核心逻辑一句>
...
[方向N] <名称> | <核心逻辑一句>
"""

DIRECTION_PATTERN = re.compile(r"\[方向\d+\]\s*(.+?)\s*\|\s*(.+)")


def build_hypothesis_messages(trajectory_summary: str,
                              symbol: str = "") -> list:
    header = (f"当前研究标的：{symbol}\n\n" if symbol else "")
    user_msg = (f"{header}以下是已有的研究轨迹（仅限当前标的）：\n"
                f"{trajectory_summary}\n\n请生成下一批研究方向。")
    return [
        {"role": "system", "content": HYPOTHESIS_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_directions(response: str) -> list:
    """Parse [方向N] 名称 | 逻辑 lines into a list of dicts."""
    directions = []
    for line in response.splitlines():
        m = DIRECTION_PATTERN.match(line.strip())
        if m:
            directions.append({
                "name": m.group(1).strip(),
                "logic": m.group(2).strip(),
            })
    return directions


def generate_directions(chat_fn, trajectory_summary: str,
                        model: str = None, max_tokens: int = 8196,
                        symbol: str = "") -> dict:
    """Call the LLM and return parsed directions + raw response."""
    messages = build_hypothesis_messages(trajectory_summary, symbol=symbol)
    result = chat_fn(messages, model=model, max_tokens=max_tokens, temperature=0.7)
    response = result["answer"]
    directions = parse_directions(response)
    return {
        "directions": directions,
        "raw_response": response,
        "usage": result["usage"],
    }
