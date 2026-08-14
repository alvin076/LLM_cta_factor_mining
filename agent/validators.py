"""Hard complexity constraints for LLM-generated factor formulas.

Enforces QuantaAlpha-style overfitting red lines:
1. Expression length <= 250 chars
2. Raw features referenced <= 6
3. Free parameter ratio < 50%
"""

import re

MAX_EXPR_LENGTH = 250
MAX_FEATURES = 6
MAX_PARAM_RATIO = 0.5

FEATURE_PATTERN = re.compile(r"df\['(\w+)'\]")
NUMBER_PATTERN = re.compile(r"\d+\.?\d*")
TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|df|'\w+'|\d+\.?\d*")


def validate_formula(formula: str) -> tuple:
    """
    Validate a factor formula against complexity red lines.

    Returns:
        (True, "") if all constraints pass
        (False, reason) if any constraint violated
    """
    # Constraint 0: no ellipsis placeholder
    if "..." in formula:
        return False, "公式包含省略号(...)，禁止占位符，请写完整表达式"

    # Constraint 1: expression length
    if len(formula) > MAX_EXPR_LENGTH:
        return False, f"表达式{len(formula)}字符，超过{MAX_EXPR_LENGTH}上限"

    # Constraint 2: raw features
    features = set(FEATURE_PATTERN.findall(formula))
    if len(features) > MAX_FEATURES:
        return False, f"引用{len(features)}个原始字段（{sorted(features)}），超过{MAX_FEATURES}个上限"

    # Constraint 3: free parameter ratio
    tokens = TOKEN_PATTERN.findall(formula)
    if tokens:
        param_count = sum(1 for t in tokens if re.match(r"^\d", t))
        ratio = param_count / len(tokens)
        if ratio >= MAX_PARAM_RATIO:
            return False, f"参数占比{ratio:.0%}，超过{MAX_PARAM_RATIO:.0%}上限"

    return True, ""
