"""Safe evaluation of LLM-generated factor formulas."""

import re
import pandas as pd
import numpy as np

from .factor_lib import FACTOR_FUNCTIONS

FORBIDDEN = [
    '__builtins__', '__import__', '__class__', '__bases__', '__subclasses__',
    '__globals__', '__code__', '__reduce__', '__reduce_ex__',
    'import ', 'exec', 'eval', 'compile',
    'open(', 'os.', 'sys.', 'subprocess', 'shutil',
    'requests', 'urllib', 'socket',
    ';', 'globals()', 'locals()',
    'getattr', 'setattr', 'delattr', 'hasattr',
]


def check_safety(formula: str) -> None:
    """Raise ValueError if formula contains forbidden patterns."""
    formula_lower = formula.lower()
    for pattern in FORBIDDEN:
        if pattern.lower() in formula_lower:
            raise ValueError(f"Forbidden pattern detected: {pattern}")


def safe_eval(formula: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely evaluate a pandas factor formula on a DataFrame.

    Args:
        formula: A pandas expression string, e.g. "df['Close'].pct_change(20)"
        df: OHLCV DataFrame with columns Open, High, Low, Close, Volume

    Returns:
        pd.Series of factor values aligned to index
    """
    check_safety(formula)

    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        **FACTOR_FUNCTIONS,
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "True": True,
            "False": False,
            "None": None,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "round": round,
            "pow": pow,
            "sqrt": np.sqrt,
            "log": np.log,
            "exp": np.exp,
            "sign": np.sign,
            "clip": np.clip,
        },
    }

    try:
        result = eval(formula, namespace)
    except NameError as e:
        raise ValueError(f"Formula references undefined variable: {e}")
    except SyntaxError as e:
        raise ValueError(f"Formula has syntax error: {e}")
    except TypeError as e:
        raise ValueError(f"Formula has type error: {e}")
    if isinstance(result, pd.Series):
        return result
    elif isinstance(result, (int, float, np.number)):
        return pd.Series(result, index=df.index)
    else:
        raise ValueError(f"Formula returned {type(result).__name__} instead of Series. 公式可能不完整（如包含省略号）")
