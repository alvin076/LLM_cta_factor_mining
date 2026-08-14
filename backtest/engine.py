"""FactorBacktester — main orchestrator for factor evaluation pipeline."""

import numpy as np
import pandas as pd

from .safety import safe_eval
from .grid_search import grid_search
from .report import is_report, oos_report


OOS_ROWS = 8760  # 1 year of 1-hour data


class FactorBacktester:
    """
    Evaluate LLM-generated alpha factors with IS/OOS grid search.

    IS = all data except last 8760 rows (1 year)
    OOS = last 8760 rows
    """

    def __init__(self, data_path: str, commission_bps: float = 6.0):
        self.commission_bps = commission_bps
        self.symbol = data_path.split("/")[-1].split("\\")[-1].replace(".csv", "")
        self.df = self._load(data_path)
        self.df_is = self.df.iloc[:-OOS_ROWS].copy()
        self.df_oos = self.df.iloc[-OOS_ROWS:].copy()

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path}")
        return df

    @property
    def is_rows(self) -> int:
        return len(self.df_is)

    @property
    def oos_rows(self) -> int:
        return len(self.df_oos)

    @property
    def data_range(self) -> tuple:
        return (str(self.df.index[0])[:10], str(self.df.index[-1])[:10])

    @property
    def is_range(self) -> tuple:
        if len(self.df_is) == 0:
            return (None, None)
        return (str(self.df_is.index[0])[:10], str(self.df_is.index[-1])[:10])

    @property
    def oos_range(self) -> tuple:
        if len(self.df_oos) == 0:
            return (None, None)
        return (str(self.df_oos.index[0])[:10], str(self.df_oos.index[-1])[:10])

    def run_is(self, formula: str) -> dict:
        """Run IS grid search. Returns raw result dict from grid_search()."""
        return grid_search(self.df_is, formula, commission_bps=self.commission_bps)

    def run_oos(self, formula: str) -> dict:
        """Run OOS grid search. Returns raw result dict from grid_search()."""
        return grid_search(self.df_oos, formula, commission_bps=self.commission_bps)

    def evaluate(
        self,
        formula: str,
        category: str = "",
    ) -> dict:
        """
        Full evaluation pipeline: IS grid search.

        Returns:
            dict with keys: is_result, is_report_text, symbol, formula, category
        """
        is_result = self.run_is(formula)

        report_text = is_report(
            is_result,
            formula,
            category,
            symbol=self.symbol,
        )

        return {
            "symbol": self.symbol,
            "formula": formula,
            "category": category,
            "is_result": is_result,
            "is_report": report_text,
        }

    def evaluate_oos(
        self,
        formula: str,
        selected_params: list,
        is_result: dict,
    ) -> dict:
        """
        Run OOS evaluation with LLM-selected parameters.

        Args:
            formula: factor formula string
            selected_params: list of (window, threshold) tuples chosen by LLM
            is_result: IS grid search result dict (for comparison)

        Returns:
            dict with keys: oos_result, oos_report_text
        """
        oos_result = self.run_oos(formula)

        report_text = oos_report(
            is_result,
            oos_result,
            selected_params,
            symbol=self.symbol,
        )

        return {
            "oos_result": oos_result,
            "oos_report": report_text,
            "selected_params": selected_params,
        }
