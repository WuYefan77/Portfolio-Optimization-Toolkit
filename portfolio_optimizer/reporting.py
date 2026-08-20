"""Tables and plots for portfolio optimisation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .utils import portfolio_performance


class PortfolioReporter:
    """Create performance tables and allocation charts."""

    def __init__(self, asset_names, risk_free_rate: float = 0.0, *, rf_rate=None):
        self.assets = list(asset_names)
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("asset_names must be a non-empty sequence of unique names")
        if rf_rate is not None:
            risk_free_rate = rf_rate
        self.risk_free_rate = float(risk_free_rate)
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.rf = self.risk_free_rate

    def calculate_performance(self, weights, expected_returns, covariance):
        expected_return, volatility = portfolio_performance(
            weights,
            expected_returns,
            covariance,
        )
        sharpe = (expected_return - self.risk_free_rate) / max(volatility, 1e-12)
        return expected_return, volatility, float(sharpe)

    calc_performance = calculate_performance

    def create_summary_table(self, portfolios, expected_returns, covariance) -> pd.DataFrame:
        rows = []
        for name, weights in portfolios.items():
            expected_return, volatility, sharpe = self.calculate_performance(
                weights,
                expected_returns,
                covariance,
            )
            rows.append({
                "Portfolio": name,
                "Return": expected_return,
                "Volatility": volatility,
                "Sharpe": sharpe,
            })
        if not rows:
            raise ValueError("portfolios must contain at least one allocation")
        return pd.DataFrame(rows).set_index("Portfolio")

    @staticmethod
    def format_performance_table(table: pd.DataFrame) -> pd.DataFrame:
        required = {"Return", "Volatility", "Sharpe"}
        if not required.issubset(table.columns):
            raise ValueError(f"table must contain columns {sorted(required)}")
        formatted = table.copy()
        formatted["Return"] = formatted["Return"].astype(float).map(lambda value: f"{value:.2%}")
        formatted["Volatility"] = formatted["Volatility"].astype(float).map(
            lambda value: f"{value:.2%}"
        )
        formatted["Sharpe"] = formatted["Sharpe"].astype(float).map(lambda value: f"{value:.4f}")
        return formatted

    def plot_weights_grouped(self, weights: pd.DataFrame, title="Portfolio weights"):
        if list(weights.index) != self.assets:
            raise ValueError("weights index must match asset_names in the same order")
        if weights.shape[1] < 1:
            raise ValueError("weights must contain at least one portfolio column")

        figure, axis = plt.subplots(figsize=(11, 6))
        x_positions = np.arange(len(self.assets))
        width = 0.85 / weights.shape[1]
        for column_index, column in enumerate(weights.columns):
            axis.bar(
                x_positions + column_index * width,
                weights[column].to_numpy(dtype=float),
                width,
                label=column,
            )
        axis.set_xticks(x_positions + width * (weights.shape[1] - 1) / 2)
        axis.set_xticklabels(self.assets, rotation=30, ha="right")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylabel("Weight")
        axis.set_title(title)
        axis.legend()
        figure.tight_layout()
        return figure, axis

    def plot_allocation_pie(
        self,
        weights: pd.Series,
        title="Portfolio allocation",
        *,
        minimum_weight: float = 1e-4,
        save_path=None,
    ):
        if not isinstance(weights, pd.Series):
            raise ValueError("weights must be a pandas Series indexed by asset name")
        minimum_weight = float(minimum_weight)
        if not np.isfinite(minimum_weight) or minimum_weight < 0.0:
            raise ValueError("minimum_weight must be finite and non-negative")
        selected = weights[weights > minimum_weight].sort_values(ascending=False)
        if selected.empty:
            raise ValueError("no weights exceed minimum_weight")

        colour_map = plt.get_cmap("tab20")
        colours = [colour_map(index % colour_map.N) for index in range(len(selected))]
        figure, axis = plt.subplots(figsize=(9, 7))
        wedges, _, _ = axis.pie(
            selected,
            autopct="%1.1f%%",
            startangle=90,
            colors=colours,
            pctdistance=0.8,
        )
        axis.legend(
            wedges,
            selected.index,
            title="Asset classes",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        axis.set_title(title)
        axis.axis("equal")
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        return figure, axis
