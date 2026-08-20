"""Factories for SciPy-compatible portfolio constraints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def full_investment_constraint() -> dict:
    """Return the equality constraint ``sum(weights) == 1``."""
    return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}


def create_style_constraints(
    asset_names: Sequence[str],
    style_assets: Iterable[str],
    lower: float,
    upper: float,
) -> tuple[dict, ...]:
    """Constrain a named asset bucket to a total weight interval.

    The returned tuple includes the full-investment equality constraint.
    """
    assets = list(asset_names)
    selected = list(style_assets)
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("asset_names must be a non-empty sequence of unique names")
    if not selected:
        raise ValueError("style_assets must contain at least one asset name")

    missing = [name for name in selected if name not in assets]
    if missing:
        raise ValueError(f"unknown style assets: {missing}")
    if len(set(selected)) != len(selected):
        raise ValueError("style_assets must not contain duplicates")

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("style bounds must be finite and satisfy lower <= upper")

    indices = np.array([assets.index(name) for name in selected], dtype=np.int64)
    return (
        full_investment_constraint(),
        {
            "type": "ineq",
            "fun": lambda weights, idx=indices, limit=lower: np.sum(weights[idx]) - limit,
        },
        {
            "type": "ineq",
            "fun": lambda weights, idx=indices, limit=upper: limit - np.sum(weights[idx]),
        },
    )
