"""Numerical utilities for static portfolio analysis."""

from __future__ import annotations

import numpy as np


def _as_finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def nearest_psd(matrix, *, eigenvalue_floor: float = 0.0) -> np.ndarray:
    """Project a square matrix onto the symmetric PSD cone by eigenvalue clipping.

    This projection restores numerical validity; it does not estimate the
    covariance structure that generated an invalid input matrix.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("matrix must be a non-empty finite square array")
    eigenvalue_floor = float(eigenvalue_floor)
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor < 0.0:
        raise ValueError("eigenvalue_floor must be finite and non-negative")

    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def portfolio_variance(weights, covariance) -> float:
    """Return portfolio variance ``weights.T @ covariance @ weights``."""
    weights = _as_finite_vector(weights, "weights")
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(weights), len(weights)) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite square matrix matching weights")
    value = float(weights @ covariance @ weights)
    if value < -1e-10:
        raise ValueError("covariance produces a materially negative portfolio variance")
    return max(value, 0.0)


def portfolio_performance(weights, expected_returns, covariance) -> tuple[float, float]:
    """Return expected return and volatility for a set of weights."""
    weights = _as_finite_vector(weights, "weights")
    expected_returns = _as_finite_vector(expected_returns, "expected_returns")
    if weights.shape != expected_returns.shape:
        raise ValueError("weights and expected_returns must have equal length")
    expected_return = float(weights @ expected_returns)
    volatility = float(np.sqrt(portfolio_variance(weights, covariance)))
    return expected_return, volatility


def exponential_utility(
    expected_return: float,
    volatility: float,
    *,
    risk_aversion: float = 1.0,
) -> float:
    """Return CARA expected utility under a normally distributed return."""
    expected_return = float(expected_return)
    volatility = float(volatility)
    risk_aversion = float(risk_aversion)
    if not np.isfinite(expected_return):
        raise ValueError("expected_return must be finite")
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError("volatility must be finite and non-negative")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be finite and positive")
    certainty_equivalent = expected_return - 0.5 * risk_aversion * volatility**2
    return float(-np.exp(-risk_aversion * certainty_equivalent))


# Backwards-compatible aliases for the original notebook API.
port_perf = portfolio_performance
exp_utility = exponential_utility
