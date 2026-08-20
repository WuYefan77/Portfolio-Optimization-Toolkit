"""Validated SLSQP routines for constrained static portfolio allocation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .constraints import full_investment_constraint
from .utils import nearest_psd, portfolio_performance, portfolio_variance


class OptimizationError(RuntimeError):
    """Raised when SLSQP fails or returns a materially infeasible solution."""


def _coerce_constraints(constraints) -> tuple[dict, ...]:
    if constraints is None:
        values = (full_investment_constraint(),)
    elif isinstance(constraints, dict):
        values = (constraints,)
    elif isinstance(constraints, Callable):
        values = tuple(constraints())
    else:
        values = tuple(constraints)

    for constraint in values:
        if (
            not isinstance(constraint, dict)
            or constraint.get("type") not in {"eq", "ineq"}
            or not callable(constraint.get("fun"))
        ):
            raise ValueError("constraints must contain SciPy equality or inequality mappings")
    return values


class StaticOptimizer:
    """Static mean--variance optimiser with explicit validation."""

    def __init__(self, expected_returns, covariance, *, repair_covariance: bool = True):
        expected_returns = np.asarray(expected_returns, dtype=np.float64).copy()
        covariance = np.asarray(covariance, dtype=np.float64).copy()
        if (
            expected_returns.ndim != 1
            or len(expected_returns) < 1
            or not np.all(np.isfinite(expected_returns))
        ):
            raise ValueError("expected_returns must be a non-empty finite vector")
        if (
            covariance.shape != (len(expected_returns), len(expected_returns))
            or not np.all(np.isfinite(covariance))
        ):
            raise ValueError("covariance must be a finite square matrix matching expected_returns")

        symmetric = 0.5 * (covariance + covariance.T)
        if repair_covariance:
            repaired = nearest_psd(symmetric)
        else:
            if np.min(np.linalg.eigvalsh(symmetric)) < -1e-10:
                raise ValueError("covariance must be positive semi-definite")
            repaired = symmetric

        self.expected_returns = expected_returns
        self.covariance = repaired
        self.covariance_adjustment = float(np.linalg.norm(repaired - covariance, ord="fro"))
        self.n_assets = len(expected_returns)

        # Compatibility with the names used by the original notebooks.
        self.mu = self.expected_returns
        self.cov = self.covariance

    def _prepare_bounds(self, bounds) -> tuple[tuple[float | None, float | None], ...]:
        if bounds is None:
            return tuple((0.0, 1.0) for _ in range(self.n_assets))
        bounds = tuple(bounds)
        if len(bounds) != self.n_assets:
            raise ValueError("bounds must contain one (lower, upper) pair per asset")
        checked = []
        for pair in bounds:
            if len(pair) != 2:
                raise ValueError("each bound must contain lower and upper values")
            lower, upper = pair
            lower = None if lower is None else float(lower)
            upper = None if upper is None else float(upper)
            if lower is not None and not np.isfinite(lower):
                raise ValueError("lower bounds must be finite or None")
            if upper is not None and not np.isfinite(upper):
                raise ValueError("upper bounds must be finite or None")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("lower bounds must not exceed upper bounds")
            checked.append((lower, upper))
        return tuple(checked)

    def _prepare_initial_weights(self, initial_weights) -> np.ndarray:
        if initial_weights is None:
            return np.full(self.n_assets, 1.0 / self.n_assets)
        weights = np.asarray(initial_weights, dtype=np.float64)
        if weights.shape != (self.n_assets,) or not np.all(np.isfinite(weights)):
            raise ValueError("initial_weights must be a finite vector with one value per asset")
        return weights

    @staticmethod
    def _maximum_violation(weights, bounds, constraints) -> float:
        violations = []
        for value, (lower, upper) in zip(weights, bounds):
            if lower is not None:
                violations.append(max(lower - value, 0.0))
            if upper is not None:
                violations.append(max(value - upper, 0.0))
        for constraint in constraints:
            values = np.atleast_1d(constraint["fun"](weights)).astype(float)
            if not np.all(np.isfinite(values)):
                return float("inf")
            if constraint["type"] == "eq":
                violations.extend(np.abs(values))
            else:
                violations.extend(np.maximum(-values, 0.0))
        return float(np.max(violations, initial=0.0))

    def _solve(
        self,
        objective,
        *,
        bounds=None,
        constraints=None,
        initial_weights=None,
        tolerance: float = 1e-7,
    ) -> OptimizeResult:
        bounds = self._prepare_bounds(bounds)
        constraints = _coerce_constraints(constraints)
        initial_weights = self._prepare_initial_weights(initial_weights)
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 1_000},
        )
        violation = self._maximum_violation(result.x, bounds, constraints)
        result.max_constraint_violation = violation
        if not result.success:
            raise OptimizationError(f"SLSQP failed: {result.message}")
        if violation > max(10.0 * tolerance, 1e-6):
            raise OptimizationError(
                f"SLSQP returned an infeasible solution (maximum violation {violation:.3e})"
            )
        return result

    def maximize_sharpe(
        self,
        risk_free_rate: float = 0.0,
        *,
        bounds=None,
        constraints=None,
        initial_weights=None,
        tolerance: float = 1e-7,
        rf: float | None = None,
        w0=None,
    ) -> OptimizeResult:
        """Maximise the ex-ante Sharpe ratio under the supplied constraints."""
        if rf is not None:
            risk_free_rate = rf
        if w0 is not None:
            if initial_weights is not None:
                raise ValueError("provide only one of initial_weights or w0")
            initial_weights = w0
        risk_free_rate = float(risk_free_rate)
        if not np.isfinite(risk_free_rate):
            raise ValueError("risk_free_rate must be finite")

        def negative_sharpe(weights):
            expected_return, volatility = portfolio_performance(
                weights,
                self.expected_returns,
                self.covariance,
            )
            return -(expected_return - risk_free_rate) / max(volatility, 1e-12)

        return self._solve(
            negative_sharpe,
            bounds=bounds,
            constraints=constraints,
            initial_weights=initial_weights,
            tolerance=tolerance,
        )

    def minimize_variance(
        self,
        *,
        bounds=None,
        constraints=None,
        initial_weights=None,
        tolerance: float = 1e-7,
        w0=None,
    ) -> OptimizeResult:
        """Minimise portfolio variance under the supplied constraints."""
        if w0 is not None:
            if initial_weights is not None:
                raise ValueError("provide only one of initial_weights or w0")
            initial_weights = w0
        return self._solve(
            lambda weights: portfolio_variance(weights, self.covariance),
            bounds=bounds,
            constraints=constraints,
            initial_weights=initial_weights,
            tolerance=tolerance,
        )

    def target_return(
        self,
        target: float,
        *,
        bounds=None,
        constraints=None,
        initial_weights=None,
        tolerance: float = 1e-7,
        w0=None,
    ) -> OptimizeResult:
        """Minimise variance subject to an expected-return lower bound."""
        target = float(target)
        if not np.isfinite(target):
            raise ValueError("target must be finite")
        if w0 is not None:
            if initial_weights is not None:
                raise ValueError("provide only one of initial_weights or w0")
            initial_weights = w0
        values = list(_coerce_constraints(constraints))
        values.append({
            "type": "ineq",
            "fun": lambda weights, minimum=target: weights @ self.expected_returns - minimum,
        })
        return self._solve(
            lambda weights: portfolio_variance(weights, self.covariance),
            bounds=bounds,
            constraints=values,
            initial_weights=initial_weights,
            tolerance=tolerance,
        )
