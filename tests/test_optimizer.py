import unittest

import numpy as np

from portfolio_optimizer import (
    OptimizationError,
    StaticOptimizer,
    create_style_constraints,
)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.expected_returns = np.array([0.10, 0.08, 0.04])
        self.covariance = np.array([
            [0.0225, 0.0135, 0.0015],
            [0.0135, 0.0144, 0.0024],
            [0.0015, 0.0024, 0.0025],
        ])
        self.optimizer = StaticOptimizer(self.expected_returns, self.covariance)

    def test_quick_start_constraints_are_satisfied(self):
        bounds = tuple((0.0, 0.60) for _ in range(3))
        constraints = create_style_constraints(
            ["Equity A", "Equity B", "Bonds"],
            ["Equity A", "Equity B"],
            lower=0.4,
            upper=0.8,
        )
        result = self.optimizer.maximize_sharpe(
            risk_free_rate=0.02,
            bounds=bounds,
            constraints=constraints,
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.max_constraint_violation, 1e-6)
        self.assertAlmostEqual(np.sum(result.x), 1.0, places=6)
        self.assertGreaterEqual(np.sum(result.x[:2]), 0.4 - 1e-6)
        self.assertLessEqual(np.sum(result.x[:2]), 0.8 + 1e-6)

    def test_single_constraint_mapping_is_accepted(self):
        constraint = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
        result = self.optimizer.minimize_variance(constraints=constraint)
        self.assertAlmostEqual(np.sum(result.x), 1.0, places=6)

    def test_invalid_covariance_can_be_rejected_instead_of_repaired(self):
        invalid = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            StaticOptimizer([0.1, 0.2], invalid, repair_covariance=False)

    def test_infeasible_problem_raises_optimization_error(self):
        infeasible = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 2.0},)
        with self.assertRaises(OptimizationError):
            self.optimizer.minimize_variance(
                bounds=tuple((0.0, 0.2) for _ in range(3)),
                constraints=infeasible,
            )


if __name__ == "__main__":
    unittest.main()
