import unittest

import numpy as np

from portfolio_optimizer import nearest_psd, portfolio_performance, portfolio_variance


class UtilityTests(unittest.TestCase):
    def test_nearest_psd_repairs_negative_eigenvalue(self):
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        repaired = nearest_psd(matrix)
        np.testing.assert_allclose(repaired, repaired.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(repaired).min(), -1e-12)

    def test_portfolio_performance_matches_direct_calculation(self):
        weights = np.array([0.4, 0.6])
        expected_returns = np.array([0.1, 0.05])
        covariance = np.diag([0.04, 0.01])
        expected_return, volatility = portfolio_performance(
            weights,
            expected_returns,
            covariance,
        )
        self.assertAlmostEqual(expected_return, 0.07)
        self.assertAlmostEqual(volatility**2, portfolio_variance(weights, covariance))


if __name__ == "__main__":
    unittest.main()
