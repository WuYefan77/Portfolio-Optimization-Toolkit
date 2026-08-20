import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_optimizer import PortfolioReporter


class ReportingTests(unittest.TestCase):
    def test_summary_and_plots_are_created_without_global_show(self):
        reporter = PortfolioReporter(["Equity", "Bonds"], risk_free_rate=0.02)
        expected_returns = np.array([0.10, 0.04])
        covariance = np.diag([0.04, 0.01])
        portfolios = {"Balanced": np.array([0.5, 0.5])}
        summary = reporter.create_summary_table(portfolios, expected_returns, covariance)
        self.assertEqual(list(summary.index), ["Balanced"])

        weights = pd.DataFrame(portfolios, index=["Equity", "Bonds"])
        figure, _ = reporter.plot_weights_grouped(weights)
        pie, _ = reporter.plot_allocation_pie(weights["Balanced"])
        self.assertIsNotNone(figure)
        self.assertIsNotNone(pie)
        plt.close(figure)
        plt.close(pie)


if __name__ == "__main__":
    unittest.main()
