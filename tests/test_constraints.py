import unittest

import numpy as np

from portfolio_optimizer import create_style_constraints


class ConstraintTests(unittest.TestCase):
    def test_style_constraints_include_full_investment(self):
        constraints = create_style_constraints(
            ["Equity", "Bonds", "Cash"],
            ["Equity"],
            lower=0.2,
            upper=0.6,
        )
        weights = np.array([0.4, 0.4, 0.2])
        self.assertAlmostEqual(constraints[0]["fun"](weights), 0.0)
        self.assertGreaterEqual(constraints[1]["fun"](weights), 0.0)
        self.assertGreaterEqual(constraints[2]["fun"](weights), 0.0)

    def test_unknown_style_asset_is_rejected(self):
        with self.assertRaises(ValueError):
            create_style_constraints(["Equity", "Bonds"], ["Property"], 0.2, 0.6)


if __name__ == "__main__":
    unittest.main()
