# Portfolio Optimization Toolkit

A compact Python toolkit and notebook collection for constrained static mean–variance portfolio optimisation, covariance-matrix repair and sensitivity analysis.

The repository uses reproducible synthetic examples so that the optimisation and numerical behaviour can be inspected without external or proprietary data.

## Features

- Maximum-Sharpe, global-minimum-variance and target-return portfolios using SciPy SLSQP.
- Full-investment, position-bound and named asset-bucket constraints.
- Input, solver-success and post-solution feasibility checks.
- Symmetric positive-semidefinite projection by eigenvalue clipping.
- Performance tables and allocation plots.
- Three notebooks covering basic use, risk profiles and covariance-repair sensitivity.

## Installation

```bash
git clone https://github.com/WuYefan77/Portfolio-Optimization-Toolkit.git
cd Portfolio-Optimization-Toolkit
python -m pip install -e .
```

Install the notebook and test dependencies when needed:

```bash
python -m pip install -e ".[notebooks,test]"
```

## Quick start

```python
import numpy as np

from portfolio_optimizer import StaticOptimizer, create_style_constraints

expected_returns = np.array([0.10, 0.08, 0.04])
covariance = np.array([
    [0.0225, 0.0135, 0.0015],
    [0.0135, 0.0144, 0.0024],
    [0.0015, 0.0024, 0.0025],
])
assets = ["Equity A", "Equity B", "Bonds"]

optimizer = StaticOptimizer(expected_returns, covariance)
bounds = tuple((0.0, 0.60) for _ in assets)
constraints = create_style_constraints(
    assets,
    ["Equity A", "Equity B"],
    lower=0.40,
    upper=0.80,
)

result = optimizer.maximize_sharpe(
    risk_free_rate=0.02,
    bounds=bounds,
    constraints=constraints,
)
print(np.round(result.x, 4))
print(result.max_constraint_violation)
```

`StaticOptimizer` raises `OptimizationError` when SLSQP fails or returns a materially infeasible solution. It does not clip or renormalise the optimiser output after solving, because doing so can silently violate the supplied constraints.

## Notebooks

```text
01_Static_Optimization_Showcase.ipynb  constrained optimisation and reporting
02_Risk_Profile_Analysis.ipynb         allocation bands across mock scenarios
03_Covariance_Repair_Sensitivity.ipynb sensitivity to covariance repair
```

The third notebook distinguishes numerical repair from statistical estimation: projecting an invalid covariance matrix onto the PSD cone makes optimisation possible, but does not recover the unknown covariance structure that generated the data.

## Numerical scope

This project focuses on one-period, ex-ante mean--variance allocation. It does not include return forecasting, transaction costs, turnover controls, estimation-error corrections, backtesting or production portfolio management infrastructure.

## License

MIT
