"""
metrics.py
----------
Financial performance metrics for backtest evaluation.

Key metrics:
  - Total return
  - Annualised return (CAGR)
  - Annualised volatility
  - Sharpe ratio (annualised, risk-free rate adjustable)
  - Sortino ratio
  - Maximum drawdown and drawdown duration
  - Calmar ratio (CAGR / max drawdown)
  - Win rate
  - Profit factor
  - Number of trades

Reference:
  - All metrics are computed on out-of-sample (test set) data only.
  - Benchmark comparison uses SPY buy-and-hold over the same period.
"""

import pandas as pd
import numpy as np
from typing import Optional


TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,  # approximate 2024-2025 level
) -> dict:
    """
    Compute a full set of financial performance metrics.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns (not cumulative).
    benchmark_returns : pd.Series, optional
        Daily benchmark (SPY) returns for comparison.
    risk_free_rate : float
        Annual risk-free rate used for Sharpe/Sortino.

    Returns
    -------
    dict
        All computed metrics.
    """
    # TODO: implement when data arrives
    r = strategy_returns.dropna()
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Returns
    total_return    = (1 + r).prod() - 1
    n_years         = len(r) / TRADING_DAYS_PER_YEAR
    cagr            = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol         = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe
    excess = r - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess.std() > 0 else 0.0

    # Sortino (downside deviation)
    downside = r[r < daily_rf] - daily_rf
    sortino  = (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside.std() > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd   = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate (positive return days)
    win_rate = (r > 0).mean()

    metrics = {
        "total_return":     round(total_return, 4),
        "cagr":             round(cagr, 4),
        "ann_volatility":   round(ann_vol, 4),
        "sharpe_ratio":     round(sharpe, 4),
        "sortino_ratio":    round(sortino, 4),
        "max_drawdown":     round(max_dd, 4),
        "calmar_ratio":     round(calmar, 4),
        "win_rate":         round(win_rate, 4),
        "n_days":           len(r),
        "n_years":          round(n_years, 2),
    }

    # Benchmark comparison
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(r.index).dropna()
        bm_total = (1 + bm).prod() - 1
        metrics["benchmark_total_return"] = round(bm_total, 4)
        metrics["alpha_vs_benchmark"]     = round(total_return - bm_total, 4)

    return metrics


def print_metrics(metrics: dict, label: str = "Strategy") -> None:
    """Pretty-print metrics dict."""
    print(f"\n── {label} Performance ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:>10.4f}")
        else:
            print(f"  {k:<28} {v:>10}")


if __name__ == "__main__":
    print("metrics.py — import compute_metrics() to use.")
