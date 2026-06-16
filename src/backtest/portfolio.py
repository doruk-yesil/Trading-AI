"""
portfolio.py
------------
Portfolio-level aggregation across multiple symbols.

v1 approach: equal-weight portfolio of all symbols with signal=1.
  - At each rebalance date, divide capital equally among all active signals.
  - Max position size capped by config.position_size_pct.
  - No leverage.

v2 ideas (not implemented yet):
  - Risk-parity weighting (inverse volatility).
  - Signal-strength weighting (proportional to predicted probability).
  - Sector-aware constraints.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.config import cfg


def equal_weight_portfolio(
    signals: dict[str, pd.Series],
    prices:  dict[str, pd.Series],
    initial_capital: Optional[float] = None,
    max_position_pct: Optional[float] = None,
    transaction_cost_pct: Optional[float] = None,
    slippage_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulate an equal-weight long-only portfolio.

    Parameters
    ----------
    signals : dict
        {symbol: pd.Series of binary signals indexed by date}
    prices : dict
        {symbol: pd.Series of close prices indexed by date}
    initial_capital : float
        Starting portfolio value in USD.
    max_position_pct : float
        Maximum fraction of portfolio in any single position.
    transaction_cost_pct : float
        One-way trade cost.
    slippage_pct : float
        Estimated slippage.

    Returns
    -------
    pd.DataFrame
        Columns: equity_curve, daily_portfolio_return, n_positions
    """
    # TODO: implement when data arrives
    initial_capital      = initial_capital      or cfg["initial_capital"]
    max_position_pct     = max_position_pct     or cfg["position_size_pct"]
    transaction_cost_pct = transaction_cost_pct or cfg["transaction_cost_pct"]
    slippage_pct         = slippage_pct         or cfg["slippage_pct"]
    total_cost_pct       = transaction_cost_pct + slippage_pct

    # Align all signals and prices to common dates
    all_dates = sorted(set.intersection(*[set(s.index) for s in signals.values()]))
    signals_df = pd.DataFrame({sym: sig.reindex(all_dates) for sym, sig in signals.items()})
    prices_df  = pd.DataFrame({sym: px.reindex(all_dates)  for sym, px in prices.items()})

    # Shift signals: execute next day
    positions = signals_df.shift(1).fillna(0)

    # Number of active positions each day
    n_pos = positions.sum(axis=1).clip(lower=1)  # avoid div-by-zero

    # Equal weight, capped by max_position_pct
    raw_weights = positions.div(n_pos, axis=0)
    weights = raw_weights.clip(upper=max_position_pct)

    # Daily returns per symbol
    returns_df = prices_df.pct_change()

    # Weighted portfolio return
    portfolio_return = (weights * returns_df).sum(axis=1)

    # Subtract transaction costs on rebalance days
    trades = positions.diff().abs().sum(axis=1)
    portfolio_return -= trades * total_cost_pct

    equity_curve = initial_capital * (1 + portfolio_return).cumprod()

    result = pd.DataFrame({
        "equity_curve":            equity_curve,
        "daily_portfolio_return":  portfolio_return,
        "n_positions":             positions.sum(axis=1),
    }, index=all_dates)

    return result


if __name__ == "__main__":
    print("portfolio.py — import equal_weight_portfolio() to use.")
