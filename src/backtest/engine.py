"""
engine.py
---------
Simple vectorised backtesting engine.

Design:
  - Vectorised (pandas-based), not event-driven.
  - Supports long-only positions.
  - Applies transaction costs and slippage on each trade.
  - No leverage in v1.
  - Single-asset mode for now; portfolio aggregation is in portfolio.py.

Transaction cost model:
  cost = position_value * (transaction_cost_pct + slippage_pct)
  Applied once on entry and once on exit.

Lookahead bias checks:
  - Signals must be shifted by 1 day before applying to returns.
    (signal generated end-of-day → executed next day's open)
  - Do NOT use same-day close to execute same-day signal.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.config import cfg


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    transaction_cost_pct: Optional[float] = None,
    slippage_pct: Optional[float] = None,
    initial_capital: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run a vectorised long-only backtest.

    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by date.
    signals : pd.Series
        Binary signal (1 = hold long, 0 = out), indexed by same dates.
        Signal generated at close of day t → trade executed at close of day t+1.
    transaction_cost_pct : float
        One-way cost as fraction of position value.
    slippage_pct : float
        Estimated slippage per trade.
    initial_capital : float
        Starting cash in USD.

    Returns
    -------
    pd.DataFrame
        Columns: close, signal, position, daily_return, strategy_return, equity_curve
    """
    # TODO: implement when data arrives
    transaction_cost_pct = transaction_cost_pct or cfg["transaction_cost_pct"]
    slippage_pct         = slippage_pct         or cfg["slippage_pct"]
    initial_capital      = initial_capital      or cfg["initial_capital"]
    total_cost_pct       = transaction_cost_pct + slippage_pct

    df = pd.DataFrame({"close": prices, "signal": signals}).dropna()

    # Shift signal by 1 day — execute next day
    df["position"] = df["signal"].shift(1).fillna(0)

    # Daily market return
    df["daily_return"] = df["close"].pct_change()

    # Detect trades (position changes)
    df["trade"] = df["position"].diff().abs()

    # Strategy return = position * market return - transaction cost on trade days
    df["strategy_return"] = (
        df["position"] * df["daily_return"]
        - df["trade"] * total_cost_pct
    )

    # Equity curve
    df["equity_curve"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    return df


if __name__ == "__main__":
    print("engine.py — import run_backtest() to use.")
