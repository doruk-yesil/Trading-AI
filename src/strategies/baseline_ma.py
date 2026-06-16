"""
baseline_ma.py
--------------
Simple moving average crossover baseline strategy.

Logic:
  - Buy signal (1) when SMA_short crosses above SMA_long.
  - Sell / no position (0) when SMA_short crosses below SMA_long.
  - Long-only. No leverage. No short selling.

This is a rule-based benchmark. Its performance sets the bar that the ML model must beat.
Transaction costs must be applied by the backtest engine, not here.

Common configurations:
  - SMA(5) vs SMA(20)  — short-term trend
  - SMA(20) vs SMA(50) — medium-term trend
  - SMA(50) vs SMA(200) — long-term (golden cross / death cross)
"""

import pandas as pd
from typing import Optional


def ma_crossover_signals(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window:  int = 50,
) -> pd.Series:
    """
    Generate long-only buy/hold signals from an MA crossover.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame indexed by date with a 'close' column.
    short_window : int
        Short MA period.
    long_window : int
        Long MA period.

    Returns
    -------
    pd.Series
        Signal series aligned to df.index:
          1 = hold long position
          0 = out of market
    """
    # TODO: implement when data arrives
    sma_short = df["close"].rolling(short_window).mean()
    sma_long  = df["close"].rolling(long_window).mean()

    signal = (sma_short > sma_long).astype(int)
    return signal.rename(f"signal_ma_{short_window}_{long_window}")


if __name__ == "__main__":
    print("baseline_ma.py — import ma_crossover_signals() to use.")
