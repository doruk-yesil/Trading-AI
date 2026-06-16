"""
baseline_rsi.py
---------------
RSI-based mean-reversion baseline strategy.

Logic:
  - Buy signal (1) when RSI crosses below oversold_threshold (e.g. 30).
  - Exit (0) when RSI rises above overbought_threshold (e.g. 70).
  - Long-only. No short selling.

Mean reversion assumption: extreme RSI readings tend to reverse.
This is a simple, well-known baseline — do not expect it to outperform consistently
on liquid large-cap stocks without additional filters.
"""

import pandas as pd
from typing import Optional

from src.features.technical_indicators import add_rsi


def rsi_signals(
    df: pd.DataFrame,
    oversold:    float = 30.0,
    overbought:  float = 70.0,
    rsi_period:  int   = 14,
) -> pd.Series:
    """
    Generate long-only signals based on RSI thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (will compute RSI if not already present).
    oversold : float
        Enter long when RSI drops below this level.
    overbought : float
        Exit when RSI rises above this level.
    rsi_period : int
        RSI look-back period.

    Returns
    -------
    pd.Series
        Signal series: 1 = in position, 0 = out.
    """
    # TODO: implement when data arrives
    if "rsi" not in df.columns:
        df = add_rsi(df, period=rsi_period)

    signal = pd.Series(0, index=df.index, name="signal_rsi")
    in_position = False

    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.isna(row["rsi"]):
            continue
        if not in_position and row["rsi"] < oversold:
            in_position = True
        elif in_position and row["rsi"] > overbought:
            in_position = False
        signal.at[idx] = int(in_position)

    return signal


if __name__ == "__main__":
    print("baseline_rsi.py — import rsi_signals() to use.")
