"""
target_builder.py
-----------------
Builds the prediction target (label) for supervised learning.

Target definition:
  Binary label = 1 if the stock's forward N-day return exceeds
                 (transaction_cost + min_return_threshold), else 0.

  forward_return = (close[t+N] / close[t]) - 1

CRITICAL — LOOKAHEAD BIAS:
  The target uses FUTURE prices. This means:
  1. Drop all rows where the target cannot be computed (last N rows of each series).
  2. NEVER let the target leak into features during training.
  3. Train/validation/test split must be done BEFORE fitting any scaler or encoder.
  4. The target column must be excluded from the feature matrix X.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.config import cfg


def build_binary_target(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    min_threshold: Optional[float] = None,
    transaction_cost: Optional[float] = None,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Add a binary classification target to the DataFrame.

    Label = 1  if  forward_return > transaction_cost + min_threshold
    Label = 0  otherwise

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + features DataFrame indexed by date.
    horizon : int
        Forward-looking window in trading days.
    min_threshold : float
        Minimum net return threshold (e.g. 0.005 = 0.5%).
    transaction_cost : float
        One-way transaction cost (e.g. 0.001 = 0.1%).
    target_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new columns:
          - 'forward_return_Nd' : raw future return
          - target_col          : binary label (0/1), NaN for last N rows
    """
    # TODO: implement when data arrives
    horizon          = horizon          or cfg["prediction_horizon"]
    min_threshold    = min_threshold    or cfg["min_return_threshold"]
    transaction_cost = transaction_cost or cfg["transaction_cost_pct"]

    df = df.copy()

    # Forward return (uses future close — drop NaN rows before training)
    df[f"forward_return_{horizon}d"] = df["close"].pct_change(horizon).shift(-horizon)

    # Binary label
    hurdle = transaction_cost + min_threshold
    df[target_col] = (df[f"forward_return_{horizon}d"] > hurdle).astype(float)

    # Mark last `horizon` rows as NaN (no future data available)
    df.loc[df.index[-horizon:], target_col] = np.nan
    df.loc[df.index[-horizon:], f"forward_return_{horizon}d"] = np.nan

    return df


def build_continuous_target(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    target_col: str = "target_return",
) -> pd.DataFrame:
    """
    Add a continuous return target for regression tasks.

    Not used in v1 (we use binary classification), but kept for future experimentation.
    """
    # TODO: implement if regression approach is explored
    horizon = horizon or cfg["prediction_horizon"]
    df = df.copy()
    df[target_col] = df["close"].pct_change(horizon).shift(-horizon)
    df.loc[df.index[-horizon:], target_col] = np.nan
    return df


if __name__ == "__main__":
    print("target_builder.py — call build_binary_target(df) after loading data.")
