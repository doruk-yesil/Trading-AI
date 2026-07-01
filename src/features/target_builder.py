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
    target_col: str = "target",
    clip_quantiles: Optional[tuple] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Add a continuous forward-return target for regression.

    Predicts the actual forward N-day return (as a decimal, e.g. 0.023 = +2.3%)
    instead of a binary beat/miss flag -- gives direction + magnitude, which is
    what a downstream position-sizing / cross-sectional ranking step needs
    (rank all symbols by predicted return, not just P(beats threshold)).

    min_return_threshold / transaction_cost are NOT applied here -- with a
    continuous target the "is this worth trading" decision belongs in the
    strategy layer (src/strategies/ml_signal_strategy.py), which compares the
    predicted return against cfg["min_return_threshold"] + costs. Keeping
    that logic out of the label keeps prediction and decision-making separate.

    Parameters
    ----------
    clip_quantiles : (float, float) or None
        Winsorizes extreme outliers (earnings gaps, halts, etc.) at these
        quantiles before they hit training -- a handful of extreme return
        days can otherwise dominate a regression loss and destabilize
        training. Set to None to disable.
    """
    horizon = horizon or cfg["prediction_horizon"]
    df = df.copy()

    raw_col = f"forward_return_{horizon}d"
    df[raw_col] = df["close"].pct_change(horizon).shift(-horizon)
    df[target_col] = df[raw_col]

    if clip_quantiles is not None:
        lo = df[target_col].quantile(clip_quantiles[0])
        hi = df[target_col].quantile(clip_quantiles[1])
        df[target_col] = df[target_col].clip(lower=lo, upper=hi)

    # Mark last `horizon` rows as NaN (no future data available)
    df.loc[df.index[-horizon:], target_col] = np.nan
    df.loc[df.index[-horizon:], raw_col] = np.nan

    return df


def build_target(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Dispatches to build_binary_target or build_continuous_target based on
    configs/config.yaml -> target_type. Callers should use this instead of
    hardcoding one or the other, so switching target_type is a one-line
    config change, not a code change in every caller.
    """
    target_type = cfg.get("target_type", "binary")
    if target_type == "binary":
        return build_binary_target(df, **kwargs)
    elif target_type == "continuous":
        # build_binary_target-style kwargs that build_continuous_target
        # doesn't use (it's threshold-free by design -- see docstring above).
        kwargs.pop("min_threshold", None)
        kwargs.pop("transaction_cost", None)
        return build_continuous_target(df, **kwargs)
    else:
        raise ValueError(
            f"Unknown target_type: {target_type!r} (expected 'binary' or 'continuous')"
        )


if __name__ == "__main__":
    print("target_builder.py — call build_target(df) after loading data "
          "(dispatches based on configs/config.yaml -> target_type).")
