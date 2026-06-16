"""
ml_signal_strategy.py
---------------------
Converts ML model predictions into tradeable signals.

The ML model outputs a probability p(label=1) for each row.
This module applies a confidence threshold and turns probabilities into
buy/hold/exit signals for the backtest engine.

Design decisions (v1):
  - Long-only.
  - Signal = 1 if p >= threshold, else 0.
  - Threshold is tuned on the validation set, not the test set.
  - No position sizing by confidence yet (v2 idea).
"""

import pandas as pd
import numpy as np
from typing import Optional


def probability_to_signal(
    probabilities: pd.Series,
    threshold: float = 0.55,
    signal_name: str = "signal_ml",
) -> pd.Series:
    """
    Convert predicted probabilities to binary long/exit signals.

    Parameters
    ----------
    probabilities : pd.Series
        Model-predicted probability of label=1, indexed by date.
    threshold : float
        Minimum probability to enter / stay in position.
        Default 0.55 (slightly above random) — tune on validation set.
    signal_name : str
        Name of the returned Series.

    Returns
    -------
    pd.Series
        1 = enter/hold long, 0 = exit.
    """
    # TODO: implement when model is trained
    signal = (probabilities >= threshold).astype(int)
    signal.name = signal_name
    return signal


def apply_signal_filters(
    signal: pd.Series,
    min_hold_days: int = 1,
) -> pd.Series:
    """
    Optional post-processing filters on the signal.

    min_hold_days: avoid rapid signal flipping by requiring the signal
    to stay consistent for at least N consecutive days before switching.

    Currently a placeholder — implement if overfitting to noise is observed.
    """
    # TODO: implement if signal turnover is too high in backtest
    return signal


if __name__ == "__main__":
    print("ml_signal_strategy.py — import probability_to_signal() to use.")
