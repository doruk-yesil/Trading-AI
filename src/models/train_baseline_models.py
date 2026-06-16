"""
train_baseline_models.py
------------------------
Trains simple scikit-learn baseline models for comparison.

Baselines to implement:
  - LogisticRegression       — fast linear baseline
  - RandomForestClassifier   — captures nonlinear patterns, helps with feature importance

These are NOT the main model. Their job is to set a performance floor.
If LightGBM doesn't beat these, something is wrong.

Walk-forward split:
  Train on first N years → validate on next M → test on final period.
  Never train on data that is chronologically after the test period.
  No random shuffling of the time series.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional


def get_feature_columns(df: pd.DataFrame, target_col: str = "target") -> list[str]:
    """
    Return the list of feature columns (exclude target, raw OHLCV, and forward-looking cols).
    """
    # TODO: refine this exclusion list as features are finalised
    exclude = {
        target_col, "open", "high", "low", "close", "volume",
        "forward_return_5d",   # raw future return — must never be a feature
    }
    return [c for c in df.columns if c not in exclude]


def time_split(
    df: pd.DataFrame,
    train_years: int = 4,
    val_years:   int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train / validation / test split.
    Remaining data after train + val is used as test.

    Returns (train_df, val_df, test_df)
    """
    # TODO: implement walk-forward version in a later iteration
    n = len(df)
    total_years = (df.index[-1] - df.index[0]).days / 365.25
    train_frac = train_years / total_years
    val_frac   = val_years   / total_years

    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_logistic_pipeline() -> Pipeline:
    """LogisticRegression with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_random_forest_pipeline() -> Pipeline:
    """RandomForestClassifier — no scaling needed but included for API consistency."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "target",
    model_type: str = "logistic",
) -> dict:
    """
    Train a baseline model and return evaluation metrics on the test split.

    Parameters
    ----------
    df : pd.DataFrame
        Feature + target DataFrame indexed by date.
    target_col : str
        Name of the target column.
    model_type : str
        'logistic' or 'random_forest'.

    Returns
    -------
    dict
        {'model': fitted_pipeline, 'train_df': ..., 'val_df': ..., 'test_df': ...}
    """
    # TODO: implement when data arrives
    df_clean = df.dropna(subset=[target_col])
    feature_cols = get_feature_columns(df_clean, target_col)

    train_df, val_df, test_df = time_split(df_clean)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df[target_col]

    if model_type == "logistic":
        pipeline = build_logistic_pipeline()
    elif model_type == "random_forest":
        pipeline = build_random_forest_pipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline.fit(X_train, y_train)

    return {
        "model":    pipeline,
        "train_df": train_df,
        "val_df":   val_df,
        "test_df":  test_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    print("train_baseline_models.py — run via main.py once data is available.")
