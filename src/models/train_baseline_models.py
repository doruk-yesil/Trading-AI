"""
train_baseline_models.py
------------------------
Trains simple scikit-learn baseline models for comparison.

Baselines to implement:
  - Ridge / LogisticRegression   — fast linear baseline (Ridge for continuous
                                    target, LogisticRegression for binary).
                                    Ridge specifically chosen over plain
                                    LinearRegression because the 173 features
                                    are heavily collinear (many are different
                                    transforms of the same underlying price
                                    series) -- L2 regularization handles that
                                    much better than unregularized OLS.
  - RandomForest(Regressor|Classifier) — captures nonlinear patterns, helps
                                    with feature importance.

These are NOT the main model. Their job is to set a performance floor.
If LightGBM/XGBoost don't beat these, something is wrong.

Dispatch: configs/config.yaml -> target_type ("binary" | "continuous")
controls which variant of each baseline gets built and how it's evaluated.
Callers don't need to know which -- just call train_and_evaluate(df).

Walk-forward split:
  Train on first N years -> validate on next M -> test on final period.
  Never train on data that is chronologically after the test period.
  No random shuffling of the time series. Split is done by calendar DATE,
  not row position -- required for a pooled cross-sectional panel (many
  symbols stacked together with different amounts of history each).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import cfg
from src.models.evaluate_model import evaluate_classifier, evaluate_regressor

# Columns that must never be used as features regardless of target_type.
NON_FEATURE_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume", "target"}


def get_feature_columns(df: pd.DataFrame, target_col: str = "target") -> list[str]:
    """
    Return the list of feature columns -- excludes target, raw OHLCV,
    identifier columns (date/symbol), and any forward_return_* column
    (the raw future-return column the target is derived from -- must
    never leak into the feature matrix regardless of prediction_horizon).
    """
    exclude = NON_FEATURE_COLUMNS | {target_col}
    return [
        c for c in df.columns
        if c not in exclude and not c.startswith("forward_return_")
    ]


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Build the X matrix for training/inference: selects feature_cols and
    replaces both NaN AND +/-inf with 0.

    Why inf matters: several ratio-style technical indicators (volume
    ratios, price ratios, %-change features) divide by a value that is
    occasionally exactly 0 (e.g. a zero-volume bar, a flat candle), which
    produces +/-inf rather than NaN. A plain .fillna(0) silently leaves
    those infinities in place and sklearn/LightGBM/XGBoost all reject them
    with 'Input X contains infinity' -- discovered by actually running this
    against real AAPL/MSFT/SPY data, not a hypothetical.
    """
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def time_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_years: Optional[float] = None,
    val_years: Optional[float] = None,
    test_years: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split by calendar DATE, anchored
    to the MOST RECENT date working backward -- NOT the earliest date working
    forward.

    Why backward: this panel pools symbols with wildly different amounts of
    history (some S&P 500 constituents go back to the 1960s, some IPO'd last
    year). Anchoring train_years to the panel's earliest date would size
    train/val/test off whichever symbol happens to have the oldest history --
    caught this by actually running it: it put ~99% of rows in "test" and a
    tiny 1960s sliver in "train". Anchoring to the latest date instead gives:
    test = the most recent test_years, val = the test_years-years before
    that, train = the train_years immediately before val. Symbols with less
    history than the train window simply contribute fewer (or zero) rows to
    train -- expected panel imbalance, not a bug.

    Deliberately NOT a row-fraction split either: with a pooled panel, row
    position doesn't correspond to a single calendar date, so every row is
    assigned to train/val/test based on its own date, uniformly across the
    whole panel.

    Returns (train_df, val_df, test_df).
    """
    train_years = train_years if train_years is not None else cfg["train_years"]
    val_years   = val_years   if val_years   is not None else cfg["validation_years"]
    test_years  = test_years  if test_years  is not None else cfg.get("test_years", 1)

    dates = pd.to_datetime(df[date_col])
    end = dates.max()
    test_start  = end - pd.Timedelta(days=round(test_years * 365.25))
    val_start   = test_start - pd.Timedelta(days=round(val_years * 365.25))
    train_start = val_start - pd.Timedelta(days=round(train_years * 365.25))

    train_df = df.loc[(dates >= train_start) & (dates < val_start)]
    val_df   = df.loc[(dates >= val_start) & (dates < test_start)]
    test_df  = df.loc[dates >= test_start]
    return train_df, val_df, test_df


def build_linear_pipeline() -> Pipeline:
    """Ridge (continuous) or LogisticRegression (binary) with scaling."""
    if cfg.get("target_type", "binary") == "continuous":
        clf = Ridge(alpha=1.0, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_random_forest_pipeline() -> Pipeline:
    """RandomForestRegressor (continuous) or RandomForestClassifier (binary)."""
    if cfg.get("target_type", "binary") == "continuous":
        clf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# Kept for backwards compatibility with any older callers.
build_logistic_pipeline = build_linear_pipeline


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "target",
    model_type: str = "linear",
) -> dict:
    """
    Train a baseline model and evaluate on validation + test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled feature + target DataFrame (multiple symbols stacked, with a
        'date' column -- see src.data.load_data.load_featured_universe(concat=True)).
    target_col : str
        Name of the target column.
    model_type : str
        'linear' (Ridge/LogisticRegression) or 'random_forest'.
        ('logistic' accepted as an alias for 'linear' for readability.)

    Returns
    -------
    dict
        {'model', 'feature_cols', 'val_metrics', 'test_metrics',
         'train_df', 'val_df', 'test_df'}
    """
    df_clean = df.dropna(subset=[target_col])
    feature_cols = get_feature_columns(df_clean, target_col)

    train_df, val_df, test_df = time_split(df_clean)

    X_train = prepare_features(train_df, feature_cols)
    y_train = train_df[target_col]

    if model_type in ("linear", "logistic"):
        pipeline = build_linear_pipeline()
    elif model_type == "random_forest":
        pipeline = build_random_forest_pipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline.fit(X_train, y_train)

    is_continuous = cfg.get("target_type", "binary") == "continuous"
    val_metrics, test_metrics = {}, {}

    if len(val_df) > 0:
        X_val = prepare_features(val_df, feature_cols)
        if is_continuous:
            val_metrics = evaluate_regressor(
                val_df[target_col], pipeline.predict(X_val),
                min_return_threshold=cfg["min_return_threshold"], label="val",
            )
        else:
            val_metrics = evaluate_classifier(
                val_df[target_col], pipeline.predict_proba(X_val)[:, 1], label="val",
            )

    if len(test_df) > 0:
        X_test = prepare_features(test_df, feature_cols)
        if is_continuous:
            test_metrics = evaluate_regressor(
                test_df[target_col], pipeline.predict(X_test),
                min_return_threshold=cfg["min_return_threshold"], label="test",
            )
        else:
            test_metrics = evaluate_classifier(
                test_df[target_col], pipeline.predict_proba(X_test)[:, 1], label="test",
            )

    return {
        "model": pipeline,
        "feature_cols": feature_cols,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    print("train_baseline_models.py — run via scripts/train_model.py.")
