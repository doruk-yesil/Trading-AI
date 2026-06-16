"""
train_xgboost.py
----------------
Trains LightGBM and XGBoost gradient boosting models.

These are the primary ML models for v1.
Both support:
  - Feature importance (useful for understanding what drives signals)
  - Early stopping on validation set (reduces overfitting risk)
  - GPU training (optional — set device='gpu' for LightGBM, tree_method='gpu_hist' for XGBoost)

Walk-forward training:
  The first version uses a simple chronological split.
  A proper walk-forward (expanding window, quarterly retraining) is the v2 upgrade.

Overfitting warnings:
  - Use early stopping — always.
  - Monitor train AUC vs val AUC gap.
  - Do NOT tune hyperparameters on the test set.
  - Keep the test set untouched until the very final evaluation.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.models.train_baseline_models import get_feature_columns, time_split
from src.utils.config import cfg


def train_lightgbm(
    df: pd.DataFrame,
    target_col: str = "target",
    use_gpu: bool = False,
) -> dict:
    """
    Train a LightGBM binary classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Feature + target DataFrame.
    target_col : str
        Binary target column name.
    use_gpu : bool
        Enable GPU training (requires LightGBM GPU build + CUDA).

    Returns
    -------
    dict
        {'model', 'train_df', 'val_df', 'test_df', 'feature_cols', 'feature_importance'}
    """
    # TODO: implement when data arrives
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    df_clean      = df.dropna(subset=[target_col])
    feature_cols  = get_feature_columns(df_clean, target_col)
    train_df, val_df, test_df = time_split(df_clean)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val   = val_df[feature_cols].fillna(0)
    y_val   = val_df[target_col]

    device = "gpu" if use_gpu else "cpu"

    params = {
        "objective":      "binary",
        "metric":         "auc",
        "num_leaves":     63,
        "learning_rate":  0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "min_child_samples": 50,
        "device":         device,
        "random_state":   cfg["model"]["random_seed"],
        "verbose":        -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    fi = pd.Series(model.feature_importance("gain"), index=feature_cols).sort_values(ascending=False)

    return {
        "model":              model,
        "train_df":           train_df,
        "val_df":             val_df,
        "test_df":            test_df,
        "feature_cols":       feature_cols,
        "feature_importance": fi,
    }


def train_xgboost(
    df: pd.DataFrame,
    target_col: str = "target",
    use_gpu: bool = False,
) -> dict:
    """
    Train an XGBoost binary classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Feature + target DataFrame.
    target_col : str
        Binary target column name.
    use_gpu : bool
        Enable GPU training (requires CUDA-enabled XGBoost build).

    Returns
    -------
    dict
        {'model', 'train_df', 'val_df', 'test_df', 'feature_cols'}
    """
    # TODO: implement when data arrives
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    df_clean      = df.dropna(subset=[target_col])
    feature_cols  = get_feature_columns(df_clean, target_col)
    train_df, val_df, test_df = time_split(df_clean)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val   = val_df[feature_cols].fillna(0)
    y_val   = val_df[target_col]

    tree_method = "hist"
    device      = "cuda" if use_gpu else "cpu"

    model = xgb.XGBClassifier(
        n_estimators       = 1000,
        learning_rate      = 0.05,
        max_depth          = 6,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 10,
        tree_method        = tree_method,
        device             = device,
        eval_metric        = "auc",
        early_stopping_rounds = 50,
        random_state       = cfg["model"]["random_seed"],
        verbosity          = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    return {
        "model":        model,
        "train_df":     train_df,
        "val_df":       val_df,
        "test_df":      test_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    print("train_xgboost.py — run via main.py once data is available.")
