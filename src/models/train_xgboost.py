"""
train_xgboost.py
----------------
Trains LightGBM and XGBoost gradient boosting models.

These are the primary ML models for v1.
Both support:
  - Feature importance (useful for understanding what drives signals)
  - Early stopping on validation set (reduces overfitting risk)
  - GPU training (optional — set device='gpu' for LightGBM, tree_method='gpu_hist' for XGBoost)

Dispatch: configs/config.yaml -> target_type ("binary" | "continuous")
switches both models between classification and regression objectives.
Continuous is the current default (see configs/config.yaml) -- predicts the
actual forward return, not a threshold beat/miss flag.

Walk-forward training:
  The first version uses a simple chronological split (by calendar date --
  see src.models.train_baseline_models.time_split).
  A proper walk-forward (expanding window, quarterly retraining) is the v2 upgrade.

Overfitting warnings:
  - Use early stopping — always.
  - Monitor train metric vs val metric gap.
  - Do NOT tune hyperparameters on the test set.
  - Keep the test set untouched until the very final evaluation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from src.models.train_baseline_models import get_feature_columns, time_split, prepare_features
from src.utils.config import cfg

IS_CONTINUOUS = cfg.get("target_type", "binary") == "continuous"


def train_lightgbm(
    df: pd.DataFrame,
    target_col: str = "target",
    use_gpu: bool = True,
) -> dict:
    """
    Train a LightGBM model -- regression or binary classification depending
    on configs/config.yaml -> target_type.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled feature + target DataFrame (see load_featured_universe(concat=True)).
    target_col : str
        Target column name.
    use_gpu : bool
        Train on GPU (default True -- Doruk has an RTX 3080). Tries LightGBM's
        "cuda" device first (NVIDIA-specific, fastest), then "gpu" (OpenCL,
        broader compatibility), and only falls back to CPU if BOTH GPU
        backends fail to actually initialize (e.g. the installed lightgbm
        wheel wasn't built with GPU support -- this varies by platform/pip
        build and isn't guaranteed just because a GPU is present). Each
        attempt is printed clearly so it's never silently running on CPU
        without you knowing.

    Returns
    -------
    dict
        {'model', 'train_df', 'val_df', 'test_df', 'feature_cols',
         'feature_importance', 'device_used'}
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    df_clean      = df.dropna(subset=[target_col])
    feature_cols  = get_feature_columns(df_clean, target_col)
    train_df, val_df, test_df = time_split(df_clean)

    X_train = prepare_features(train_df, feature_cols)
    y_train = train_df[target_col]
    X_val   = prepare_features(val_df, feature_cols)
    y_val   = val_df[target_col]

    base_params = {
        "num_leaves":     63,
        "learning_rate":  0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "min_child_samples": 50,
        "random_state":   cfg["model"]["random_seed"],
        "verbose":        -1,
    }
    if IS_CONTINUOUS:
        base_params.update({"objective": "regression", "metric": "rmse"})
    else:
        base_params.update({"objective": "binary", "metric": "auc"})

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]

    devices_to_try = ["cuda", "gpu", "cpu"] if use_gpu else ["cpu"]
    model = None
    device_used = None
    for device in devices_to_try:
        try:
            print(f"  LightGBM: attempting device='{device}'...")
            model = lgb.train(
                {**base_params, "device": device},
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            device_used = device
            print(f"  LightGBM: trained successfully on device='{device}'.")
            break
        except Exception as exc:
            print(f"  LightGBM: device='{device}' failed ({exc}).")
            if device == devices_to_try[-1]:
                raise
            print("  Falling back to the next device...")

    fi = pd.Series(model.feature_importance("gain"), index=feature_cols).sort_values(ascending=False)

    return {
        "model":              model,
        "train_df":           train_df,
        "val_df":             val_df,
        "test_df":            test_df,
        "feature_cols":       feature_cols,
        "feature_importance": fi,
        "device_used":        device_used,
    }


def train_xgboost(
    df: pd.DataFrame,
    target_col: str = "target",
    use_gpu: bool = True,
) -> dict:
    """
    Train an XGBoost model -- regression or binary classification depending
    on configs/config.yaml -> target_type.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled feature + target DataFrame.
    target_col : str
        Target column name.
    use_gpu : bool
        Train on GPU (default True -- Doruk has an RTX 3080). The pip
        xgboost wheel bundles CUDA support on Windows/Linux out of the box
        (unlike LightGBM, which needs a GPU-specific build), so this should
        just work if NVIDIA drivers are installed -- but falls back to CPU
        with a clear message if GPU init fails for any reason (missing
        driver, no CUDA runtime, etc.) rather than crashing the whole run.

    Returns
    -------
    dict
        {'model', 'train_df', 'val_df', 'test_df', 'feature_cols', 'device_used'}
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    df_clean      = df.dropna(subset=[target_col])
    feature_cols  = get_feature_columns(df_clean, target_col)
    train_df, val_df, test_df = time_split(df_clean)

    X_train = prepare_features(train_df, feature_cols)
    y_train = train_df[target_col]
    X_val   = prepare_features(val_df, feature_cols)
    y_val   = val_df[target_col]

    def build_model(device: str):
        common = dict(
            n_estimators       = 1000,
            learning_rate      = 0.05,
            max_depth          = 6,
            subsample          = 0.8,
            colsample_bytree   = 0.8,
            min_child_weight   = 10,
            tree_method        = "hist",
            device             = device,
            early_stopping_rounds = 50,
            random_state       = cfg["model"]["random_seed"],
            verbosity          = 0,
        )
        if IS_CONTINUOUS:
            return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
        return xgb.XGBClassifier(eval_metric="auc", **common)

    devices_to_try = ["cuda", "cpu"] if use_gpu else ["cpu"]
    model = None
    device_used = None
    for device in devices_to_try:
        try:
            print(f"  XGBoost: attempting device='{device}'...")
            model = build_model(device)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
            device_used = device
            print(f"  XGBoost: trained successfully on device='{device}'.")
            break
        except Exception as exc:
            print(f"  XGBoost: device='{device}' failed ({exc}).")
            if device == devices_to_try[-1]:
                raise
            print("  Falling back to CPU...")

    return {
        "model":        model,
        "train_df":     train_df,
        "val_df":       val_df,
        "test_df":      test_df,
        "feature_cols": feature_cols,
        "device_used":  device_used,
    }


if __name__ == "__main__":
    print("train_xgboost.py — run via scripts/train_model.py.")
