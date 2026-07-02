"""
cross_validation.py
--------------------
Purged k-fold cross-validation across multiple contiguous time blocks of the
pooled panel, instead of trusting a single train/val/test split.

Why this exists: a single chronological split (see train_baseline_models.
time_split) tells you how the model does on ONE specific recent period.
That period could happen to be unusually calm or unusually volatile --
Information Coefficient computed on it is one sample, not a reliable
estimate. Before trusting a model with real money, you want to know it
performs consistently across SEVERAL different time periods, not just the
one you happened to test on.

This is a simplified, tractable subset of Marcos Lopez de Prado's full
Combinatorial Purged Cross-Validation (CPCV) from "Advances in Financial
Machine Learning": full CPCV forms every combinatorial train/test PATH across
N groups (C(N,k) paths), which is expensive and mainly useful for estimating
the full distribution of backtest outcomes for deployment sign-off. What's
implemented here -- "purged k-fold" -- is the practical middle ground used
much more commonly in practice: split the timeline into N contiguous blocks,
use each block once as the held-out test fold, purge + embargo the boundary
rows on both sides (same buffer logic as time_split), and train on the
remaining data. That answers the actual question this project needs answered
right now: "does this model's edge hold up across multiple different
periods, or did it get lucky on one split?"

Each fold's "train" pool also gets an internal validation slice carved out
(by date, not row-shuffle) so LightGBM/XGBoost can still use early stopping
within a fold -- this internal val is never used for the reported metric,
only for stopping training. The reported metric is always fold TEST performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator, Optional

from src.utils.config import cfg
from src.models.train_baseline_models import (
    get_feature_columns,
    prepare_features,
    purge_embargo_days,
    train_and_evaluate,
)
from src.models.train_xgboost import train_lightgbm, train_xgboost
from src.models.evaluate_model import evaluate_classifier, evaluate_regressor

BASELINE_MODEL_TYPES = ("linear", "random_forest")
GBM_MODEL_TYPES = ("lightgbm", "xgboost")


def purged_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    date_col: str = "date",
    horizon: Optional[int] = None,
    embargo_extra_days: int = 3,
    val_fraction: float = 0.15,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Yields (fold_index, train_df, val_df, test_df) for n_splits contiguous
    time blocks spanning df's full date range.

    For fold i: the i-th block (by calendar date, evenly spaced across the
    panel's date range) is the TEST set. Everything else is candidate
    train/val, EXCEPT rows within `purge_embargo_days()` of either edge of
    the test block -- those are dropped entirely (purged), since their
    label window overlaps into the test period. The remaining pool is
    sorted by date and the most recent `val_fraction` of it becomes the
    internal validation slice (for GBM early stopping); an additional purge
    buffer is applied between that internal train/val boundary too, for
    consistency with time_split's leakage-safety logic.

    Folds where purging leaves an empty train or test set (can happen with
    very small/short synthetic data, or n_splits set too high for the
    panel's date range) are still yielded -- callers should check for
    len(train_df) == 0 / len(test_df) == 0 and skip.
    """
    dates = pd.to_datetime(df[date_col])
    start, end = dates.min(), dates.max()
    total_days = (end - start).days
    buffer = pd.Timedelta(days=purge_embargo_days(horizon, embargo_extra_days))

    edges = [start + pd.Timedelta(days=round(total_days * i / n_splits)) for i in range(n_splits + 1)]

    for i in range(n_splits):
        test_start, test_end = edges[i], edges[i + 1]
        if i == n_splits - 1:
            test_mask = (dates >= test_start) & (dates <= test_end)
        else:
            test_mask = (dates >= test_start) & (dates < test_end)
        test_df = df.loc[test_mask]

        purge_mask = (dates >= (test_start - buffer)) & (dates < (test_end + buffer))
        pool_df = df.loc[(~test_mask) & (~purge_mask)].sort_values(date_col)

        if len(pool_df) == 0:
            yield i, pool_df, pool_df, test_df
            continue

        val_cutoff = int(round(len(pool_df) * (1 - val_fraction)))
        pool_train = pool_df.iloc[:val_cutoff]
        pool_val = pool_df.iloc[val_cutoff:]

        if len(pool_val) > 0 and len(pool_train) > 0:
            val_start_date = pd.to_datetime(pool_val[date_col]).min()
            pool_train_dates = pd.to_datetime(pool_train[date_col])
            pool_train = pool_train.loc[pool_train_dates < (val_start_date - buffer)]

        yield i, pool_train, pool_val, test_df


def _evaluate_on_test(
    model,
    model_type: str,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    is_continuous: bool,
) -> dict:
    """Evaluate an already-trained GBM model on a fold's test split."""
    X_test = prepare_features(test_df, feature_cols)
    y_test = test_df[target_col]

    if model_type == "lightgbm":
        preds = model.predict(X_test)
    else:  # xgboost
        preds = model.predict(X_test) if is_continuous else model.predict_proba(X_test)[:, 1]

    if is_continuous:
        return evaluate_regressor(
            y_test, preds, min_return_threshold=cfg["min_return_threshold"],
            label=f"{model_type}/fold_test",
        )
    return evaluate_classifier(y_test, preds, label=f"{model_type}/fold_test")


def _summarize_folds(fold_results: list[dict], is_continuous: bool) -> dict:
    if not fold_results:
        return {"n_folds": 0}

    metrics = [f["test_metrics"] for f in fold_results if f["test_metrics"]]
    if not metrics:
        return {"n_folds": len(fold_results), "n_scored_folds": 0}

    if is_continuous:
        ics = [m.get("information_coefficient") for m in metrics]
        dirs = [m.get("directional_accuracy") for m in metrics]
        maes = [m.get("mae") for m in metrics]
        return {
            "n_folds": len(fold_results),
            "n_scored_folds": len(metrics),
            "ic_mean": float(np.nanmean(ics)),
            "ic_std": float(np.nanstd(ics)),
            "directional_accuracy_mean": float(np.nanmean(dirs)),
            "mae_mean": float(np.nanmean(maes)),
        }

    aucs = [m.get("auc") for m in metrics]
    return {
        "n_folds": len(fold_results),
        "n_scored_folds": len(metrics),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
    }


def run_purged_cv(
    df: pd.DataFrame,
    model_type: str = "lightgbm",
    target_col: str = "target",
    n_splits: int = 5,
    use_gpu: bool = True,
    horizon: Optional[int] = None,
    embargo_extra_days: int = 3,
    val_fraction: float = 0.15,
) -> dict:
    """
    Runs purged k-fold CV for a single model type across n_splits time
    blocks and returns per-fold + summary (mean/std) metrics.

    Returns
    -------
    dict
        {'model_type', 'n_splits', 'folds': [per-fold dicts], 'summary'}
    """
    if model_type not in BASELINE_MODEL_TYPES + GBM_MODEL_TYPES:
        raise ValueError(f"Unknown model_type: {model_type}")

    is_continuous = cfg.get("target_type", "binary") == "continuous"
    df_clean = df.dropna(subset=[target_col])

    fold_results = []
    for fold_idx, train_df, val_df, test_df in purged_kfold_splits(
        df_clean, n_splits=n_splits, horizon=horizon,
        embargo_extra_days=embargo_extra_days, val_fraction=val_fraction,
    ):
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Fold {fold_idx}: skipped (empty train or test after purging)")
            continue

        splits = (train_df, val_df, test_df)
        print(f"  Fold {fold_idx}: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
              f"({pd.to_datetime(test_df['date']).min().date()} -> {pd.to_datetime(test_df['date']).max().date()})")

        if model_type in BASELINE_MODEL_TYPES:
            result = train_and_evaluate(train_df, target_col=target_col, model_type=model_type, splits=splits)
            test_metrics = result["test_metrics"]
            device_used = "cpu"
        else:
            train_fn = train_lightgbm if model_type == "lightgbm" else train_xgboost
            result = train_fn(train_df, target_col=target_col, use_gpu=use_gpu, splits=splits)
            test_metrics = _evaluate_on_test(
                result["model"], model_type, test_df, result["feature_cols"], target_col, is_continuous,
            )
            device_used = result.get("device_used", "cpu")

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "test_metrics": test_metrics,
            "device_used": device_used,
        })

    summary = _summarize_folds(fold_results, is_continuous)
    return {
        "model_type": model_type,
        "n_splits": n_splits,
        "folds": fold_results,
        "summary": summary,
    }


if __name__ == "__main__":
    print("cross_validation.py — run via scripts/cross_validate.py.")
