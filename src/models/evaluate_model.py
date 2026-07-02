"""
evaluate_model.py
-----------------
ML model evaluation — classification + regression metrics and prediction outputs.

Note: ML metrics (AUC, IC, precision, recall) tell you if the model is
statistically predictive. They do NOT tell you if it makes money.
Always follow up with backtest/metrics.py for financial performance.

High AUC/IC does NOT guarantee profitability.
Low AUC/IC does NOT mean the model is worthless (it may still generate alpha).

Regression note (target_type: "continuous"): raw R² on forward stock returns
is almost always tiny (a few % at best) and is NOT the right way to judge
whether the model has genuine skill -- financial returns are dominated by
noise, so R² near zero is normal even for a model that's actually useful for
ranking. The practitioner-standard metric here is the Information
Coefficient (Spearman rank correlation between predicted and realized
returns): it measures whether the model correctly RANKS symbols by expected
return, which is what a cross-sectional position-sizing strategy actually
needs -- see evaluate_regressor() below.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from sklearn.metrics import (
        roc_auc_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        average_precision_score,
        mean_absolute_error,
        mean_squared_error,
    )
    from scipy.stats import spearmanr
except ImportError:
    pass  # scikit-learn / scipy are hard requirements


def evaluate_classifier(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    threshold: float = 0.5,
    label: str = "test",
) -> dict:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels.
    y_pred_proba : pd.Series
        Predicted probability of label=1.
    threshold : float
        Decision threshold for class assignment.
    label : str
        Label for display (e.g. 'val', 'test').

    Returns
    -------
    dict
        auc, avg_precision, precision, recall, f1, confusion_matrix
    """
    # TODO: implement when model is trained
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc           = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    report        = classification_report(y_true, y_pred, output_dict=True)
    cm            = confusion_matrix(y_true, y_pred)

    results = {
        "label":         label,
        "auc":           round(auc, 4),
        "avg_precision": round(avg_precision, 4),
        "precision_1":   round(report.get("1", {}).get("precision", 0), 4),
        "recall_1":      round(report.get("1", {}).get("recall", 0), 4),
        "f1_1":          round(report.get("1", {}).get("f1-score", 0), 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n── {label} classification metrics ──")
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"  {k:<20} {v}")

    return results


def evaluate_regressor(
    y_true: pd.Series,
    y_pred: pd.Series,
    min_return_threshold: Optional[float] = None,
    label: str = "test",
) -> dict:
    """
    Compute evaluation metrics for a continuous forward-return target
    (configs/config.yaml -> target_type: "continuous").

    Parameters
    ----------
    y_true : pd.Series
        Realized forward return (decimal, e.g. 0.023 = +2.3%).
    y_pred : pd.Series
        Predicted forward return.
    min_return_threshold : float, optional
        If given, also reports directional/hit metrics as if predictions
        above this threshold were "acted on" -- lets you sanity-check a
        continuous model against the same economic bar the old binary
        target used, without needing to retrain a classifier.
    label : str
        Label for display (e.g. 'val', 'test').

    Returns
    -------
    dict
        mae, rmse, information_coefficient (Spearman rank correlation --
        THE primary metric here, not r2), directional_accuracy, and
        (if min_return_threshold given) precision/recall of the
        threshold-implied "trade" decision.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    ic, ic_pvalue = spearmanr(y_true, y_pred)

    direction_match = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(direction_match.mean())

    results = {
        "label": label,
        "mae": round(float(mae), 5),
        "rmse": round(float(rmse), 5),
        "information_coefficient": round(float(ic), 4),
        "ic_pvalue": round(float(ic_pvalue), 4),
        "directional_accuracy": round(directional_accuracy, 4),
    }

    if min_return_threshold is not None:
        actual_hit = (y_true > min_return_threshold).astype(int)
        predicted_hit = (y_pred > min_return_threshold).astype(int)
        if predicted_hit.sum() > 0:
            precision = (actual_hit & predicted_hit).sum() / predicted_hit.sum()
            results["threshold_precision"] = round(float(precision), 4)
        results["n_signals_above_threshold"] = int(predicted_hit.sum())

    print(f"\n── {label} regression metrics ──")
    for k, v in results.items():
        print(f"  {k:<28} {v}")
    print("  (information_coefficient is the primary quality signal here, not mae/rmse)")

    return results


def evaluate_by_period(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.Series,
    freq: str = "Q",
    is_continuous: bool = True,
    label: str = "test",
    min_period_size: int = 20,
) -> dict:
    """
    Breaks down performance by calendar period (quarter by default -- pass
    freq="Y" for year) instead of one aggregate number for the whole test
    window.

    Why this matters: a single aggregate IC over a 1-2 year test window can
    hide the fact that the model only worked during one calm stretch and was
    flat or negative everywhere else. Before trusting a model, you want to
    see whether its edge is broadly consistent across regimes (different
    volatility environments, bull/bear periods) or concentrated in one lucky
    quarter. This is the reporting counterpart to
    src/models/cross_validation.py's purged k-fold CV: that checks
    consistency across separately-TRAINED models; this checks consistency
    within a single already-trained model's test-set predictions over time.

    Periods with fewer than `min_period_size` rows are skipped -- too few
    observations for a rank correlation / AUC to mean anything.

    Returns
    -------
    dict
        {'periods': {period_label: metrics_dict, ...}, 'summary': {...}}
        summary holds mean/std/min/max of the primary metric (IC for
        continuous targets, AUC for binary) across scored periods.
    """
    period = pd.PeriodIndex(pd.to_datetime(dates), freq=freq).astype(str)
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "period": period})

    by_period = {}
    primary_values = []
    primary_name = "information_coefficient" if is_continuous else "auc"

    for period_label, group in frame.groupby("period"):
        if len(group) < min_period_size:
            continue
        if is_continuous:
            ic, _ = spearmanr(group["y_true"], group["y_pred"])
            direction_match = np.sign(group["y_true"]) == np.sign(group["y_pred"])
            metrics = {
                "n": int(len(group)),
                "information_coefficient": round(float(ic), 4) if not np.isnan(ic) else None,
                "directional_accuracy": round(float(direction_match.mean()), 4),
            }
            primary_val = metrics["information_coefficient"]
        else:
            try:
                auc_val = roc_auc_score(group["y_true"], group["y_pred"])
            except ValueError:
                auc_val = None  # single-class period -- AUC undefined
            metrics = {"n": int(len(group)), "auc": round(float(auc_val), 4) if auc_val is not None else None}
            primary_val = metrics["auc"]

        by_period[period_label] = metrics
        if primary_val is not None:
            primary_values.append(primary_val)

    summary = {
        "n_periods_scored": len(primary_values),
        f"{primary_name}_mean": round(float(np.mean(primary_values)), 4) if primary_values else None,
        f"{primary_name}_std": round(float(np.std(primary_values)), 4) if primary_values else None,
        f"{primary_name}_min": round(float(np.min(primary_values)), 4) if primary_values else None,
        f"{primary_name}_max": round(float(np.max(primary_values)), 4) if primary_values else None,
    }

    print(f"\n── {label} by-period ({freq}) breakdown ──")
    print(f"  {'period':<10} {'n':>8} {primary_name:>12}")
    for period_label, m in by_period.items():
        val = m.get(primary_name)
        print(f"  {period_label:<10} {m['n']:>8} {val if val is not None else 'n/a':>12}")
    print(f"  -- {primary_name} across periods: mean={summary[f'{primary_name}_mean']} "
          f"std={summary[f'{primary_name}_std']} min={summary[f'{primary_name}_min']} "
          f"max={summary[f'{primary_name}_max']} ({summary['n_periods_scored']} periods scored)")

    return {"periods": by_period, "summary": summary}


def get_regression_predictions(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    min_return_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run model inference for a regression model and return predicted returns
    + an implied trade signal (predicted return > min_return_threshold).

    Returns
    -------
    pd.DataFrame
        Columns: 'predicted_return', 'signal' (1 if above threshold else 0)
    """
    from src.utils.config import cfg

    X = df[feature_cols].fillna(0)
    predicted_return = model.predict(X)

    threshold = (
        min_return_threshold if min_return_threshold is not None
        else cfg["min_return_threshold"]
    )

    return pd.DataFrame(
        {
            "predicted_return": predicted_return,
            "signal": (predicted_return > threshold).astype(int),
        },
        index=df.index,
    )


def get_predictions(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run model inference and return a DataFrame with probabilities and signals.

    Returns
    -------
    pd.DataFrame
        Columns: 'prob_1', 'signal' (at 0.5 threshold)
    """
    # TODO: implement when model is trained
    X = df[feature_cols].fillna(0)

    # LightGBM vs sklearn/XGBoost predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X)  # LightGBM raw predict returns probabilities

    return pd.DataFrame(
        {"prob_1": proba, "signal": (proba >= 0.5).astype(int)},
        index=df.index,
    )


if __name__ == "__main__":
    print("evaluate_model.py — import evaluate_classifier() to use.")
