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
