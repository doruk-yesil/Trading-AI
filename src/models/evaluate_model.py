"""
evaluate_model.py
-----------------
ML model evaluation — classification metrics and prediction outputs.

Note: ML classification metrics (AUC, precision, recall) tell you if the model
is statistically predictive. They do NOT tell you if it makes money.
Always follow up with backtest/metrics.py for financial performance.

High AUC does NOT guarantee profitability.
Low AUC does NOT mean the model is worthless (it may still generate alpha).
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
    )
except ImportError:
    pass  # scikit-learn is a hard requirement


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
