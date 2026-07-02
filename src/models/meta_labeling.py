"""
meta_labeling.py
-----------------
Meta-labeling (Marcos Lopez de Prado, "Advances in Financial Machine
Learning", ch. 3): a SECOND model that doesn't predict the return itself --
it predicts whether the PRIMARY model's signal, on the instances where the
primary model actually signals a trade, is likely to be correct. The primary
model answers "what's the predicted return, and which direction" (the
"side"); the meta-model answers "should we actually act on this specific
signal, or sit it out" (the "bet size"/filter decision).

Why this is useful here: the primary model (lightgbm/xgboost, continuous
target) is optimized to rank ALL symbols well on average (Information
Coefficient). It is NOT optimized to know when it's LIKELY WRONG on a
specific signal. A meta-model trained specifically on "was the primary
right, on the subset of times it signaled a trade" can improve precision
(fewer bad trades acted on) at the cost of recall (fewer trades total) --
a trade-off that's usually worth it for a strategy that only needs to be
selectively right, not comprehensively right about every symbol every day.

Leakage note: the meta-model is trained on the PRIMARY model's predictions
on data the primary model did NOT train on (its validation split) -- using
in-sample (train-set) primary predictions to train the meta-model would be
badly overfit, since the primary model has already memorized its own
training set to some degree.

Not using compute_uniqueness_weights() here (unlike the primary models):
that function assumes a dense, ~1-row-per-trading-day panel per symbol, sized
by ROW POSITION. Meta-training only uses the sparse SUBSET of rows where the
primary model actually signaled a trade -- treating that sparse, irregular
subset as if it were consecutive daily rows would overstate overlap between
instances that may be months apart. Lopez de Prado's own uniqueness weighting
is actually designed for exactly this kind of sparse, event-based sample --
a proper implementation would need real calendar-day overlap on the sparse
event set, which is a larger follow-up, not done here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.utils.config import cfg


def build_meta_labels(
    y_true: pd.Series,
    y_pred: pd.Series,
    min_return_threshold: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Derives the primary model's "acted" mask and the meta-label from its
    predictions and the realized outcome.

    acted[i]      = True if the primary model's |predicted return| clears
                    the minimum-edge threshold, i.e. it would have signaled
                    a trade on row i at all.
    meta_label[i] = 1 if that signal's DIRECTION was correct (sign of
                    predicted return matches sign of realized return), 0
                    otherwise. Only meaningful where acted[i] is True --
                    the meta-model is trained/evaluated on the acted subset
                    only, since that's the only population it needs to make
                    a decision about at inference time (there's no "act or
                    not" decision to filter when the primary didn't signal
                    anything in the first place).

    Returns
    -------
    (acted, meta_label) -- both pd.Series aligned to y_true's index.
    """
    threshold = (
        min_return_threshold if min_return_threshold is not None
        else cfg["min_return_threshold"]
    )
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    acted = y_pred.abs() > threshold
    meta_label = (np.sign(y_true) == np.sign(y_pred)).astype(int)
    return acted, meta_label


def build_meta_pipeline() -> Pipeline:
    """
    Logistic regression -- deliberately simple. Meta-labeling's job is a
    focused binary filter, not another full model; a simpler, more stable
    classifier is standard practice here (and trains fast enough to iterate
    on quickly, unlike a full GBM retrain).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=cfg["model"]["random_seed"], class_weight="balanced")),
    ])


def train_meta_model(
    meta_X: pd.DataFrame,
    meta_y: pd.Series,
) -> Pipeline:
    """
    Fits the meta-model. class_weight="balanced" (see build_meta_pipeline)
    matters because "primary was right" vs "primary was wrong" on its own
    signaled trades is often not a 50/50 split.
    """
    pipeline = build_meta_pipeline()
    pipeline.fit(meta_X, meta_y)
    return pipeline


def evaluate_meta_filter(
    meta_model: Pipeline,
    meta_X: pd.DataFrame,
    meta_y: pd.Series,
    decision_threshold: float = 0.5,
    label: str = "test",
) -> dict:
    """
    Compares "act on every primary signal" (baseline precision = meta_y's
    own hit rate) against "act only where the meta-model says to" (filtered
    precision) on the primary model's acted-subset for this split.

    Returns
    -------
    dict
        baseline_precision  -- hit rate if you acted on every primary signal
        baseline_n          -- number of primary signals in this split
        filtered_precision  -- hit rate on the subset the meta-model approved
        filtered_n          -- how many signals survive the filter
        filtered_recall     -- fraction of GOOD primary signals the filter kept
        meta_auc            -- meta-model's own AUC at separating right/wrong signals
    """
    baseline_precision = float(meta_y.mean())
    baseline_n = int(len(meta_y))

    proba = meta_model.predict_proba(meta_X)[:, 1]
    approved = proba >= decision_threshold

    filtered_n = int(approved.sum())
    if filtered_n > 0:
        filtered_precision = float(meta_y.to_numpy()[approved].mean())
    else:
        filtered_precision = None

    filtered_recall = float(recall_score(meta_y, approved)) if meta_y.nunique() > 1 else None
    try:
        meta_auc = float(roc_auc_score(meta_y, proba))
    except ValueError:
        meta_auc = None  # single-class split -- AUC undefined

    results = {
        "label": label,
        "baseline_precision": round(baseline_precision, 4),
        "baseline_n": baseline_n,
        "filtered_precision": round(filtered_precision, 4) if filtered_precision is not None else None,
        "filtered_n": filtered_n,
        "filtered_recall": round(filtered_recall, 4) if filtered_recall is not None else None,
        "meta_auc": round(meta_auc, 4) if meta_auc is not None else None,
    }

    print(f"\n── {label} meta-labeling filter ──")
    print(f"  Unfiltered (act on every primary signal): precision={results['baseline_precision']} "
          f"over {results['baseline_n']} signals")
    print(f"  Meta-filtered (act only where meta-model approves): precision={results['filtered_precision']} "
          f"over {results['filtered_n']} signals ({results['filtered_n']}/{results['baseline_n']} kept)")
    print(f"  meta_auc={results['meta_auc']}  "
          "(higher = meta-model better separates correct vs incorrect primary signals)")
    print("  Precision up + fewer signals is the expected, useful trade-off -- meta-labeling")
    print("  trades recall for precision, it does not add new information the primary lacked.")

    return results


if __name__ == "__main__":
    print("meta_labeling.py — run via scripts/train_meta_model.py.")
