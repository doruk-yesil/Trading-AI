"""
model_io.py
-----------
Shared save/load logic for trained model artifacts.

The whole point of this module: a saved model is USELESS on its own for
inference unless you also know exactly which feature columns (and in what
order) it was trained on. Recomputing that list at inference time from
scratch is dangerous -- if feature engineering changes between training and
inference (a column added/removed/renamed in technical_indicators.py), the
recomputed list would silently differ from what the model actually learned,
and feeding a shape-mismatched or reordered matrix into predict() either
raises a confusing error or -- worse -- silently produces wrong numbers.

So every saved model is a self-contained bundle: the model object PLUS the
exact feature_cols list PLUS the metadata needed to sanity-check it's being
used the way it was trained (target_type, prediction_horizon). Training
(scripts/train_model.py) and inference (scripts/predict.py) both go through
this module so the bundle format only has to be right in one place.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib

from src.utils.config import cfg

BUNDLE_SCHEMA_VERSION = 1


def save_model_bundle(
    model: Any,
    feature_cols: list[str],
    path: Path,
    model_type: str,
    device_used: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a trained model together with everything needed to use it correctly
    later: exact feature column list/order, target_type, prediction_horizon.

    Returns the path written to.
    """
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model": model,
        "feature_cols": list(feature_cols),
        "model_type": model_type,
        "target_type": cfg.get("target_type"),
        "prediction_horizon": cfg.get("prediction_horizon"),
        "device_used": device_used,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        bundle.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: Path) -> dict:
    """
    Load a bundle saved by save_model_bundle(). Raises a clear error (not a
    confusing pickle/attribute error) if the file predates this bundle
    format -- i.e. was saved before feature_cols persistence was added.
    """
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "feature_cols" not in bundle:
        raise ValueError(
            f"{path} was not saved by save_model_bundle() (missing feature_cols) -- "
            "it predates this bundle format. Retrain with scripts/train_model.py to "
            "get a bundle usable by scripts/predict.py."
        )
    return bundle


def validate_feature_match(bundle: dict, available_columns: list[str]) -> None:
    """
    Raise a clear error if the current data is missing any column the model
    was trained on -- instead of silently predicting on a wrong/reordered
    matrix. Call this before running inference.
    """
    missing = [c for c in bundle["feature_cols"] if c not in available_columns]
    if missing:
        raise ValueError(
            f"Model '{bundle['model_type']}' (saved {bundle['saved_at']}) was trained on "
            f"{len(missing)} feature column(s) that are missing from the current data: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            "This means feature engineering has changed since this model was trained. "
            "Retrain with scripts/train_model.py before predicting."
        )
