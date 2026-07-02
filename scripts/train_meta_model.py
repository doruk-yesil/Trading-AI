"""
train_meta_model.py
--------------------
Trains a meta-labeling filter (src/models/meta_labeling.py) on top of an
already-trained primary model (scripts/train_model.py). The primary model
predicts a return / direction; this secondary model predicts whether that
specific signal is likely to be correct, and is used to filter which of the
primary's signals actually get acted on.

Leakage-safe by construction:
  - Primary model was trained on train_df.
  - Meta-model is trained on the primary's predictions on val_df (data the
    primary never trained on) -- NOT on train_df, which would let the
    meta-model learn from an overfit primary.
  - Both the meta-model AND the final combined precision are then evaluated
    on test_df, untouched by either model until this point.

This assumes the featured panel on disk hasn't changed since the primary
model was trained (time_split() is deterministic given the same panel + the
same configs/config.yaml split settings, so it reproduces the exact same
train/val/test boundaries) -- if you've re-run scripts/build_features.py
since training the primary model, retrain the primary first.

Usage:
  python scripts/train_meta_model.py
  python scripts/train_meta_model.py --model-name xgboost
  python scripts/train_meta_model.py --model-dir reports/models/20260702_120000
  python scripts/train_meta_model.py --decision-threshold 0.6   # stricter filter
"""

from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import cfg
from src.utils.paths import ensure_dirs, MODELS_DIR, EXPERIMENTS_DIR
from src.utils.universe import resolve_universe
from src.data.load_data import load_featured_universe
from src.models.train_baseline_models import time_split, prepare_features
from src.models.model_io import load_model_bundle, validate_feature_match, save_model_bundle
from src.models.meta_labeling import build_meta_labels, train_meta_model, evaluate_meta_filter

DEFAULT_MODEL_NAME = "lightgbm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a meta-labeling filter on top of a saved primary model.")
    parser.add_argument("--model-dir", default=None,
                         help="reports/models/<timestamp> holding the primary model. Defaults to the latest run.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, choices=["linear", "random_forest", "lightgbm", "xgboost"],
                         help=f"Which primary model to build a filter on top of (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe.")
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                         help="Meta-model probability threshold to approve a primary signal (default: 0.5). "
                              "Raise for a stricter filter (higher precision, fewer trades).")
    return parser.parse_args()


def find_latest_model_dir() -> Path:
    runs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No trained models found in {MODELS_DIR}. Run scripts/train_model.py first.")
    return runs[-1]


def predict_with_primary(bundle: dict, X: pd.DataFrame) -> pd.Series:
    model, model_type = bundle["model"], bundle["model_type"]
    is_continuous = bundle["target_type"] == "continuous"
    if model_type == "lightgbm":
        preds = model.predict(X)
    elif model_type == "xgboost":
        preds = model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]
    return pd.Series(preds, index=X.index)


def build_meta_features(df: pd.DataFrame, feature_cols: list[str], primary_preds: pd.Series) -> pd.DataFrame:
    """feature_cols (same as primary) + the primary model's own prediction --
    lets the meta-model learn things like 'primary is unreliable when it
    predicts a large move' in addition to whatever the raw features show."""
    X = prepare_features(df, feature_cols).copy()
    X["primary_pred"] = primary_preds.reindex(X.index).fillna(0.0)
    return X


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if cfg.get("target_type", "binary") != "continuous":
        raise ValueError(
            "Meta-labeling as implemented here (build_meta_labels) assumes a continuous "
            "target (sign + threshold on predicted return). target_type is currently "
            f"'{cfg.get('target_type')}' -- not supported yet."
        )

    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model_dir()
    model_path = model_dir / f"{args.model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No {args.model_name}.joblib in {model_dir}")

    print("=" * 70)
    print("  Trading-AI | Meta-Labeling Filter Training")
    print("=" * 70)
    print(f"  Primary model: {model_path}")

    bundle = load_model_bundle(model_path)
    feature_cols = bundle["feature_cols"]
    print(f"  Trained: {bundle['saved_at']}  (horizon={bundle['prediction_horizon']}d)\n")

    symbols = resolve_universe(override_symbols=args.symbols)
    panel = load_featured_universe(symbols=symbols, concat=True)
    if isinstance(panel, dict):
        raise RuntimeError("No featured data found. Run scripts/build_features.py first.")
    validate_feature_match(bundle, list(panel.columns))

    df_clean = panel.dropna(subset=["target"])
    train_df, val_df, test_df = time_split(df_clean)
    print(f"  Reproduced split: train={len(train_df)} val={len(val_df)} test={len(test_df)}\n")

    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("val or test split is empty -- can't train/evaluate a meta-model without both.")

    threshold = cfg["min_return_threshold"]

    # ── Meta-training data: primary's predictions on VAL (data it never trained on) ──
    X_val = prepare_features(val_df, feature_cols)
    primary_val_preds = predict_with_primary(bundle, X_val)
    acted_val, meta_label_val = build_meta_labels(val_df["target"], primary_val_preds, threshold)

    # build_meta_labels() resets its input to a plain 0..n-1 index (see its
    # docstring) -- realign meta_X the same way before applying the boolean mask.
    meta_X_val_full = build_meta_features(val_df, feature_cols, primary_val_preds).reset_index(drop=True)
    meta_X_train = meta_X_val_full.loc[acted_val].copy()
    meta_y_train = meta_label_val.loc[acted_val].copy()

    print(f"  Primary signaled a trade on {int(acted_val.sum())} / {len(val_df)} val rows "
          f"({acted_val.mean()*100:.1f}%) -- meta-model trains on this subset.")

    if meta_X_train["primary_pred"].empty or meta_y_train.nunique() < 2:
        raise RuntimeError(
            "Not enough / not diverse enough val-set signals to train a meta-model "
            "(need both correct and incorrect primary signals in the acted subset)."
        )

    meta_model = train_meta_model(meta_X_train, meta_y_train)

    # ── Meta-evaluation: primary's predictions on TEST (untouched by both models) ──
    X_test = prepare_features(test_df, feature_cols)
    primary_test_preds = predict_with_primary(bundle, X_test)
    acted_test, meta_label_test = build_meta_labels(test_df["target"], primary_test_preds, threshold)

    meta_X_test_full = build_meta_features(test_df, feature_cols, primary_test_preds).reset_index(drop=True)
    meta_X_test = meta_X_test_full.loc[acted_test].copy()
    meta_y_test = meta_label_test.loc[acted_test].copy()

    print(f"  Primary signaled a trade on {int(acted_test.sum())} / {len(test_df)} test rows "
          f"({acted_test.mean()*100:.1f}%).")

    if len(meta_X_test) == 0:
        raise RuntimeError("Primary model produced zero signals on the test split -- nothing to evaluate the filter on.")

    filter_results = evaluate_meta_filter(
        meta_model, meta_X_test, meta_y_test,
        decision_threshold=args.decision_threshold, label="test",
    )

    # ── Persist ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_model_path = model_dir / f"meta_{args.model_name}.joblib"
    save_model_bundle(
        meta_model,
        feature_cols + ["primary_pred"],
        meta_model_path,
        model_type=f"meta_{args.model_name}",
        extra={
            "primary_model_path": str(model_path),
            "primary_model_type": bundle["model_type"],
            "decision_threshold": args.decision_threshold,
        },
    )
    print(f"\n  Meta-model saved -> {meta_model_path}")

    out_path = EXPERIMENTS_DIR / f"meta_labeling_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "primary_model_path": str(model_path),
                "primary_model_type": bundle["model_type"],
                "meta_model_path": str(meta_model_path),
                "decision_threshold": args.decision_threshold,
                "val_n_signals": int(acted_val.sum()),
                "val_n_total": len(val_df),
                "test_n_signals": int(acted_test.sum()),
                "test_n_total": len(test_df),
                "filter_results": filter_results,
            },
            f, indent=2, default=str,
        )
    print(f"  Results saved      -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
