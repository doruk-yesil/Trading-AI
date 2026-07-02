"""
predict.py
----------
Loads an ALREADY-TRAINED model (see scripts/train_model.py) and runs
inference on today's data -- no training happens here. This is the "give it
new data, get a real prediction" step: for each symbol, takes its most
recent row (the one whose target is still unknown -- we haven't lived
through the next prediction_horizon days yet) and predicts the forward
return.

Why this is a separate script from training: training is slow (minutes) and
should only happen when you deliberately want to update what the model has
learned (e.g. after scripts/daily_refresh.py brings in a new day, on some
cadence -- weekly, monthly, whenever). Predicting is fast (milliseconds) and
should happen every time you actually want today's signal, without paying
the training cost again. Run this daily; retrain occasionally.

Safety: refuses to run if the current data doesn't have every feature column
the model was trained on (see src/models/model_io.py validate_feature_match)
-- rather than silently predicting on a mismatched matrix.

Usage:
  python scripts/predict.py                          # latest trained run, lightgbm
  python scripts/predict.py --model-name xgboost
  python scripts/predict.py --model-dir reports/models/20260702_120000
  python scripts/predict.py --symbols AAPL MSFT SPY
  python scripts/predict.py --top-n 30
"""

from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.config import cfg
from src.utils.paths import ensure_dirs, MODELS_DIR, PREDICTIONS_DIR
from src.utils.universe import resolve_universe
from src.data.load_data import load_featured_universe
from src.models.train_baseline_models import prepare_features
from src.models.model_io import load_model_bundle, validate_feature_match

DEFAULT_MODEL_NAME = "lightgbm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with an already-trained model.")
    parser.add_argument("--model-dir", default=None,
                         help="reports/models/<timestamp> to load from. Defaults to the most recent run.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                         choices=["linear", "random_forest", "lightgbm", "xgboost"],
                         help=f"Which model within the run to use (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe.")
    parser.add_argument("--top-n", type=int, default=20, help="How many top-ranked symbols to print.")
    return parser.parse_args()


def find_latest_model_dir() -> Path:
    runs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_DIR}. Run scripts/train_model.py first."
        )
    return runs[-1]


def predict_with_model(model, model_type: str, X: pd.DataFrame, is_continuous: bool) -> np.ndarray:
    """Dispatch .predict()/.predict_proba() correctly per model type -- each
    library exposes this differently (see docstring in src/models/model_io.py
    for why the bundle stores model_type explicitly rather than guessing)."""
    if model_type == "lightgbm":
        # lgb.Booster.predict() already returns probabilities for binary
        # objective and raw values for regression -- no separate predict_proba.
        return model.predict(X)
    if model_type == "xgboost":
        return model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]
    # linear / random_forest -- sklearn Pipeline
    return model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]


def get_latest_row_per_symbol(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, take the single most recent row -- this is "today's"
    (or whatever the latest downloaded day is) unlabeled observation, i.e.
    exactly the row whose target the model should predict rather than one
    it could have seen a ground-truth answer for during training.
    """
    idx = panel.groupby("symbol")["date"].idxmax()
    return panel.loc[idx].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model_dir()
    model_path = model_dir / f"{args.model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No {args.model_name}.joblib in {model_dir}")

    print("=" * 70)
    print("  Trading-AI | Predict (inference only, no training)")
    print("=" * 70)
    print(f"  Model:  {model_path}")

    bundle = load_model_bundle(model_path)
    print(f"  Trained: {bundle['saved_at']}  (target_type={bundle['target_type']}, "
          f"horizon={bundle['prediction_horizon']}d, device={bundle.get('device_used')})")

    is_continuous = bundle["target_type"] == "continuous"
    feature_cols = bundle["feature_cols"]
    model = bundle["model"]

    symbols = resolve_universe(override_symbols=args.symbols)
    print(f"  Universe: {len(symbols)} symbols\n")

    print("Loading latest featured data...")
    panel = load_featured_universe(symbols=symbols, concat=True)
    if isinstance(panel, dict):
        raise RuntimeError("No featured data found. Run scripts/build_features.py first.")

    validate_feature_match(bundle, list(panel.columns))

    latest = get_latest_row_per_symbol(panel)
    print(f"  Predicting on {len(latest)} symbols' most recent row "
          f"(as of {latest['date'].min()} -> {latest['date'].max()})\n")

    X = prepare_features(latest, feature_cols)
    predictions = predict_with_model(model, bundle["model_type"], X, is_continuous)

    result = pd.DataFrame({
        "symbol": latest["symbol"].values,
        "date": latest["date"].values,
        "predicted_return" if is_continuous else "predicted_probability": predictions,
    })
    pred_col = "predicted_return" if is_continuous else "predicted_probability"
    result = result.sort_values(pred_col, ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    if is_continuous:
        threshold = cfg["min_return_threshold"]
        result["above_threshold"] = result[pred_col] > threshold
        n_signals = int(result["above_threshold"].sum())
        print(f"  {n_signals} / {len(result)} symbols predicted above the "
              f"{threshold * 100:.2f}% minimum-edge threshold\n")

    print(f"  Top {args.top_n} ranked symbols:")
    print(result.head(args.top_n).to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PREDICTIONS_DIR / f"predictions_{timestamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    meta_path = PREDICTIONS_DIR / f"predictions_{timestamp}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_path": str(model_path),
            "model_type": bundle["model_type"],
            "model_trained_at": bundle["saved_at"],
            "target_type": bundle["target_type"],
            "prediction_horizon": bundle["prediction_horizon"],
            "predicted_at": timestamp,
            "universe_size": len(symbols),
        }, f, indent=2, default=str)

    print(f"\n  Predictions saved -> {out_path}")
    print(f"  Metadata saved    -> {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
