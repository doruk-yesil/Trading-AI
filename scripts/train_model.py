"""
train_model.py
---------------
Trains and compares the baseline models + first 2 "real" models
(LightGBM, XGBoost) on the pooled cross-sectional panel (all symbols in
data/processed/ stacked together, split by calendar date -- not per symbol).

Why pooled instead of one model per symbol: with the S&P 500 universe this
gives roughly 1-1.5M pooled row-observations instead of ~2-3K per single
symbol -- comfortably enough for LightGBM/XGBoost, and lets every symbol
benefit from patterns learned across the whole cross-section rather than
overfitting to its own short history.

Models trained (in order):
  1. linear         -- Ridge (continuous) / LogisticRegression (binary)
  2. random_forest   -- RandomForestRegressor / RandomForestClassifier
  3. lightgbm        -- primary model candidate
  4. xgboost         -- primary model candidate

1-2 are the performance floor (see train_baseline_models.py). 3-4 must beat
them, or something is wrong -- that comparison is the whole point of this
script, not just "run the fancy model in isolation".

A 5th "ensemble" row is added automatically whenever 2+ models are trained in
the same run: the plain average of their test-set predictions. Not a
separately persisted model -- it's a derived combination of whichever models
were trained this run (see the "members" field in the saved JSON). Averaging
tends to reduce variance versus any single model when the underlying models
make different kinds of errors.

Evaluation: dispatches to evaluate_classifier() or evaluate_regressor() based
on configs/config.yaml -> target_type. For target_type: "continuous" (the
current default), the Information Coefficient (Spearman rank correlation) is
the primary quality signal -- NOT raw R²/MAE. See src/models/evaluate_model.py.
Every model's test-set performance is also broken down by calendar quarter
(evaluate_by_period) -- a good aggregate IC that's concentrated in one quarter
is a very different result from one that holds up consistently.

Outputs (both saved to disk, not just printed):
  reports/experiments/train_comparison_<timestamp>.json -- val + test metrics
      (aggregate and by-period) for every model + the ensemble, plus the path
      to each model's saved file.
  reports/models/<timestamp>/{linear,random_forest,lightgbm,xgboost}.joblib
      -- the actual trained model objects (joblib -- works uniformly for
      sklearn Pipelines, LightGBM Boosters, and XGBoost's sklearn wrappers).

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --symbols AAPL MSFT SPY   # quick test on a few symbols
  python scripts/train_model.py --models linear lightgbm  # skip random_forest/xgboost
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
from src.utils.paths import ensure_dirs, EXPERIMENTS_DIR, MODELS_DIR
from src.utils.universe import resolve_universe
from src.data.load_data import load_featured_universe
from src.models.train_baseline_models import train_and_evaluate, prepare_features
from src.models.train_xgboost import train_lightgbm, train_xgboost
from src.models.model_io import save_model_bundle

ALL_MODELS = ["linear", "random_forest", "lightgbm", "xgboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare baseline + GBM models.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe for a quick test run.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                         help="Which models to train (default: all 4).")
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU training for LightGBM/XGBoost (default: GPU, since Doruk has an RTX 3080). "
             "Each falls back to CPU automatically anyway if GPU init fails.",
    )
    return parser.parse_args()


def load_pooled_panel(symbols) -> pd.DataFrame:
    """Load + concatenate every symbol's featured parquet into one panel."""
    panel = load_featured_universe(symbols=symbols, concat=True)
    if isinstance(panel, dict):
        # concat=True normally guarantees a DataFrame; dict means nothing loaded.
        raise RuntimeError(
            "No featured data found. Run scripts/build_features.py first "
            "(see DATA_COMMANDS.md)."
        )
    return panel


def summarize_metrics(model_name: str, result: dict) -> dict:
    return {
        "model": model_name,
        "val": result.get("val_metrics", {}),
        "test": result.get("test_metrics", {}),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()

    symbols = resolve_universe(override_symbols=args.symbols)
    print("=" * 70)
    print("  Trading-AI | Model Training & Comparison")
    print("=" * 70)
    print(f"  Universe:    {len(symbols)} symbols")
    print(f"  target_type: {cfg.get('target_type')}")
    print(f"  horizon:     {cfg.get('prediction_horizon')} trading days")
    print(f"  Models:      {args.models}")
    print(f"  GPU:         {'requested (cuda -> gpu -> cpu fallback)' if not args.cpu_only else 'disabled (--cpu-only)'}")
    print()

    print("Loading pooled panel from data/processed/ ...")
    panel = load_pooled_panel(symbols)
    print(f"  Loaded panel: {panel.shape[0]} rows, {panel.shape[1]} columns\n")

    is_continuous = cfg.get("target_type", "binary") == "continuous"
    all_results = {}

    if "linear" in args.models:
        print("Training linear baseline...")
        all_results["linear"] = train_and_evaluate(panel, model_type="linear")

    if "random_forest" in args.models:
        print("Training random forest baseline...")
        all_results["random_forest"] = train_and_evaluate(panel, model_type="random_forest")

    use_gpu = not args.cpu_only

    if "lightgbm" in args.models:
        print("Training LightGBM...")
        all_results["lightgbm"] = train_lightgbm(panel, use_gpu=use_gpu)

    if "xgboost" in args.models:
        print("Training XGBoost...")
        all_results["xgboost"] = train_xgboost(panel, use_gpu=use_gpu)

    # ── Compute comparison metrics (val + test) for every model + persist the
    #    trained model objects themselves, so results AND the models that
    #    produced them can both be inspected later (not just left in memory
    #    and discarded when the script exits). ──
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    from src.models.evaluate_model import evaluate_classifier, evaluate_regressor, evaluate_by_period

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_model_dir = MODELS_DIR / timestamp
    run_model_dir.mkdir(parents=True, exist_ok=True)

    def predict_for_comparison(name, model, X):
        """Same dispatch as scripts/predict.py -- each library exposes
        predict differently, and this needs raw predictions (not just the
        aggregate metric) to also run the by-period breakdown below."""
        if name == "lightgbm":
            return model.predict(X)
        if name == "xgboost":
            return model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]
        return model.predict(X) if is_continuous else model.predict_proba(X)[:, 1]

    comparison = {}
    test_preds_by_model: dict[str, pd.Series] = {}
    test_df_ref, target_ref = None, None

    for name, result in all_results.items():
        feature_cols = result["feature_cols"]
        model = result["model"]
        test_df = result["test_df"]

        # Test-set metrics (GBMs compute fresh; baselines already have theirs
        # from train_and_evaluate -- reuse rather than recompute the
        # aggregate, but still need fresh predictions for the by-period
        # breakdown, which train_and_evaluate doesn't return).
        if len(test_df) == 0:
            test_metrics, by_period = {}, {}
        else:
            X_test = prepare_features(test_df, feature_cols)
            preds = predict_for_comparison(name, model, X_test)
            test_preds_by_model[name] = pd.Series(preds, index=test_df.index)
            if test_df_ref is None:
                test_df_ref = test_df
                target_ref = test_df["target"]

            if name in ("lightgbm", "xgboost"):
                if is_continuous:
                    test_metrics = evaluate_regressor(
                        test_df["target"], preds,
                        min_return_threshold=cfg["min_return_threshold"], label=f"{name}/test",
                    )
                else:
                    test_metrics = evaluate_classifier(test_df["target"], preds, label=f"{name}/test")
            else:
                test_metrics = result.get("test_metrics", {})

            by_period = evaluate_by_period(
                test_df["target"], preds, test_df["date"],
                freq="Q", is_continuous=is_continuous, label=f"{name}/test",
            )

        # GBMs use val only for early stopping, not a separate reported metric here.
        val_metrics = {} if name in ("lightgbm", "xgboost") else result.get("val_metrics", {})

        # Persist model + exact feature_cols list/order + metadata as one
        # self-contained bundle -- see src/models/model_io.py for why this
        # matters (predict.py must never guess the feature list). GBMs also
        # carry their feature_importance along -- scripts/analyze_feature_importance.py
        # reads it straight from the saved bundle instead of needing a retrain.
        model_path = run_model_dir / f"{name}.joblib"
        extra = None
        if "feature_importance" in result:
            extra = {"feature_importance": result["feature_importance"].to_dict()}
        save_model_bundle(
            model, feature_cols, model_path,
            model_type=name, device_used=result.get("device_used", "cpu"), extra=extra,
        )

        comparison[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "by_period": by_period,
            "model_path": str(model_path),
            "n_features": len(feature_cols),
            "device_used": result.get("device_used", "cpu"),
        }

    # ── Ensemble: simple average of predictions across every model trained
    #    this run (only meaningful with >= 2 models -- see --models to
    #    control which ones participate). Averaging tends to reduce variance
    #    versus any single model, especially when the underlying models make
    #    different kinds of errors (linear/RF vs GBM). Not a separately
    #    persisted model -- it's a derived combination of the ones already
    #    saved above, recomputable at inference time by loading + averaging
    #    the member models directly. ──
    if len(test_preds_by_model) >= 2:
        pred_frame = pd.concat(test_preds_by_model, axis=1).dropna()
        if len(pred_frame) > 0:
            ensemble_preds = pred_frame.mean(axis=1)
            y_true_ens = target_ref.loc[pred_frame.index]
            dates_ens = test_df_ref.loc[pred_frame.index, "date"]

            print(f"\nEvaluating ensemble ({' + '.join(test_preds_by_model.keys())})...")
            if is_continuous:
                ensemble_test_metrics = evaluate_regressor(
                    y_true_ens, ensemble_preds,
                    min_return_threshold=cfg["min_return_threshold"], label="ensemble/test",
                )
            else:
                ensemble_test_metrics = evaluate_classifier(y_true_ens, ensemble_preds, label="ensemble/test")

            ensemble_by_period = evaluate_by_period(
                y_true_ens, ensemble_preds, dates_ens,
                freq="Q", is_continuous=is_continuous, label="ensemble/test",
            )

            comparison["ensemble"] = {
                "val": {},
                "test": ensemble_test_metrics,
                "by_period": ensemble_by_period,
                "model_path": None,
                "n_features": None,
                "device_used": None,
                "members": list(test_preds_by_model.keys()),
            }

    print("\n  Summary (test set):")
    if is_continuous:
        print(f"  {'model':<15} {'IC':>8} {'dir_acc':>10} {'mae':>10}")
        for name, m in comparison.items():
            t = m["test"]
            print(f"  {name:<15} {t.get('information_coefficient', float('nan')):>8.4f} "
                  f"{t.get('directional_accuracy', float('nan')):>10.4f} "
                  f"{t.get('mae', float('nan')):>10.5f}")
    else:
        print(f"  {'model':<15} {'AUC':>8}")
        for name, m in comparison.items():
            print(f"  {name:<15} {m['test'].get('auc', float('nan')):>8.4f}")

    # ── Save experiment (metrics + where to find the actual model files) ──
    out_path = EXPERIMENTS_DIR / f"train_comparison_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "target_type": cfg.get("target_type"),
                "prediction_horizon": cfg.get("prediction_horizon"),
                "universe_size": len(symbols),
                "panel_rows": int(panel.shape[0]),
                "model_dir": str(run_model_dir),
                "results": comparison,
            },
            f, indent=2, default=str,
        )
    print(f"\n  Experiment metrics saved -> {out_path}")
    print(f"  Trained models saved     -> {run_model_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
