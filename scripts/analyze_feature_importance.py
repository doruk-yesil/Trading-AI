"""
analyze_feature_importance.py
------------------------------
Ranks a trained GBM's feature importance and suggests a reduced feature set
-- does NOT retrain or change anything automatically. This is a diagnostic /
planning tool: with 173 features (many collinear -- see
train_baseline_models.py's docstring on why Ridge is used over plain linear
regression), a lot of them likely contribute very little. Knowing which ones
matter helps decide what to prune in a future feature-engineering pass, and
helps sanity-check that the model is picking up genuinely informative signals
rather than noise.

Reads feature_importance straight from the saved model bundle (see
scripts/train_model.py, which now persists it for lightgbm/xgboost) -- no
retraining needed, this just re-reads what's already on disk.

Usage:
  python scripts/analyze_feature_importance.py
  python scripts/analyze_feature_importance.py --model-name xgboost
  python scripts/analyze_feature_importance.py --model-dir reports/models/20260702_120000
  python scripts/analyze_feature_importance.py --keep-pct 0.90   # more aggressive pruning suggestion
  python scripts/analyze_feature_importance.py --top-n 50        # suggest top 50 features instead
"""

from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.paths import ensure_dirs, MODELS_DIR, EXPERIMENTS_DIR
from src.models.model_io import load_model_bundle

DEFAULT_MODEL_NAME = "lightgbm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank feature importance and suggest a pruned feature set.")
    parser.add_argument("--model-dir", default=None,
                         help="reports/models/<timestamp> to load from. Defaults to the most recent run.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, choices=["lightgbm", "xgboost"],
                         help=f"Which model's importance to analyze (default: {DEFAULT_MODEL_NAME}). "
                              "linear/random_forest aren't supported here -- gain-based importance "
                              "isn't meaningful for a scaled linear model, and RF importance wasn't "
                              "wired into the saved bundle.")
    parser.add_argument("--keep-pct", type=float, default=0.95,
                         help="Suggest keeping the smallest set of top features that together account for "
                              "this fraction of total importance (default: 0.95).")
    parser.add_argument("--top-n", type=int, default=None,
                         help="Instead of --keep-pct, suggest exactly the top N features.")
    return parser.parse_args()


def find_latest_model_dir() -> Path:
    runs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No trained models found in {MODELS_DIR}. Run scripts/train_model.py first.")
    return runs[-1]


def main() -> None:
    args = parse_args()
    ensure_dirs()

    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model_dir()
    model_path = model_dir / f"{args.model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No {args.model_name}.joblib in {model_dir}")

    print("=" * 70)
    print("  Trading-AI | Feature Importance Analysis")
    print("=" * 70)
    print(f"  Model: {model_path}")

    bundle = load_model_bundle(model_path)
    if "feature_importance" not in bundle:
        raise ValueError(
            f"{model_path} has no feature_importance stored -- it was saved before "
            "scripts/train_model.py started persisting it, or is a model_type that doesn't "
            "carry importance (linear/random_forest). Retrain with scripts/train_model.py "
            f"--models {args.model_name} to get a bundle usable here."
        )

    fi = pd.Series(bundle["feature_importance"]).sort_values(ascending=False)
    total = fi.sum()
    cumulative_pct = (fi.cumsum() / total) if total > 0 else fi.cumsum()

    print(f"  Trained: {bundle['saved_at']}  |  {len(fi)} total features\n")
    print(f"  {'rank':<6} {'feature':<35} {'importance':>12} {'cum %':>8}")
    for rank, (feat, imp) in enumerate(fi.items(), start=1):
        print(f"  {rank:<6} {feat:<35} {imp:>12.1f} {cumulative_pct[feat]*100:>7.1f}%")

    if args.top_n is not None:
        suggested = fi.index[: args.top_n].tolist()
        rule = f"top_n={args.top_n}"
    else:
        n_keep = int((cumulative_pct <= args.keep_pct).sum()) + 1
        n_keep = min(n_keep, len(fi))
        suggested = fi.index[:n_keep].tolist()
        rule = f"keep_pct={args.keep_pct}"

    dropped = [f for f in fi.index if f not in suggested]

    print("\n" + "=" * 70)
    print(f"  SUGGESTION ({rule}): keep {len(suggested)} / {len(fi)} features, drop {len(dropped)}")
    print("=" * 70)
    if dropped:
        preview = ", ".join(dropped[:15])
        print(f"  Lowest-importance features that would be dropped: {preview}"
              f"{' ...' if len(dropped) > 15 else ''}")
    print("\n  Nothing was changed automatically -- this is a suggestion, not an action.")
    print("  To actually prune, remove the dropped columns in src/features/technical_indicators.py")
    print("  (or filter them out in get_feature_columns()) and retrain -- then compare IC before/after,")
    print("  fewer features isn't automatically better, it just trains faster and is less prone to")
    print("  overfitting on noise columns.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENTS_DIR / f"feature_importance_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model_path": str(model_path),
                "model_trained_at": bundle["saved_at"],
                "rule": rule,
                "ranked_features": [
                    {"rank": r, "feature": feat, "importance": float(imp), "cumulative_pct": round(float(cumulative_pct[feat]), 4)}
                    for r, (feat, imp) in enumerate(fi.items(), start=1)
                ],
                "suggested_keep": suggested,
                "suggested_drop": dropped,
            },
            f, indent=2,
        )
    print(f"\n  Full ranking + suggestion saved -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
