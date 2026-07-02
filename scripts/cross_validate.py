"""
cross_validate.py
------------------
Runs purged k-fold cross-validation (src/models/cross_validation.py) across
multiple contiguous time blocks of the pooled panel, instead of relying on
a single train/val/test split. This is a robustness check, not a training
step: a good IC on one split can just mean that split happened to be an easy
period. This script trains N fresh copies of the model (one per fold) and
reports how consistent performance is across all of them.

Not part of the daily/weekly training flow (scripts/train_model.py still
produces the deployed model). Run this before trusting a model with real
money, and again whenever the feature set or target definition changes.

Usage:
  python scripts/cross_validate.py
  python scripts/cross_validate.py --models lightgbm xgboost
  python scripts/cross_validate.py --n-splits 5 --symbols AAPL MSFT SPY
"""

from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import cfg
from src.utils.paths import ensure_dirs, EXPERIMENTS_DIR
from src.utils.universe import resolve_universe
from src.data.load_data import load_featured_universe
from src.models.cross_validation import run_purged_cv

ALL_MODELS = ["linear", "random_forest", "lightgbm", "xgboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Purged k-fold cross-validation across multiple time periods.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe for a quick test run.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                         help="Which model types to cross-validate (default: all 4).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of purged folds (default: 5).")
    parser.add_argument("--cpu-only", action="store_true",
                         help="Force CPU for LightGBM/XGBoost (default: GPU with automatic CPU fallback).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    symbols = resolve_universe(override_symbols=args.symbols)
    print("=" * 70)
    print("  Trading-AI | Purged K-Fold Cross-Validation")
    print("=" * 70)
    print(f"  Universe: {len(symbols)} symbols")
    print(f"  Folds:    {args.n_splits}")
    print(f"  Models:   {args.models}\n")

    panel = load_featured_universe(symbols=symbols, concat=True)
    if isinstance(panel, dict):
        raise RuntimeError("No featured data found. Run scripts/build_features.py first (see DATA_COMMANDS.md).")
    print(f"  Loaded panel: {panel.shape[0]} rows, {panel.shape[1]} columns\n")

    is_continuous = cfg.get("target_type", "binary") == "continuous"
    use_gpu = not args.cpu_only

    all_results = {}
    for model_type in args.models:
        print(f"--- {model_type} ---")
        result = run_purged_cv(panel, model_type=model_type, n_splits=args.n_splits, use_gpu=use_gpu)
        all_results[model_type] = result
        s = result["summary"]
        if is_continuous:
            print(f"  IC: mean={s.get('ic_mean'):.4f} std={s.get('ic_std'):.4f}  "
                  f"dir_acc={s.get('directional_accuracy_mean'):.4f}  "
                  f"({s.get('n_scored_folds')}/{s.get('n_folds')} folds scored)\n"
                  if s.get("n_scored_folds") else f"  No scored folds.\n")
        else:
            print(f"  AUC: mean={s.get('auc_mean'):.4f} std={s.get('auc_std'):.4f}  "
                  f"({s.get('n_scored_folds')}/{s.get('n_folds')} folds scored)\n"
                  if s.get("n_scored_folds") else f"  No scored folds.\n")

    print("=" * 70)
    print("  SUMMARY (mean +/- std across folds, test-set only)")
    print("=" * 70)
    if is_continuous:
        print(f"  {'model':<15} {'IC mean':>10} {'IC std':>10} {'dir_acc':>10}")
        for name, r in all_results.items():
            s = r["summary"]
            print(f"  {name:<15} {s.get('ic_mean', float('nan')):>10.4f} "
                  f"{s.get('ic_std', float('nan')):>10.4f} "
                  f"{s.get('directional_accuracy_mean', float('nan')):>10.4f}")
    else:
        print(f"  {'model':<15} {'AUC mean':>10} {'AUC std':>10}")
        for name, r in all_results.items():
            s = r["summary"]
            print(f"  {name:<15} {s.get('auc_mean', float('nan')):>10.4f} {s.get('auc_std', float('nan')):>10.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENTS_DIR / f"cross_validation_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "target_type": cfg.get("target_type"),
                "prediction_horizon": cfg.get("prediction_horizon"),
                "n_splits": args.n_splits,
                "universe_size": len(symbols),
                "panel_rows": int(panel.shape[0]),
                "results": all_results,
            },
            f, indent=2, default=str,
        )
    print(f"\n  Cross-validation results saved -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
