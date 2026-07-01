"""
main.py
-------
Entry point for the Trading-AI pipeline.

Stages:
  1. Load raw data
  2. Validate data quality
  3. Preprocess (clean + align)
  4. Build features
  5. Build targets
  6. Run baseline strategies
  7. Train ML model
  8. Run backtest
  9. Save experiment results

Run:
  python src/main.py --mode all
  python src/main.py --mode backtest --model lightgbm
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path so `src` is importable when this file is run
# directly (e.g. `python src/main.py`), matching how scripts/*.py already
# bootstrap themselves. Without this, Python sets sys.path[0] to `src/`
# itself (not the project root), so `from src.utils...` fails with
# "ModuleNotFoundError: No module named 'src'".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dirs, EXPERIMENTS_DIR
from src.utils.config import cfg
from src.utils.universe import resolve_universe


def parse_args():
    parser = argparse.ArgumentParser(description="Trading-AI pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "data", "features", "train", "backtest"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline_ma", "baseline_rsi", "logistic", "random_forest", "lightgbm", "xgboost"],
        default="lightgbm",
        help="Model to train and evaluate.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols from config (e.g. --symbols AAPL MSFT SPY).",
    )
    return parser.parse_args()


def save_experiment(results: dict, experiment_name: str) -> Path:
    """Save experiment results as JSON with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = EXPERIMENTS_DIR / f"{experiment_name}_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Experiment saved → {out_path}")
    return out_path


def run_pipeline(args):
    """Main pipeline — each step is a TODO until data is available."""

    ensure_dirs()

    symbols = resolve_universe(override_symbols=args.symbols)
    print(f"\nTrading-AI Pipeline")
    print(f"  Symbols:    {symbols}")
    print(f"  Date range: {cfg['start_date']} → {cfg['end_date']}")
    print(f"  Model:      {args.model}")
    print(f"  Mode:       {args.mode}")
    print()

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    if args.mode in ("all", "data", "features", "train", "backtest"):
        print("Stage 1: Loading data...")
        from src.data.load_data import load_universe
        universe = load_universe(symbols=symbols)
        if not universe:
            print("  No data loaded. Add raw data files to data/raw/ and re-run.")
            print("  See scripts/data_format_guide.md for the expected format.")
            return
        print(f"  Loaded {len(universe)} symbols.\n")

    # ── Stage 2: Validate ─────────────────────────────────────────────────────
    if args.mode in ("all", "data"):
        print("Stage 2: Validating data quality...")
        from src.data.validate_data import validate_universe
        issues = validate_universe(universe)
        n_issues = sum(len(v) for v in issues.values())
        print(f"  Total issues found: {n_issues}\n")

    # ── Stage 3: Preprocess ───────────────────────────────────────────────────
    if args.mode in ("all", "data", "features", "train", "backtest"):
        print("Stage 3: Preprocessing...")
        from src.data.preprocess import run_preprocessing
        universe = run_preprocessing(universe)
        print()

    # ── Stage 4: Features + Target ────────────────────────────────────────────
    if args.mode in ("all", "features", "train", "backtest"):
        print("Stage 4: Building features and targets...")
        from src.features.technical_indicators import build_all_features
        from src.features.target_builder import build_target
        featured = {}
        for sym, df in universe.items():
            df = build_all_features(df)
            df = build_target(df)
            featured[sym] = df
            print(f"  {sym}: {len(df.columns)} columns, {df['target'].notna().sum()} labelled rows")
        print()

    # ── Stage 5: Train model ──────────────────────────────────────────────────
    if args.mode in ("all", "train", "backtest"):
        print(f"Stage 5: Training model ({args.model})...")
        # TODO: implement per model type
        print("  (not yet implemented — awaiting data)\n")

    # ── Stage 6: Backtest ─────────────────────────────────────────────────────
    if args.mode in ("all", "backtest"):
        print("Stage 6: Running backtest...")
        # TODO: implement
        print("  (not yet implemented — awaiting model training)\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
