"""
build_features.py
-----------------
One-time (or on-demand) script that turns raw OHLCV parquet files into
model-ready featured parquet files.

Run this ONCE after downloading data, and again whenever raw data is refreshed.
All models read from data/processed/ — never from data/raw/ directly.

Workflow
--------
  1. python scripts/download_yahoo_finance.py   ← download raw data
  2. python scripts/build_features.py           ← generate features + target
  3. python src/main.py --mode train            ← train model (reads processed/)

Output
------
  data/processed/{SYMBOL}_featured.parquet
    Columns: date, symbol, open, high, low, close, volume,
             <173 feature columns>,
             forward_return_5d, target

Usage
-----
  # All symbols from config
  python scripts/build_features.py

  # Specific symbols only
  python scripts/build_features.py --symbols AAPL MSFT SPY

  # Custom horizon / threshold
  python scripts/build_features.py --horizon 10 --threshold 0.01
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import cfg
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_dirs
from src.features.technical_indicators import generate_ohlcv_features
from src.features.target_builder import build_binary_target


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build featured parquet files from raw OHLCV data."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to process. Defaults to config universe.",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help=f"Prediction horizon in trading days. Default: {cfg['prediction_horizon']}",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=f"Minimum return threshold for label=1. Default: {cfg['min_return_threshold']}",
    )
    parser.add_argument(
        "--raw-dir", default=str(RAW_DATA_DIR),
        help="Directory containing raw parquet files.",
    )
    parser.add_argument(
        "--out-dir", default=str(PROCESSED_DATA_DIR),
        help="Output directory for featured parquet files.",
    )
    parser.add_argument(
        "--no-target", action="store_true",
        help="Skip target column generation (features only).",
    )
    return parser.parse_args()


# ── Core ──────────────────────────────────────────────────────────────────────

def build_one_symbol(
    symbol: str,
    raw_dir: Path,
    out_dir: Path,
    horizon: int,
    threshold: float,
    add_target: bool,
) -> dict:
    """
    Load raw data for one symbol, generate features + target, save to parquet.

    Returns a summary dict with symbol, rows, feature count, output path.
    """
    # ── Load raw ──────────────────────────────────────────────────────────────
    parquet_path = raw_dir / f"{symbol}.parquet"
    csv_path     = raw_dir / f"{symbol}.csv"

    if parquet_path.exists():
        raw = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        raw = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No raw file found for {symbol}. "
            f"Run: python scripts/download_yahoo_finance.py --symbols {symbol}"
        )

    # Ensure symbol column exists
    raw = raw.copy()
    raw.columns = [c.lower().strip() for c in raw.columns]
    if "symbol" not in raw.columns:
        raw["symbol"] = symbol

    # ── Feature engineering ───────────────────────────────────────────────────
    featured = generate_ohlcv_features(raw)

    # ── Target ────────────────────────────────────────────────────────────────
    if add_target:
        featured = build_binary_target(
            featured,
            horizon=horizon,
            min_threshold=threshold,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / f"{symbol}_featured.parquet"
    featured.to_parquet(out_path, index=False)

    n_features = featured.shape[1] - 7  # subtract OHLCV + date + symbol
    if add_target:
        n_features -= 2  # subtract forward_return and target

    labeled_rows = int(featured["target"].notna().sum()) if add_target else 0

    return {
        "symbol":       symbol,
        "total_rows":   len(featured),
        "labeled_rows": labeled_rows,
        "n_features":   n_features,
        "output":       str(out_path),
    }


def main() -> None:
    args    = parse_args()
    symbols = args.symbols or cfg["symbols"]
    horizon   = args.horizon   or cfg["prediction_horizon"]
    threshold = args.threshold or cfg["min_return_threshold"]
    raw_dir   = Path(args.raw_dir)
    out_dir   = Path(args.out_dir)

    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Trading-AI  |  Feature Build")
    print("=" * 55)
    print(f"  Symbols:   {symbols}")
    print(f"  Horizon:   {horizon} trading days")
    print(f"  Threshold: {threshold * 100:.2f}% net return for label=1")
    print(f"  Target:    {'yes' if not args.no_target else 'no (features only)'}")
    print(f"  Raw dir:   {raw_dir}")
    print(f"  Out dir:   {out_dir}")
    print()

    results  = []
    failures = []

    for sym in symbols:
        t0 = time.time()
        try:
            summary = build_one_symbol(
                symbol=sym,
                raw_dir=raw_dir,
                out_dir=out_dir,
                horizon=horizon,
                threshold=threshold,
                add_target=not args.no_target,
            )
            elapsed = time.time() - t0
            results.append(summary)
            label_info = (
                f"  labeled={summary['labeled_rows']}"
                if not args.no_target else ""
            )
            print(
                f"  ✓  {sym:<6}  "
                f"rows={summary['total_rows']:<5}  "
                f"features={summary['n_features']:<4}"
                f"{label_info}  "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            failures.append((sym, str(exc)))
            print(f"  ✗  {sym:<6}  FAILED: {exc}")

    print()
    print(f"  Done.  {len(results)} succeeded, {len(failures)} failed.")

    if failures:
        print("\n  Failed symbols:")
        for sym, err in failures:
            print(f"    {sym}: {err}")
        sys.exit(1)

    print()
    print("  Next step:")
    print("    python src/main.py --mode train --model lightgbm")
    print("=" * 55)


if __name__ == "__main__":
    main()
