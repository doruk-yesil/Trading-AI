"""
daily_refresh.py
-----------------
Single entry point for the automated daily data refresh. This is the script
that should be run on a schedule (see scripts/run_daily_refresh.bat +
Windows Task Scheduler setup in README.md).

Each run:
  1. Resolves the universe (S&P 500 + always-include extras, see
     src/utils/universe.py).
  2. Re-downloads daily OHLCV for the whole universe (overwrites data/raw/,
     max available history per symbol) -- scripts/download_yahoo_finance.py
  3. Accumulates hourly OHLCV into data/hourly/ (separate, NOT used by the
     training pipeline yet -- kept for future use since Yahoo's hourly
     lookback is capped at ~730 days) -- scripts/download_hourly_data.py
  4. Rebuilds data/processed/*_featured.parquet from the fresh daily data
     -- scripts/build_features.py

Steps run as subprocesses (not imports) so a failure in one stage doesn't
take down the others, and each stage's own CLI / manifest logic stays the
single source of truth.

Note: the current feature set (173 columns) and 5-day binary target are
placeholders (mock) and expected to be redefined. This script doesn't
hardcode that -- it just always regenerates whatever build_features.py
currently produces, so once the feature/target definitions change, this
pipeline picks it up automatically with no changes needed here.

Usage:
  python scripts/daily_refresh.py
  python scripts/daily_refresh.py --skip-hourly
  python scripts/daily_refresh.py --symbols AAPL MSFT SPY   # quick test run
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ROOT, ensure_dirs, REPORTS_DIR
from src.utils.universe import resolve_universe

SCRIPTS_DIR = ROOT / "scripts"
LOG_DIR = REPORTS_DIR / "refresh_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full daily data refresh pipeline.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe for a quick test run.")
    parser.add_argument("--skip-daily", action="store_true", help="Skip the daily OHLCV re-download.")
    parser.add_argument("--skip-hourly", action="store_true", help="Skip the hourly accumulation step.")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature rebuild.")
    return parser.parse_args()


def run_stage(name: str, cmd: list[str]) -> tuple[bool, str]:
    print("=" * 70)
    print(f"  STAGE: {name}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"  ✗ {name} FAILED (exit code {result.returncode})\n")
        return False, result.stdout + result.stderr
    print(f"  ✓ {name} complete\n")
    return True, result.stdout


def main() -> None:
    args = parse_args()
    ensure_dirs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now()
    print(f"Trading-AI daily refresh starting at {run_started.isoformat(timespec='seconds')}")

    universe = resolve_universe(override_symbols=args.symbols)
    print(f"Resolved universe: {len(universe)} symbols\n")

    python = sys.executable
    symbol_flag = ["--symbols", *args.symbols] if args.symbols else []

    stage_results = {}

    if not args.skip_daily:
        ok, out = run_stage(
            "Daily OHLCV download (overwrite, max history)",
            [python, str(SCRIPTS_DIR / "download_yahoo_finance.py"), *symbol_flag],
        )
        stage_results["daily_download"] = ok
    else:
        print("Skipping daily download stage.\n")

    if not args.skip_hourly:
        ok, out = run_stage(
            "Hourly OHLCV accumulation (future use, separate folder)",
            [python, str(SCRIPTS_DIR / "download_hourly_data.py"), *symbol_flag],
        )
        stage_results["hourly_download"] = ok
    else:
        print("Skipping hourly accumulation stage.\n")

    if not args.skip_features:
        ok, out = run_stage(
            "Feature rebuild (data/processed/*_featured.parquet)",
            [python, str(SCRIPTS_DIR / "build_features.py"), *symbol_flag],
        )
        stage_results["build_features"] = ok
    else:
        print("Skipping feature rebuild stage.\n")

    run_finished = datetime.now()
    elapsed = (run_finished - run_started).total_seconds()

    log_path = LOG_DIR / f"refresh_{run_started.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w") as f:
        f.write(f"Started:  {run_started.isoformat()}\n")
        f.write(f"Finished: {run_finished.isoformat()}\n")
        f.write(f"Elapsed:  {elapsed:.1f}s\n")
        f.write(f"Universe: {len(universe)} symbols\n")
        f.write(f"Stage results: {stage_results}\n")

    print("=" * 70)
    print(f"  Daily refresh finished in {elapsed:.1f}s")
    print(f"  Stage results: {stage_results}")
    print(f"  Log saved -> {log_path}")
    print("=" * 70)

    if not all(stage_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
