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

import os
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

# Force UTF-8 for child stages. Without this, on Windows a subprocess whose
# stdout is captured (a pipe, not a real console) falls back to the system
# locale codepage (often cp1252), which crashes the moment a script prints a
# non-ASCII character (e.g. build_features.py's old checkmark symbols). Same
# risk applies when this script itself is run non-interactively (Task
# Scheduler, output redirected to a file) -- see run_daily_refresh.bat.
#
# PYTHONUNBUFFERED=1 matters separately: when a child's stdout is a pipe
# (not a real terminal), Python block-buffers it by default, so print()
# calls don't actually reach us until the buffer fills or the process exits.
# For a 504-symbol download that takes minutes, that looks exactly like a
# frozen terminal -- nothing prints until the whole stage is done. This
# forces the child to flush every line immediately instead.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full daily data refresh pipeline.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe for a quick test run.")
    parser.add_argument("--skip-daily", action="store_true", help="Skip the daily OHLCV re-download.")
    parser.add_argument("--skip-hourly", action="store_true", help="Skip the hourly accumulation step.")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature rebuild.")
    return parser.parse_args()


def run_stage(name: str, cmd: list[str]) -> tuple[bool, str]:
    """
    Runs a stage and streams its output LIVE, line by line, instead of
    buffering everything until the subprocess exits. A plain
    subprocess.run(capture_output=True) waits for the whole stage (minutes,
    for a 504-symbol download) before printing a single character -- looks
    exactly like a hung terminal even though it's working fine in the
    background. Popen + reading stdout line-by-line as it arrives fixes that.
    """
    print("=" * 70)
    print(f"  STAGE: {name}")
    print("=" * 70)

    process = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=CHILD_ENV, bufsize=1,
    )

    output_lines: list[str] = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    process.wait()

    combined_output = "".join(output_lines)
    if process.returncode != 0:
        print(f"  FAIL  {name} (exit code {process.returncode})\n")
        return False, combined_output
    print(f"  OK    {name} complete\n")
    return True, combined_output


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
    stage_outputs = {}

    if not args.skip_daily:
        ok, out = run_stage(
            "Daily OHLCV download (overwrite, max history)",
            [python, str(SCRIPTS_DIR / "download_yahoo_finance.py"), *symbol_flag],
        )
        stage_results["daily_download"] = ok
        stage_outputs["daily_download"] = out
    else:
        print("Skipping daily download stage.\n")

    if not args.skip_hourly:
        ok, out = run_stage(
            "Hourly OHLCV accumulation (future use, separate folder)",
            [python, str(SCRIPTS_DIR / "download_hourly_data.py"), *symbol_flag],
        )
        stage_results["hourly_download"] = ok
        stage_outputs["hourly_download"] = out
    else:
        print("Skipping hourly accumulation stage.\n")

    if not args.skip_features:
        ok, out = run_stage(
            "Feature rebuild (data/processed/*_featured.parquet)",
            [python, str(SCRIPTS_DIR / "build_features.py"), *symbol_flag],
        )
        stage_results["build_features"] = ok
        stage_outputs["build_features"] = out
    else:
        print("Skipping feature rebuild stage.\n")

    run_finished = datetime.now()
    elapsed = (run_finished - run_started).total_seconds()

    # Full per-stage stdout/stderr is persisted here -- not just True/False --
    # so a failure can be diagnosed later (e.g. from a Task Scheduler run
    # nobody watched live) without needing to reproduce it interactively.
    log_path = LOG_DIR / f"refresh_{run_started.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Started:  {run_started.isoformat()}\n")
        f.write(f"Finished: {run_finished.isoformat()}\n")
        f.write(f"Elapsed:  {elapsed:.1f}s\n")
        f.write(f"Universe: {len(universe)} symbols\n")
        f.write(f"Stage results: {stage_results}\n")
        for stage_name, output_text in stage_outputs.items():
            f.write(f"\n{'=' * 70}\n{stage_name} -- full output\n{'=' * 70}\n")
            f.write(output_text)
            f.write("\n")

    print("=" * 70)
    print(f"  Daily refresh finished in {elapsed:.1f}s")
    print(f"  Stage results: {stage_results}")
    print(f"  Log saved -> {log_path}")
    print("=" * 70)

    if not all(stage_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
