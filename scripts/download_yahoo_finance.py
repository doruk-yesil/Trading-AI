"""
download_yahoo_finance.py
-------------------------
Download historical daily OHLCV data from Yahoo Finance and store one file
per symbol in data/raw/ as Parquet. Overwrites existing files so the raw
dataset always reflects the latest full pull (reproducible, not append-only).

Universe: resolved via src/utils/universe.py. Default (configs/config.yaml
universe_mode: "sp500") pulls the *current* full S&P 500 constituent list
plus always-include extras (SPY benchmark). Pass --symbols to override with
a small manual list for quick tests.

History depth: defaults to period="max" (the earliest history Yahoo has for
each symbol), not a fixed start/end date -- so newer/older listings each get
as much history as actually exists. Pass --start-date/--end-date to instead
pull a fixed window (period and start/end are mutually exclusive on Yahoo's
API, so passing dates switches the script out of "max" mode automatically).

Examples:
  python scripts/download_yahoo_finance.py
  python scripts/download_yahoo_finance.py --symbols AAPL MSFT SPY
  python scripts/download_yahoo_finance.py --start-date 2018-01-01 --end-date 2024-12-31

Notes:
  - Stores normalized columns: date, open, high, low, close, volume.
  - Existing files are overwritten so the raw dataset stays reproducible.
  - Run via scripts/daily_refresh.py for the full daily+hourly+features cycle.
"""

from __future__ import annotations

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from src.utils.paths import RAW_DATA_DIR, ensure_dirs
from src.utils.universe import resolve_universe

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
INTER_SYMBOL_SLEEP_SECONDS = 0.3  # be polite to Yahoo across a large (500+) universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance daily OHLCV data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Tickers to download. Overrides universe_mode entirely (e.g. quick tests).",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="Yahoo Finance period (e.g. max, 10y, 5y). Ignored if --start-date is given.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD. If given (with --end-date), overrides --period.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date YYYY-MM-DD. If given (with --start-date), overrides --period.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Yahoo Finance interval, e.g. 1d, 1wk.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DATA_DIR),
        help="Output directory for raw data files.",
    )
    parser.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable auto-adjusted prices from Yahoo Finance.",
    )
    return parser.parse_args()


def sanitize_symbol(symbol: str) -> str:
    return symbol.upper().replace("^", "")


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("Yahoo Finance returned no rows.")

    df = history.reset_index().copy()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        raise ValueError("Could not find a date column in Yahoo Finance output.")

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from Yahoo Finance output: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df[required].dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def download_symbol(
    symbol: str,
    period: str,
    start_date: str | None,
    end_date: str | None,
    interval: str,
    auto_adjust: bool,
    raw_dir: Path,
) -> tuple[Path, int, str, str]:
    """Download + normalize + save one symbol. Retries transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ticker = yf.Ticker(symbol)
            if start_date and end_date:
                history = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=False,
                )
            else:
                history = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=False,
                )
            normalized = normalize_history(history)

            out_path = raw_dir / f"{sanitize_symbol(symbol)}.parquet"
            normalized.to_parquet(out_path, index=False)
            earliest = normalized["date"].min().strftime("%Y-%m-%d")
            latest = normalized["date"].max().strftime("%Y-%m-%d")
            return out_path, len(normalized), earliest, latest
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc  # type: ignore[misc]


def main() -> None:
    args = parse_args()
    ensure_dirs()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_universe(override_symbols=args.symbols)
    auto_adjust = not args.no_auto_adjust
    using_fixed_range = bool(args.start_date and args.end_date)

    manifest_rows = []
    print("Yahoo Finance daily download starting...")
    print(f"  Universe size: {len(symbols)} symbols")
    if using_fixed_range:
        print(f"  Date range: {args.start_date} -> {args.end_date} (fixed window)")
    else:
        print(f"  Period: {args.period} (max available history per symbol)")
    print(f"  Interval: {args.interval}")
    print(f"  Output dir: {raw_dir}")
    print()

    n_ok, n_failed = 0, 0
    for i, symbol in enumerate(symbols, start=1):
        try:
            out_path, row_count, earliest, latest = download_symbol(
                symbol=symbol,
                period=args.period,
                start_date=args.start_date,
                end_date=args.end_date,
                interval=args.interval,
                auto_adjust=auto_adjust,
                raw_dir=raw_dir,
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "file": out_path.name,
                    "rows": row_count,
                    "earliest_date": earliest,
                    "latest_date": latest,
                    "interval": args.interval,
                    "auto_adjust": auto_adjust,
                    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            n_ok += 1
            print(f"  [{i}/{len(symbols)}] Saved {symbol} -> {out_path.name} "
                  f"({row_count} rows, {earliest} -> {latest})")
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{len(symbols)}] FAILED {symbol}: {exc}")
        time.sleep(INTER_SYMBOL_SLEEP_SECONDS)

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = raw_dir / "yahoo_download_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest saved -> {manifest_path}")

    print(f"\nDownload complete. {n_ok} succeeded, {n_failed} failed out of {len(symbols)}.")


if __name__ == "__main__":
    main()
