"""
download_hourly_data.py
------------------------
Downloads hourly (1h) OHLCV bars into data/hourly/ -- a SEPARATE folder from
data/raw/ (the daily archive used by the current pipeline).

Why this exists: we are not training on hourly data yet, and the current
feature/target definitions are daily-horizon (mock, still being redesigned).
But Yahoo Finance only serves a rolling ~730-day window for 1h bars -- if we
wait until we actually need hourly history, everything older than ~2 years
before that day will already be gone. So this script runs daily, ACCUMULATES
(merges + de-duplicates on date) new bars on top of whatever was already
saved, so the local archive keeps growing past the 730-day rolling limit
even though any single Yahoo request can't reach back that far.

Not wired into build_features.py / the training pipeline -- pure archival.

Usage:
  python scripts/download_hourly_data.py
  python scripts/download_hourly_data.py --symbols AAPL MSFT SPY
"""

from __future__ import annotations

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from src.utils.paths import HOURLY_DATA_DIR, ensure_dirs
from src.utils.universe import resolve_universe

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
INTER_SYMBOL_SLEEP_SECONDS = 0.3

# Yahoo's actual enforced ceiling for interval="1h" as of writing.
# https://github.com/ranaroussi/yfinance -- intraday intervals below 1d are
# capped; 60m/1h tops out at 730 days.
HOURLY_MAX_PERIOD = "730d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accumulate hourly OHLCV bars (future use, not used by current pipeline)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override the universe.")
    parser.add_argument("--interval", default="1h", help="Yahoo interval (1h default).")
    parser.add_argument("--out-dir", default=str(HOURLY_DATA_DIR), help="Output directory.")
    return parser.parse_args()


def sanitize_symbol(symbol: str) -> str:
    return symbol.upper().replace("^", "")


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("Yahoo Finance returned no rows.")
    df = history.reset_index().copy()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    else:
        raise ValueError("Could not find a date column in Yahoo Finance output.")

    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df[required].dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_and_accumulate(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """
    Merge freshly-pulled bars on top of whatever was already archived,
    de-duplicating on timestamp. This is what lets the local file grow past
    Yahoo's rolling 730-day serving window over time -- old bars that have
    since aged out of Yahoo's API are kept because they're already on disk.
    """
    if existing is None or existing.empty:
        return fresh
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def download_symbol_hourly(
    symbol: str,
    interval: str,
    out_dir: Path,
) -> tuple[Path, int, str, str, int]:
    """Fetch + merge + save one symbol. Returns (path, total_rows, earliest, latest, new_rows)."""
    out_path = out_dir / f"{sanitize_symbol(symbol)}.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else None

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            history = yf.Ticker(symbol).history(
                period=HOURLY_MAX_PERIOD,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            fresh = normalize_history(history)
            merged = merge_and_accumulate(existing, fresh)
            merged.to_parquet(out_path, index=False)

            new_rows = len(merged) - (len(existing) if existing is not None else 0)
            earliest = merged["date"].min().strftime("%Y-%m-%d %H:%M")
            latest = merged["date"].max().strftime("%Y-%m-%d %H:%M")
            return out_path, len(merged), earliest, latest, new_rows
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc  # type: ignore[misc]


def main() -> None:
    args = parse_args()
    ensure_dirs()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_universe(override_symbols=args.symbols)

    print("Hourly accumulation download starting (NOT used by current training pipeline)...")
    print(f"  Universe size: {len(symbols)} symbols")
    print(f"  Interval: {args.interval}  (Yahoo max lookback: {HOURLY_MAX_PERIOD})")
    print(f"  Output dir: {out_dir}")
    print()

    manifest_rows = []
    n_ok, n_failed = 0, 0
    for i, symbol in enumerate(symbols, start=1):
        try:
            out_path, total_rows, earliest, latest, new_rows = download_symbol_hourly(
                symbol=symbol, interval=args.interval, out_dir=out_dir,
            )
            manifest_rows.append({
                "symbol": symbol,
                "file": out_path.name,
                "total_rows": total_rows,
                "new_rows_this_run": new_rows,
                "earliest": earliest,
                "latest": latest,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            n_ok += 1
            print(f"  [{i}/{len(symbols)}] {symbol}: {total_rows} rows total "
                  f"(+{new_rows} new), {earliest} -> {latest}")
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{len(symbols)}] FAILED {symbol}: {exc}")
        time.sleep(INTER_SYMBOL_SLEEP_SECONDS)

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(out_dir / "hourly_download_manifest.csv", index=False)

    print(f"\nHourly accumulation complete. {n_ok} succeeded, {n_failed} failed out of {len(symbols)}.")


if __name__ == "__main__":
    main()
