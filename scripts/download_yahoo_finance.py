"""
download_yahoo_finance.py
-------------------------
Download historical OHLCV data from Yahoo Finance and store one file per symbol
in data/raw/ as Parquet.

Examples:
  python scripts/download_yahoo_finance.py
  python scripts/download_yahoo_finance.py --symbols AAPL MSFT SPY
  python scripts/download_yahoo_finance.py --start-date 2018-01-01 --end-date 2024-12-31

Notes:
  - Uses the symbols from configs/config.yaml by default.
  - Stores normalized columns: date, open, high, low, close, volume.
  - Existing files are overwritten so the raw dataset stays reproducible.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import cfg
from src.utils.paths import RAW_DATA_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Tickers to download. Defaults to the configured universe.",
    )
    parser.add_argument(
        "--start-date",
        default=cfg["start_date"],
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=cfg["end_date"],
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Yahoo Finance interval, e.g. 1d, 1h, 1wk.",
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
    start_date: str,
    end_date: str,
    interval: str,
    auto_adjust: bool,
    raw_dir: Path,
) -> tuple[Path, int]:
    history = yf.Ticker(symbol).history(
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
    )
    normalized = normalize_history(history)

    out_path = raw_dir / f"{sanitize_symbol(symbol)}.parquet"
    normalized.to_parquet(out_path, index=False)
    return out_path, len(normalized)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols or cfg["symbols"]
    auto_adjust = not args.no_auto_adjust

    manifest_rows = []
    print("Yahoo Finance download starting...")
    print(f"  Symbols: {symbols}")
    print(f"  Date range: {args.start_date} -> {args.end_date}")
    print(f"  Interval: {args.interval}")
    print(f"  Output dir: {raw_dir}")

    for symbol in symbols:
        try:
            out_path, row_count = download_symbol(
                symbol=symbol,
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
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "interval": args.interval,
                    "auto_adjust": auto_adjust,
                    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            print(f"  Saved {symbol} -> {out_path} ({row_count} rows)")
        except Exception as exc:
            print(f"  Failed {symbol}: {exc}")

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = raw_dir / "yahoo_download_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest saved -> {manifest_path}")

    print("Download complete.")


if __name__ == "__main__":
    main()