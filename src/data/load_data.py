"""
load_data.py
------------
Loads raw OHLCV data from disk into a standardised pandas DataFrame.

Expected raw file format: one CSV or Parquet file per symbol.
  - Filename: {SYMBOL}.csv  or  {SYMBOL}.parquet
  - Required columns: date, open, high, low, close, volume
  - Date format: YYYY-MM-DD
  - No missing rows in the middle of the series (gaps are validated separately)
  - Adjusted prices preferred (split + dividend adjusted close)

See scripts/data_format_guide.md for the full specification your teammate should follow.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.config import cfg
from src.utils.universe import resolve_universe


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def load_symbol(
    symbol: str,
    data_dir: Path = RAW_DATA_DIR,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for a single symbol from disk.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL"
    data_dir : Path
        Directory containing raw data files.
    start_date : str, optional
        Filter rows from this date inclusive (YYYY-MM-DD).
    end_date : str, optional
        Filter rows up to this date inclusive (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Indexed by date (DatetimeIndex), columns: open, high, low, close, volume.

    Raises
    ------
    FileNotFoundError
        If no data file is found for the symbol.
    ValueError
        If required columns are missing.
    """
    # TODO: implement when data arrives
    # Try parquet first, fall back to CSV
    parquet_path = data_dir / f"{symbol}.parquet"
    csv_path     = data_dir / f"{symbol}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No data file found for {symbol} in {data_dir}.\n"
            f"Expected: {parquet_path} or {csv_path}"
        )

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing required columns: {missing}")

    # Set DatetimeIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Optional date filtering
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    return df


def load_universe(
    symbols: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data for all symbols in the universe.

    Returns a dict: {symbol: DataFrame}
    Symbols that fail to load are skipped with a warning.

    start_date/end_date are optional filters -- left as None by default so a
    full period="max" raw pull (see scripts/download_yahoo_finance.py) isn't
    silently truncated to the old fixed 2015-2024 config window.
    """
    if symbols is None:
        symbols = resolve_universe()

    universe = {}
    for sym in symbols:
        try:
            universe[sym] = load_symbol(sym, start_date=start_date, end_date=end_date)
            print(f"  Loaded {sym}: {len(universe[sym])} rows")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    return universe


def load_featured_symbol(
    symbol: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load pre-built featured parquet for a single symbol from data/processed/.

    These files are created by:
        python scripts/build_features.py

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL"
    processed_dir : Path
        Directory containing *_featured.parquet files.
    start_date / end_date : str, optional
        Date filter (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        173 feature columns + date, symbol, OHLCV, forward_return_5d, target.

    Raises
    ------
    FileNotFoundError
        If the featured file doesn't exist yet (run build_features.py first).
    """
    path = processed_dir / f"{symbol}_featured.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Featured file not found: {path}\n"
            f"Run:  python scripts/build_features.py --symbols {symbol}"
        )

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    return df


def load_featured_universe(
    symbols: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    concat: bool = False,
) -> "dict[str, pd.DataFrame] | pd.DataFrame":
    """
    Load featured DataFrames for all symbols.

    Parameters
    ----------
    symbols : list, optional
        Defaults to config universe.
    start_date / end_date : str, optional
        Date filter applied to each symbol.
    concat : bool
        If True, return a single concatenated DataFrame (all symbols stacked).
        If False (default), return a dict {symbol: DataFrame}.

    Returns
    -------
    dict or pd.DataFrame
    """
    if symbols is None:
        symbols = resolve_universe()

    universe = {}
    for sym in symbols:
        try:
            universe[sym] = load_featured_symbol(
                sym, start_date=start_date, end_date=end_date
            )
            df = universe[sym]
            labeled = int(df["target"].notna().sum()) if "target" in df.columns else 0
            print(f"  Loaded {sym}: {len(df)} rows, {labeled} labeled")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if concat and universe:
        return pd.concat(universe.values(), ignore_index=True)

    return universe


if __name__ == "__main__":
    # Quick smoke test — will show warnings until data is present
    data = load_universe()
    print(f"\nLoaded {len(data)} raw symbols.")

    featured = load_featured_universe()
    print(f"Loaded {len(featured)} featured symbols.")
