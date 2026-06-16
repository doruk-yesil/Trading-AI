"""
preprocess.py
-------------
Cleans and standardises raw OHLCV data before feature engineering.

Steps:
  1. Forward-fill small gaps (e.g. single missing trading day due to holiday misalignment)
  2. Drop rows with remaining NaNs in core columns
  3. Ensure DatetimeIndex is sorted ascending
  4. Optionally align all symbols to the same trading calendar (intersection of dates)
  5. Save processed data to data/processed/ as Parquet

Important:
  - Do NOT apply any normalisation here that looks at the full series
    (no z-score, no min-max scaling). Those belong inside the ML pipeline
    with a fitted scaler trained only on the training window.
  - Do NOT create any forward-looking features here.
"""

import pandas as pd
from pathlib import Path

from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.config import cfg


def clean_ohlcv(df: pd.DataFrame, max_fill_days: int = 3) -> pd.DataFrame:
    """
    Basic cleaning for a single OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV indexed by DatetimeIndex.
    max_fill_days : int
        Forward-fill gaps of at most this many consecutive missing days.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # TODO: implement when data arrives
    df = df.sort_index()
    df = df.ffill(limit=max_fill_days)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def align_to_benchmark(
    universe: dict[str, pd.DataFrame],
    benchmark: str = None,
) -> dict[str, pd.DataFrame]:
    """
    Align all symbols to the same set of trading dates.

    Uses the intersection of dates across all symbols (or benchmark dates if provided).
    This prevents feature misalignment when concatenating across symbols.

    Parameters
    ----------
    universe : dict
        {symbol: DataFrame}
    benchmark : str, optional
        If provided, use this symbol's dates as the reference calendar.

    Returns
    -------
    dict
        {symbol: DataFrame} — all aligned to the same DatetimeIndex.
    """
    # TODO: implement when data arrives
    if benchmark is None:
        benchmark = cfg.get("benchmark", "SPY")

    if benchmark in universe:
        ref_dates = universe[benchmark].index
    else:
        # Fallback: use intersection of all dates
        ref_dates = None
        for sym_df in universe.values():
            if ref_dates is None:
                ref_dates = sym_df.index
            else:
                ref_dates = ref_dates.intersection(sym_df.index)

    aligned = {}
    for sym, df in universe.items():
        aligned[sym] = df.reindex(ref_dates).ffill(limit=1).dropna()

    return aligned


def save_processed(df: pd.DataFrame, symbol: str, out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """Save cleaned DataFrame to Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    df.to_parquet(out_path)
    return out_path


def run_preprocessing(universe: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Full preprocessing pipeline.
    1. Clean each symbol
    2. Align to benchmark calendar
    3. Save to processed/

    Returns cleaned + aligned universe.
    """
    # TODO: implement when data arrives
    cleaned = {sym: clean_ohlcv(df) for sym, df in universe.items()}
    aligned = align_to_benchmark(cleaned)

    for sym, df in aligned.items():
        path = save_processed(df, sym)
        print(f"  Saved {sym} → {path}  ({len(df)} rows)")

    return aligned


if __name__ == "__main__":
    print("preprocess.py — run via main.py once data is available.")
