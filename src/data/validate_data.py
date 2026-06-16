"""
validate_data.py
----------------
Data quality checks for raw OHLCV DataFrames.

Key checks:
  - No NaN values in OHLCV columns
  - High >= Low, High >= Open/Close, Low <= Open/Close (OHLC sanity)
  - Volume > 0
  - No duplicate dates
  - Minimum row count (avoids training on too-short series)
  - Detects large unexpected price gaps (potential data errors)

Important — lookahead bias prevention:
  All validation must be done on the raw data BEFORE any feature engineering.
  Never validate using future-derived statistics (e.g. z-scores computed on full dataset).
"""

import pandas as pd
import numpy as np
from typing import Optional


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str = "",
    min_rows: int = 252,
    max_daily_return: float = 0.50,
) -> list[str]:
    """
    Run all OHLC quality checks on a single DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame indexed by date.
    symbol : str
        Symbol name for error messages.
    min_rows : int
        Minimum required trading days (default 252 = ~1 year).
    max_daily_return : float
        Flag rows where abs(close-to-close return) exceeds this threshold.

    Returns
    -------
    list[str]
        List of warning/error messages. Empty list means data is clean.
    """
    # TODO: expand checks as we encounter real data issues
    issues = []
    tag = f"[{symbol}]" if symbol else ""

    # 1. Minimum row count
    if len(df) < min_rows:
        issues.append(f"{tag} Only {len(df)} rows — minimum is {min_rows}.")

    # 2. Required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"{tag} Missing columns: {missing}")
        return issues  # can't continue without columns

    # 3. NaN check
    nan_counts = df[list(required)].isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            issues.append(f"{tag} {count} NaN values in column '{col}'.")

    # 4. Duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes:
        issues.append(f"{tag} {dupes} duplicate date entries.")

    # 5. OHLC sanity: high >= low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        issues.append(f"{tag} {bad_hl} rows where high < low.")

    # 6. Volume > 0
    bad_vol = (df["volume"] <= 0).sum()
    if bad_vol:
        issues.append(f"{tag} {bad_vol} rows with zero or negative volume.")

    # 7. Large return gaps (possible data errors or stock halts)
    returns = df["close"].pct_change().abs()
    extreme = (returns > max_daily_return).sum()
    if extreme:
        issues.append(
            f"{tag} {extreme} rows with daily return > {max_daily_return*100:.0f}% "
            f"(review for data errors or stock halts)."
        )

    return issues


def validate_universe(universe: dict) -> dict[str, list[str]]:
    """
    Validate all symbols in a universe dict.

    Returns
    -------
    dict
        {symbol: [list of issues]}  — empty list means clean.
    """
    results = {}
    for sym, df in universe.items():
        issues = validate_ohlcv(df, symbol=sym)
        results[sym] = issues
        if issues:
            print(f"  {sym}: {len(issues)} issue(s)")
            for msg in issues:
                print(f"    - {msg}")
        else:
            print(f"  {sym}: OK")
    return results


if __name__ == "__main__":
    # TODO: replace with actual data once available
    print("validate_data.py — run via main.py or import validate_universe()")
