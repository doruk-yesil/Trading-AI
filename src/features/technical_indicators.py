"""
technical_indicators.py
-----------------------
Computes technical features from a single-symbol OHLCV DataFrame.

All indicators are computed in a strictly causal way — no future data is used.
Each function takes a DataFrame and returns the same DataFrame with new columns added.

LOOKAHEAD BIAS WARNING:
  Never use df.rolling(...).apply(func) with functions that peek at future values.
  Always use shift(1) when you need yesterday's value.
  The train/test split must happen AFTER feature engineering — but features must
  only look backward at any given row.

Feature categories:
  - Returns: daily, log, multi-window
  - Moving averages: SMA 5/10/20/50/100/200
  - Price relative to MAs
  - MA crossover signals
  - RSI
  - MACD
  - Bollinger Bands
  - Volatility (rolling std of returns)
  - Volume features: rolling avg, volume ratio
  - ATR (Average True Range)
  - Momentum over multiple windows
  - Benchmark (SPY) relative features
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.config import cfg


# ── Returns ───────────────────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily and log returns."""
    df = df.copy()
    df["return_1d"]     = df["close"].pct_change()
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_multi_window_returns(df: pd.DataFrame, windows: Optional[list] = None) -> pd.DataFrame:
    """Cumulative returns over multiple windows."""
    df = df.copy()
    windows = windows or cfg["feature_windows"]["momentum_windows"]
    for w in windows:
        df[f"return_{w}d"] = df["close"].pct_change(w)
    return df


# ── Moving Averages ───────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame, windows: Optional[list] = None) -> pd.DataFrame:
    """Simple moving averages (SMA)."""
    df = df.copy()
    windows = windows or cfg["feature_windows"]["moving_averages"]
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
    return df


def add_price_vs_ma(df: pd.DataFrame, windows: Optional[list] = None) -> pd.DataFrame:
    """Price relative to moving average: (close - SMA) / SMA."""
    df = df.copy()
    windows = windows or cfg["feature_windows"]["moving_averages"]
    for w in windows:
        col = f"sma_{w}"
        if col not in df.columns:
            df[col] = df["close"].rolling(w).mean()
        df[f"close_vs_sma_{w}"] = (df["close"] - df[col]) / df[col]
    return df


def add_ma_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    MA crossover features.
    sma_5_vs_20: (SMA5 - SMA20) / SMA20  — short vs medium trend
    sma_20_vs_50: medium vs long trend
    sma_50_vs_200: golden/death cross indicator
    """
    df = df.copy()
    for short, long in [(5, 20), (20, 50), (50, 200)]:
        s_col = f"sma_{short}"
        l_col = f"sma_{long}"
        if s_col not in df.columns:
            df[s_col] = df["close"].rolling(short).mean()
        if l_col not in df.columns:
            df[l_col] = df["close"].rolling(long).mean()
        df[f"sma_{short}_vs_{long}"] = (df[s_col] - df[l_col]) / df[l_col]
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
    """Relative Strength Index using Wilder's smoothing."""
    df = df.copy()
    period = period or cfg["feature_windows"]["rsi_period"]

    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ─────────────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD, signal line, and histogram."""
    df = df.copy()
    fast   = cfg["feature_windows"]["macd"]["fast"]
    slow   = cfg["feature_windows"]["macd"]["slow"]
    signal = cfg["feature_windows"]["macd"]["signal"]

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"]           = ema_fast - ema_slow
    df["macd_signal"]    = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands: upper, lower, and %B position.
    %B = (close - lower) / (upper - lower)  → 0 = at lower band, 1 = at upper band
    """
    df = df.copy()
    window  = cfg["feature_windows"]["bollinger_bands"]["window"]
    num_std = cfg["feature_windows"]["bollinger_bands"]["num_std"]

    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


# ── Volatility ────────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame, windows: list = [10, 20, 60]) -> pd.DataFrame:
    """Rolling annualised volatility of log returns."""
    df = df.copy()
    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    for w in windows:
        df[f"vol_{w}d"] = df["log_return_1d"].rolling(w).std() * np.sqrt(252)
    return df


# ── Volume ────────────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling volume average and volume ratio (today vs average)."""
    df = df.copy()
    for w in cfg["feature_windows"]["volume_ma_windows"]:
        df[f"vol_ma_{w}"] = df["volume"].rolling(w).mean()

    # Volume ratio: today's volume / 20-day average
    if "vol_ma_20" not in df.columns:
        df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma_20"]
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
    """Average True Range — measure of daily price range / volatility."""
    df = df.copy()
    period = period or cfg["feature_windows"]["atr_period"]

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]           = true_range.ewm(span=period, adjust=False).mean()
    df["atr_pct_close"] = df["atr"] / df["close"]   # normalised ATR
    return df


# ── Momentum ──────────────────────────────────────────────────────────────────

def add_momentum(df: pd.DataFrame, windows: Optional[list] = None) -> pd.DataFrame:
    """
    Momentum: return over the past N days (already in add_multi_window_returns).
    Here we also add rate-of-change (ROC) and a momentum z-score.
    """
    df = df.copy()
    windows = windows or cfg["feature_windows"]["momentum_windows"]
    for w in windows:
        col = f"return_{w}d"
        if col not in df.columns:
            df[col] = df["close"].pct_change(w)
        # Rolling z-score of the momentum (look-back normalisation, causal)
        roll = df[col].rolling(252, min_periods=63)
        df[f"momentum_z_{w}d"] = (df[col] - roll.mean()) / roll.std()
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicator functions in sequence.
    Returns a new DataFrame with all feature columns appended.
    """
    # TODO: adjust order / add features as needed
    df = add_returns(df)
    df = add_multi_window_returns(df)
    df = add_moving_averages(df)
    df = add_price_vs_ma(df)
    df = add_ma_crossovers(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_atr(df)
    df = add_momentum(df)
    return df


if __name__ == "__main__":
    print("technical_indicators.py — import build_all_features() to use.")
