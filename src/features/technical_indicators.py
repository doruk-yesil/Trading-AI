"""
technical_indicators.py
-----------------------
OHLCV feature engineering for the Trading-AI research platform.

Public entry point
------------------
    generate_ohlcv_features(df) -> pd.DataFrame

Input DataFrame must have columns (case-insensitive):
    date, symbol, open, high, low, close, volume

Features are computed per symbol, sorted by date ascending.
All rolling lookbacks are strictly backward-looking (no lookahead bias).
Rolling high/low values used for comparisons are shifted by 1 trading day
before comparing with today's Close.

Safe division
-------------
All divisions go through safe_div() which returns NaN where the denominator
is zero or NaN — no silent fake-value filling.

Backward-compatible aliases
---------------------------
    build_all_features(df)  — wraps generate_ohlcv_features for single-symbol use
    add_rsi(df, period)     — thin wrapper kept for strategy modules
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: set[str] = {"date", "open", "high", "low", "close", "volume"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """
    Divide a by b element-wise.
    Returns NaN wherever b is zero or NaN — never fills with fake values.
    """
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    elif b == 0:
        return pd.Series(np.nan, index=a.index)
    return a / b


def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase, coerce types, validate required columns.
    Raises ValueError if any required column is missing.
    """
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_ohlcv_features: missing required columns: {missing}\n"
            f"Got: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI using Wilder's exponential smoothing (alpha = 1/period)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = safe_div(avg_gain, avg_loss)
    return 100 - safe_div(pd.Series(100, index=close.index), 1 + rs)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    """
    Returns (atr_series, true_range_series).
    ATR uses Wilder's EWM smoothing.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr, tr


def _compute_adx(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (adx, plus_di, minus_di) using Wilder's smoothing.
    All values are percentages (0-100).
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm_raw  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm_raw = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm_raw,  index=high.index, dtype=float)
    minus_dm = pd.Series(minus_dm_raw, index=high.index, dtype=float)

    # Wilder smoothing
    alpha = 1 / period
    atr_s    = tr.ewm(alpha=alpha,    min_periods=period, adjust=False).mean()
    plus_di  = 100 * safe_div(
        plus_dm.ewm(alpha=alpha,  min_periods=period, adjust=False).mean(), atr_s
    )
    minus_di = 100 * safe_div(
        minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean(), atr_s
    )

    dx  = 100 * safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


# ── Feature-group functions (all operate on single-symbol DataFrames) ─────────

def _base_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1 — Base price features.
    Uses only OHLC data from the current and previous row.
    """
    prev_close = df["close"].shift(1)
    hl_range   = df["high"] - df["low"]

    df["daily_return"]            = safe_div(df["close"], prev_close) - 1
    df["log_return"]              = np.log(safe_div(df["close"], prev_close))
    df["gap_return"]              = safe_div(df["open"],  prev_close) - 1
    df["intraday_return"]         = safe_div(df["close"], df["open"])  - 1
    df["overnight_return"]        = safe_div(df["open"],  prev_close) - 1   # == gap_return
    df["high_low_range"]          = hl_range
    df["high_low_range_pct"]      = safe_div(hl_range,            df["close"])
    df["open_close_range_pct"]    = safe_div(df["close"] - df["open"], df["open"])
    df["close_position_in_candle"]= safe_div(df["close"] - df["low"],  hl_range)

    return df


def _candle_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 2 — Candle structure features.
    All values derived from the single current candle.
    """
    body        = (df["close"] - df["open"]).abs()
    candle_range= df["high"] - df["low"]
    upper_wick  = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick  = df[["open", "close"]].min(axis=1) - df["low"]

    df["candle_body"]           = body
    df["candle_body_pct"]       = safe_div(body,        df["open"])
    df["candle_range"]          = candle_range
    df["upper_wick"]            = upper_wick
    df["lower_wick"]            = lower_wick
    df["body_to_range"]         = safe_div(body,        candle_range)
    df["upper_wick_to_range"]   = safe_div(upper_wick,  candle_range)
    df["lower_wick_to_range"]   = safe_div(lower_wick,  candle_range)

    btr = safe_div(body, candle_range)
    uwr = safe_div(upper_wick, candle_range)
    lwr = safe_div(lower_wick, candle_range)

    df["green_candle"]     = (df["close"] > df["open"]).astype(float)
    df["red_candle"]       = (df["close"] < df["open"]).astype(float)
    df["doji_like"]        = (btr < 0.1).astype(float)
    df["long_upper_wick"]  = (uwr > 0.5).astype(float)
    df["long_lower_wick"]  = (lwr > 0.5).astype(float)
    df["large_body"]       = (btr > 0.7).astype(float)

    return df


def _moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 3 — SMA / EMA values, normalised ratios, crossover flags.
    """
    windows = [5, 10, 20, 50, 100, 200]

    # Compute raw MAs
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # Close vs MA (normalised)
    for w in windows:
        df[f"close_vs_sma_{w}"] = safe_div(df["close"], df[f"sma_{w}"]) - 1
        df[f"close_vs_ema_{w}"] = safe_div(df["close"], df[f"ema_{w}"]) - 1

    # MA crossover ratios
    for short, long in [(5, 20), (20, 50), (50, 200)]:
        df[f"sma{short}_vs_sma{long}"] = safe_div(df[f"sma_{short}"], df[f"sma_{long}"]) - 1
        df[f"ema{short}_vs_ema{long}"] = safe_div(df[f"ema_{short}"], df[f"ema_{long}"]) - 1

    # Flags
    df["close_gt_sma20"]  = (df["close"] > df["sma_20"]).astype(float)
    df["close_gt_sma50"]  = (df["close"] > df["sma_50"]).astype(float)
    df["close_gt_sma200"] = (df["close"] > df["sma_200"]).astype(float)
    df["sma20_gt_sma50"]  = (df["sma_20"]  > df["sma_50"]).astype(float)
    df["sma50_gt_sma200"] = (df["sma_50"]  > df["sma_200"]).astype(float)
    df["ema20_gt_ema50"]  = (df["ema_20"]  > df["ema_50"]).astype(float)
    df["ema50_gt_ema200"] = (df["ema_50"]  > df["ema_200"]).astype(float)

    return df


def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4 — Multi-window returns, ROC, raw and normalised momentum.
    """
    windows = [1, 3, 5, 10, 20, 60, 120, 252]
    for w in windows:
        df[f"return_{w}d"] = df["close"].pct_change(w)

    for w in [5, 10, 20, 60]:
        df[f"roc_{w}"] = safe_div(df["close"], df["close"].shift(w)) - 1

    # Raw momentum (price difference)
    df["momentum_5"]  = df["close"] - df["close"].shift(5)
    df["momentum_20"] = df["close"] - df["close"].shift(20)

    # Normalised momentum (same as roc, kept explicitly for clarity)
    df["norm_momentum_5"]  = safe_div(df["close"], df["close"].shift(5))  - 1
    df["norm_momentum_20"] = safe_div(df["close"], df["close"].shift(20)) - 1

    return df


def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 5 — Rolling volatility, ATR, average candle range.
    """
    log_ret = np.log(safe_div(df["close"], df["close"].shift(1)))

    for w in [5, 10, 20, 60]:
        df[f"rolling_volatility_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    atr14, tr = _compute_atr(df["high"], df["low"], df["close"], 14)
    atr20, _  = _compute_atr(df["high"], df["low"], df["close"], 20)

    df["true_range"]       = tr
    df["atr_14"]           = atr14
    df["atr_20"]           = atr20
    df["atr14_pct_close"]  = safe_div(atr14, df["close"])
    df["atr20_pct_close"]  = safe_div(atr20, df["close"])

    hl = df["high"] - df["low"]
    for w in [5, 20, 60]:
        df[f"avg_hl_range_{w}"] = hl.rolling(w).mean()

    df["avg_candle_range_20"] = hl.rolling(20).mean()

    return df


def _rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 6 — RSI at three periods plus derived signals.
    """
    df["rsi_7"]  = _compute_rsi(df["close"], 7)
    df["rsi_14"] = _compute_rsi(df["close"], 14)
    df["rsi_21"] = _compute_rsi(df["close"], 21)

    df["rsi14_change_1d"] = df["rsi_14"].diff(1)
    df["rsi14_change_5d"] = df["rsi_14"].diff(5)

    df["rsi14_below_30"]  = (df["rsi_14"] < 30).astype(float)
    df["rsi14_above_70"]  = (df["rsi_14"] > 70).astype(float)
    df["rsi14_mid_zone"]  = ((df["rsi_14"] >= 40) & (df["rsi_14"] <= 60)).astype(float)

    return df


def _macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 7 — MACD line, signal, histogram and normalised versions.
    """
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal

    df["macd_line"]           = macd_line
    df["macd_signal"]         = macd_signal
    df["macd_histogram"]      = macd_hist
    df["macd_line_pct"]       = safe_div(macd_line,   df["close"])
    df["macd_signal_pct"]     = safe_div(macd_signal, df["close"])
    df["macd_histogram_pct"]  = safe_div(macd_hist,   df["close"])
    df["macd_line_gt_signal"] = (macd_line > macd_signal).astype(float)
    df["macd_hist_increasing"]= (macd_hist > macd_hist.shift(1)).astype(float)

    return df


def _bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 8 — Bollinger Bands (20-day, 2 std) and positional features.
    """
    window = 20
    std_mult = 2

    mid   = df["close"].rolling(window).mean()
    std   = df["close"].rolling(window).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw    = safe_div(upper - lower, mid)
    pct_b = safe_div(df["close"] - lower, upper - lower)

    df["bb_middle"]       = mid
    df["bb_upper"]        = upper
    df["bb_lower"]        = lower
    df["bb_bandwidth"]    = bw
    df["bb_pct_b"]        = pct_b
    df["close_vs_bb_upper"]  = safe_div(df["close"], upper)  - 1
    df["close_vs_bb_lower"]  = safe_div(df["close"], lower)  - 1
    df["close_vs_bb_middle"] = safe_div(df["close"], mid)    - 1

    return df


def _stochastic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 9 — Stochastic %K (14) and %D (3-period SMA of %K).
    """
    k_period = 14
    d_period = 3

    lowest_low    = df["low"].rolling(k_period).min()
    highest_high  = df["high"].rolling(k_period).max()

    stoch_k = 100 * safe_div(df["close"] - lowest_low, highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()

    df["stoch_k"]         = stoch_k
    df["stoch_d"]         = stoch_d
    df["stoch_k_minus_d"] = stoch_k - stoch_d
    df["stoch_k_gt_d"]    = (stoch_k > stoch_d).astype(float)
    df["stoch_k_below_20"]= (stoch_k < 20).astype(float)
    df["stoch_k_above_80"]= (stoch_k > 80).astype(float)

    return df


def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 10 — Volume moving averages, ratios, change, and flags.
    """
    for w in [5, 20, 50, 100]:
        df[f"vol_sma_{w}"] = df["volume"].rolling(w).mean()

    df["volume_vs_sma20"]       = safe_div(df["volume"], df["vol_sma_20"])
    df["volume_vs_sma50"]       = safe_div(df["volume"], df["vol_sma_50"])
    df["volume_change_1d"]      = df["volume"].pct_change(1)
    df["volume_change_5d"]      = df["volume"].pct_change(5)
    df["volume_spike"]          = (df["volume"] > 2 * df["vol_sma_20"]).astype(float)
    df["up_day_high_vol"]       = (
        (df["close"] > df["open"]) & (df["volume"] > df["vol_sma_20"])
    ).astype(float)
    df["down_day_high_vol"]     = (
        (df["close"] < df["open"]) & (df["volume"] > df["vol_sma_20"])
    ).astype(float)
    df["return_x_vol_ratio"]    = df["daily_return"] * df["volume_vs_sma20"]

    return df


def _support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 11 — Previous rolling highs/lows and distance/breakout features.

    LOOKAHEAD PREVENTION:
    Rolling high/low values are shifted by 1 day (.shift(1)) before any
    comparison with today's Close, ensuring no future information leaks in.
    """
    windows = [20, 50, 100, 252]

    for w in windows:
        prev_high = df["high"].rolling(w).max().shift(1)
        prev_low  = df["low"].rolling(w).min().shift(1)

        df[f"prev_{w}d_high"] = prev_high
        df[f"prev_{w}d_low"]  = prev_low

        df[f"close_vs_prev_{w}d_high"] = safe_div(df["close"], prev_high) - 1
        df[f"close_vs_prev_{w}d_low"]  = safe_div(df["close"], prev_low)  - 1

    # Breakout / breakdown flags (only for 20 and 50 windows)
    for w in [20, 50]:
        prev_high = df[f"prev_{w}d_high"]
        prev_low  = df[f"prev_{w}d_low"]

        df[f"breakout_above_{w}d_high"] = (df["close"] > prev_high).astype(float)
        df[f"breakdown_below_{w}d_low"] = (df["close"] < prev_low).astype(float)

    return df


def _trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 12 — ADX, DI+, DI- (14-period) and threshold flags.
    """
    adx, plus_di, minus_di = _compute_adx(df["high"], df["low"], df["close"], 14)

    df["adx_14"]         = adx
    df["di_plus_14"]     = plus_di
    df["di_minus_14"]    = minus_di
    df["di_plus_gt_minus"] = (plus_di > minus_di).astype(float)
    df["adx_gt_20"]      = (adx > 20).astype(float)
    df["adx_gt_25"]      = (adx > 25).astype(float)

    return df


def _drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 13 — Rolling max/min close and distance from extremes.

    Note: rolling max/min here are NOT shifted — they represent 'where is
    today's close relative to the highest/lowest close seen in the past N days
    including today'. This is causal: we know today's close at end-of-day.
    """
    for w in [20, 60, 252]:
        roll_max = df["close"].rolling(w).max()
        roll_min = df["close"].rolling(w).min()

        df[f"rolling_max_close_{w}"]   = roll_max
        df[f"rolling_min_close_{w}"]   = roll_min
        df[f"drawdown_from_{w}d_high"] = safe_div(df["close"], roll_max) - 1
        df[f"distance_from_{w}d_low"]  = safe_div(df["close"], roll_min) - 1

    return df


def _vwap_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 14 — Daily VWAP approximations (NOT true intraday VWAP).

    Since only daily OHLCV data is available, these are rolling proxies only.
    Do not use these as exact intraday VWAP values.
    """
    typical_price   = (df["high"] + df["low"] + df["close"]) / 3
    weighted_close  = (df["high"] + df["low"] + 2 * df["close"]) / 4

    df["typical_price"]   = typical_price
    df["weighted_close"]  = weighted_close

    window = 20
    tp_vol_sum  = (typical_price * df["volume"]).rolling(window).sum()
    vol_sum     = df["volume"].rolling(window).sum()

    rolling_vwap = safe_div(tp_vol_sum, vol_sum)
    df["rolling_vwap_20_approx"]      = rolling_vwap
    df["close_vs_rolling_vwap_20"]    = safe_div(df["close"], rolling_vwap) - 1

    return df


# ── Per-symbol dispatcher ─────────────────────────────────────────────────────

def _compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 14 feature groups to a single-symbol DataFrame.
    Caller is responsible for ensuring df is sorted by date ascending.
    """
    # Suppress pandas PerformanceWarning about fragmented DataFrames —
    # we intentionally add ~170 columns sequentially; correctness is unaffected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df = _base_price_features(df)
        df = _candle_structure_features(df)
        df = _moving_average_features(df)
        df = _momentum_features(df)
        df = _volatility_features(df)
        df = _rsi_features(df)
        df = _macd_features(df)
        df = _bollinger_features(df)
        df = _stochastic_features(df)
        df = _volume_features(df)
        df = _support_resistance_features(df)
        df = _trend_strength_features(df)
        df = _drawdown_features(df)
        df = _vwap_proxy_features(df)
    # Defragment the DataFrame for downstream performance
    return df.copy()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main public entry point.

    Accepts a multi-symbol or single-symbol OHLCV DataFrame and returns it
    with all technical feature columns appended.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns (case-insensitive):
        date, symbol, open, high, low, close, volume

        If 'symbol' is absent (single-symbol use), a placeholder column
        'symbol' = 'UNKNOWN' is added automatically.

    Returns
    -------
    pd.DataFrame
        Original rows + ~150 feature columns.
        date and symbol columns are preserved.
        Rows with insufficient lookback history will have NaN in those features.
        Do NOT drop NaNs here — let the ML pipeline decide the strategy.

    Example
    -------
    >>> import pandas as pd
    >>> from src.features.technical_indicators import generate_ohlcv_features
    >>> raw = pd.read_parquet("data/raw/AAPL.parquet")
    >>> raw["symbol"] = "AAPL"
    >>> featured = generate_ohlcv_features(raw)
    >>> print(featured.shape)
    """
    df = _validate_and_normalize(df)

    # Allow single-symbol DataFrames (no 'symbol' column)
    single_symbol = "symbol" not in df.columns
    if single_symbol:
        df["symbol"] = "UNKNOWN"

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = (
            df.groupby("symbol", group_keys=False)
              .apply(_compute_all_features)
        )

    if single_symbol:
        result = result.drop(columns=["symbol"])

    return result.reset_index(drop=True)


# ── Backward-compatible aliases ───────────────────────────────────────────────

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias for generate_ohlcv_features().
    Kept so that src/main.py does not need to be changed.
    """
    return generate_ohlcv_features(df)


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Backward-compatible shim used by src/strategies/baseline_rsi.py.
    Adds a single 'rsi' column (RSI with the given period).
    """
    df = df.copy()
    df["rsi"] = _compute_rsi(df["close"], period)
    return df


# ── Optional smoke-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    # Synthetic OHLCV — 300 rows, two symbols
    rng  = np.random.default_rng(42)
    n    = 300
    syms = ["AAPL", "MSFT"]
    dfs  = []
    for sym in syms:
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        opens  = closes * (1 + rng.normal(0, 0.005, n))
        highs  = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, n))
        lows   = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, n))
        vols   = rng.integers(1_000_000, 10_000_000, n).astype(float)
        dates  = pd.date_range("2020-01-01", periods=n, freq="B")
        dfs.append(pd.DataFrame({
            "date": dates, "symbol": sym,
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": vols,
        }))

    raw = pd.concat(dfs, ignore_index=True)
    out = generate_ohlcv_features(raw)

    n_features = out.shape[1] - 7
    print(f"Input rows:      {len(raw)}")
    print(f"Output rows:     {len(out)}")
    print(f"Feature columns: {n_features}")
    print(f"Total columns:   {out.shape[1]}")
    nan_pct = out.isna().mean().mean() * 100
    print(f"Overall NaN pct: {nan_pct:.1f}%  (expected -- early rows lack lookback history)")
