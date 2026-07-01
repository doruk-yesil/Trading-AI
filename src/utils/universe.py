"""
universe.py
-----------
Resolves the trading universe (list of ticker symbols) the pipeline should
pull/process, so this never has to be hand-maintained in config.yaml.

Modes (configs/config.yaml -> universe_mode):
  "sp500"  - fetch the *current* live S&P 500 constituent list from Wikipedia
             at runtime, plus always-include extras from cfg["symbols"]
             (e.g. SPY, which is a benchmark ETF, not an index constituent).
             Falls back to a local cache if the live fetch fails (no internet,
             Wikipedia layout change, etc.).
  "manual" - use exactly cfg["symbols"], no live fetch.

Why fetch live instead of hardcoding 500 tickers here: S&P 500 constituents
change several times a year (additions/removals). Hardcoding a list here
would silently go stale. Fetching at runtime means whoever actually *runs*
the downloader always gets today's real list.

Note: this module intentionally does the network call at call-time, not at
import time -- import must stay side-effect free / offline-safe.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.utils.paths import CONFIGS_DIR
from src.utils.config import cfg

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Plain-CSV fallback if Wikipedia blocks/changes layout -- community-maintained,
# updated periodically, no scraping/parsing fragility.
SP500_CSV_FALLBACK_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)
SP500_CACHE_PATH = CONFIGS_DIR / "sp500_constituents_cache.csv"

# Wikipedia (and some CDNs) return HTTP 403 for requests with no / a generic
# urllib User-Agent -- pandas.read_html(url) hits this every time. Fetch the
# page ourselves with a browser-like User-Agent, then hand the HTML text to
# pandas.read_html() instead of a bare URL.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 15


def _sanitize_yahoo_ticker(raw_symbol: str) -> str:
    """
    Wikipedia lists share classes with a dot (e.g. BRK.B, BF.B).
    Yahoo Finance / yfinance expects a dash instead (BRK-B, BF-B).
    """
    return raw_symbol.strip().upper().replace(".", "-")


def _find_symbol_column(table: pd.DataFrame) -> str:
    for candidate in ("Symbol", "Ticker symbol", "Ticker"):
        if candidate in table.columns:
            return candidate
    raise ValueError(f"Could not find a symbol column in {list(table.columns)}")


def _fetch_from_wikipedia() -> list[str]:
    """
    pandas.read_html(url) fetches with a bare urllib User-Agent, which
    Wikipedia rejects with HTTP 403. Fetch the HTML ourselves with a
    browser-like header instead, then parse the text.
    """
    resp = requests.get(SP500_WIKI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = tables[0]
    symbol_col = _find_symbol_column(table)
    return sorted({_sanitize_yahoo_ticker(s) for s in table[symbol_col].astype(str)})


def _fetch_from_csv_fallback() -> list[str]:
    resp = requests.get(SP500_CSV_FALLBACK_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    table = pd.read_csv(io.StringIO(resp.text))
    symbol_col = _find_symbol_column(table)
    return sorted({_sanitize_yahoo_ticker(s) for s in table[symbol_col].astype(str)})


def fetch_sp500_symbols(use_cache_on_failure: bool = True) -> list[str]:
    """
    Fetch the current S&P 500 constituent list.

    Tries Wikipedia first, then a community-maintained GitHub CSV mirror,
    then falls back to the local cache from the last successful fetch.

    Returns
    -------
    list[str]
        Sorted, deduplicated, Yahoo-Finance-formatted ticker symbols.

    Raises
    ------
    RuntimeError
        If every live source fails AND no local cache is available
        (or use_cache_on_failure is False).
    """
    errors = []
    for source_name, fetch_fn in (("Wikipedia", _fetch_from_wikipedia), ("GitHub CSV mirror", _fetch_from_csv_fallback)):
        try:
            symbols = fetch_fn()
            SP500_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(symbols, name="symbol").to_csv(SP500_CACHE_PATH, index=False)
            return symbols
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            print(f"WARNING: {source_name} fetch failed ({exc}).")

    if use_cache_on_failure and SP500_CACHE_PATH.exists():
        cached = pd.read_csv(SP500_CACHE_PATH)["symbol"].tolist()
        print(f"  Falling back to cached list ({len(cached)} symbols): {SP500_CACHE_PATH}")
        return cached

    raise RuntimeError(
        "Could not fetch S&P 500 list from any source (no internet?) and no local cache "
        f"exists at {SP500_CACHE_PATH}.\n  Errors: {'; '.join(errors)}\n"
        "Run this once with a working internet connection first, "
        "or switch configs/config.yaml universe_mode to 'manual'."
    )


def resolve_universe(
    extra_symbols: Optional[list[str]] = None,
    override_symbols: Optional[list[str]] = None,
) -> list[str]:
    """
    Resolve the final list of symbols the pipeline should operate on.

    Parameters
    ----------
    extra_symbols : list[str], optional
        Symbols to always include on top of the resolved universe
        (defaults to cfg["symbols"], e.g. the SPY benchmark).
    override_symbols : list[str], optional
        If given (e.g. from a CLI --symbols flag), bypasses universe_mode
        entirely and returns exactly this list. Lets you still run
        `--symbols AAPL MSFT` for quick tests without pulling all 500.

    Returns
    -------
    list[str]
    """
    if override_symbols:
        return sorted({_sanitize_yahoo_ticker(s) for s in override_symbols})

    mode = cfg.get("universe_mode", "manual")
    always_include = extra_symbols if extra_symbols is not None else cfg.get("symbols", [])
    always_include = [_sanitize_yahoo_ticker(s) for s in always_include]

    if mode == "sp500":
        symbols = fetch_sp500_symbols()
    elif mode == "manual":
        symbols = []
    else:
        raise ValueError(f"Unknown universe_mode: {mode!r} (expected 'sp500' or 'manual')")

    combined = sorted(set(symbols) | set(always_include))
    return combined


if __name__ == "__main__":
    universe = resolve_universe()
    print(f"Resolved universe: {len(universe)} symbols")
    print(universe[:20], "..." if len(universe) > 20 else "")
