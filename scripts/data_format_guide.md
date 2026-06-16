# Data Format Guide — For the Data Collection Teammate

This document specifies the exact format the Trading-AI pipeline expects for raw historical US stock data.

---

## TL;DR

- One file per symbol (e.g. `AAPL.csv` or `AAPL.parquet`)
- Place files in: `data/raw/`
- Required columns: `date, open, high, low, close, volume`
- Date format: `YYYY-MM-DD`
- Prices: **adjusted** for splits and dividends (adjusted close is the most important)
- No duplicates, no missing rows in the middle of the series

---

## File Format

### Preferred: Parquet
```
data/raw/AAPL.parquet
data/raw/MSFT.parquet
data/raw/SPY.parquet    ← benchmark, required
...
```
Parquet is faster and more compact than CSV. Use `df.to_parquet("AAPL.parquet")` in pandas.

### Acceptable: CSV
```
data/raw/AAPL.csv
data/raw/MSFT.csv
...
```

---

## Required Columns

| Column   | Type    | Description                                              |
|----------|---------|----------------------------------------------------------|
| `date`   | string  | Trading date in `YYYY-MM-DD` format                     |
| `open`   | float   | Adjusted opening price                                   |
| `high`   | float   | Adjusted intraday high                                   |
| `low`    | float   | Adjusted intraday low                                    |
| `close`  | float   | **Adjusted** closing price (split + dividend adjusted)   |
| `volume` | integer | Daily traded volume (shares)                             |

Column names must be **lowercase**.

### Example CSV header and first rows:
```
date,open,high,low,close,volume
2015-01-02,109.33,111.44,108.84,109.33,53204626
2015-01-05,108.40,108.65,105.41,106.25,64285491
2015-01-06,106.54,107.43,104.08,105.97,65797116
```

---

## Symbols Required (v1 Universe)

| Symbol | Description              |
|--------|--------------------------|
| AAPL   | Apple                    |
| MSFT   | Microsoft                |
| GOOGL  | Alphabet (Google)        |
| AMZN   | Amazon                   |
| NVDA   | NVIDIA                   |
| META   | Meta                     |
| JPM    | JPMorgan Chase           |
| SPY    | S&P 500 ETF (benchmark)  |

SPY is **mandatory** — it is used as a market benchmark in feature engineering.

---

## Date Range

- Start: `2015-01-01`
- End:   `2024-12-31`
- The more history the better (going back to 2010 is fine too).

---

## Recommended Data Sources

| Source              | Notes                                                       |
|---------------------|-------------------------------------------------------------|
| `yfinance` (Python) | Free, easy. `yf.download("AAPL", start="2015-01-01")`      |
| Stooq               | Free historical data, CSV download                          |
| Alpha Vantage       | Free tier has rate limits                                   |
| Tiingo              | Reliable adjusted data, free tier available                 |

### Quick collection script (yfinance):
```python
import yfinance as yf
import pandas as pd
from pathlib import Path

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "SPY"]
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for sym in SYMBOLS:
    df = yf.download(sym, start="2015-01-01", end="2024-12-31", auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_parquet(OUT_DIR / f"{sym}.parquet", index=False)
    print(f"Saved {sym}: {len(df)} rows")
```

---

## Common Mistakes to Avoid

- **Unadjusted prices**: Using raw prices without split/dividend adjustment breaks all momentum and return calculations.
- **Mixed time zones**: All dates should be NYSE market dates (Eastern Time), no time component needed.
- **Survivorship bias**: Do not exclude delisted stocks if you later expand the universe. For v1 (liquid large caps only), this is fine.
- **Future leak**: Do not include earnings dates, analyst ratings, or any data that was not available at market close on that date.
- **Wrong volume units**: Volume should be in shares, not lots or thousands.

---

## Validation

Once files are in `data/raw/`, run:
```
python scripts/check_env.py
python src/main.py --mode data
```

The pipeline will print warnings for any data quality issues found.
