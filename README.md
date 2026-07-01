# Trading-AI

A research platform for developing and backtesting machine learning-based trading signals on US equities. This is **not** a live trading system — the goal is to research whether ML-derived signals can generate risk-adjusted alpha after realistic transaction costs.

---

## Project Goal

Build a rigorous, reproducible pipeline that:
1. Ingests historical OHLCV data for a small universe of liquid US stocks
2. Engineers technical features (momentum, mean-reversion, volatility, volume)
3. Trains supervised ML classifiers to predict forward return signals
4. Backtests all strategies with realistic transaction costs and slippage
5. Saves every experiment result for comparison

**What this is not:**
- Not a live trading bot
- Not a crypto trading system
- Not a reinforcement learning agent (yet)
- Not a get-rich-quick system — performance claims are unreliable until rigorously validated out-of-sample

---

## Current Status

**Phase 1 — Data pipeline automated, baselines/ML not yet run.**

Daily OHLCV data collection is live (İlke's original `scripts/download_yahoo_finance.py`,
now extended). `data/raw/` and `data/processed/` are populated end-to-end. As of
2026-07-02 the universe was expanded from a fixed 8-symbol list to the full
S&P 500 (fetched live at runtime, see `src/utils/universe.py`), pulling max
available history per symbol instead of a fixed 2015-2024 window. A daily
automated refresh (`scripts/daily_refresh.py`) re-downloads daily data,
accumulates hourly bars into a separate `data/hourly/` archive for future
use, and rebuilds `data/processed/*_featured.parquet` -- see "Automated
Daily Refresh" below to schedule it.

Note: the current 173-column feature set and 5-day binary target are
placeholders (mock) and expected to be redefined -- this doesn't block the
data pipeline, which regenerates whatever `build_features.py` currently
produces regardless of how the feature/target definitions change later.

See `scripts/data_format_guide.md` for the exact data format your data collection teammate should deliver.

---

## Staged Roadmap

### Phase 1 — Data & Baselines (current)
- [x] Collect historical OHLCV data for the S&P 500 universe + SPY benchmark (automated daily)
- [ ] Validate and preprocess data
- [ ] Build all technical features
- [ ] Run and evaluate MA crossover and RSI baseline strategies
- [ ] Establish benchmark equity curves

### Phase 2 — Supervised ML (next)
- [ ] Build binary classification targets (5-day forward return > hurdle)
- [ ] Train LightGBM and XGBoost models with walk-forward validation
- [ ] Compare ML signal strategy vs. rule-based baselines
- [ ] Feature importance analysis
- [ ] Tune prediction threshold on validation set

### Phase 3 — Walk-Forward & Portfolio (later)
- [ ] Implement proper walk-forward retraining (quarterly)
- [ ] Multi-stock portfolio backtest with equal weighting
- [ ] Risk metrics: Sharpe, Sortino, max drawdown, Calmar ratio
- [ ] Transaction cost sensitivity analysis

### Phase 4 — Deep Learning (future, GPU)
- [ ] LSTM / GRU sequence models
- [ ] Transformer Encoder approach
- [ ] Leverage NVIDIA RTX 3080 for training
- [ ] Compare vs. gradient boosting baseline

### Phase 5 — Live Paper Trading (future)
- [ ] Paper trading integration (simulated, no real money)
- [ ] Real-time data feed evaluation

---

## Project Structure

```
trading-ai/
├── data/
│   ├── raw/                  ← Raw OHLCV files (not in git)
│   ├── processed/            ← Cleaned + aligned data (not in git)
│   └── external/             ← External data (macro, sector, etc.)
├── notebooks/                ← Exploration notebooks
├── reports/
│   ├── figures/              ← Generated charts
│   ├── backtests/            ← Backtest output files
│   └── experiments/          ← JSON experiment results
├── scripts/
│   ├── check_env.py          ← Environment verification
│   └── data_format_guide.md  ← Data format spec for teammate
├── src/
│   ├── data/
│   │   ├── load_data.py      ← OHLCV file loader
│   │   ├── validate_data.py  ← Data quality checks
│   │   └── preprocess.py     ← Cleaning and alignment
│   ├── features/
│   │   ├── technical_indicators.py  ← All TA features
│   │   └── target_builder.py        ← Forward return labels
│   ├── strategies/
│   │   ├── baseline_ma.py          ← MA crossover baseline
│   │   ├── baseline_rsi.py         ← RSI baseline
│   │   └── ml_signal_strategy.py   ← ML probability → signal
│   ├── models/
│   │   ├── train_baseline_models.py ← Logistic, RandomForest
│   │   ├── train_xgboost.py         ← LightGBM + XGBoost
│   │   └── evaluate_model.py        ← Classification metrics
│   ├── backtest/
│   │   ├── engine.py         ← Single-asset backtest
│   │   ├── metrics.py        ← Sharpe, drawdown, etc.
│   │   └── portfolio.py      ← Multi-stock portfolio
│   ├── utils/
│   │   ├── config.py         ← Config loader
│   │   └── paths.py          ← Centralised path constants
│   └── main.py               ← Pipeline entry point
├── configs/
│   └── config.yaml           ← All parameters
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment
python scripts/check_env.py

# 4. Add data (see scripts/data_format_guide.md)
# Place AAPL.parquet, MSFT.parquet, ... SPY.parquet in data/raw/

# 5. Run pipeline
python src/main.py --mode all
```

---

## Running Individual Stages

```bash
# Data validation only
python src/main.py --mode data

# Feature engineering only
python src/main.py --mode features

# Train a specific model
python src/main.py --mode train --model lightgbm
python src/main.py --mode train --model baseline_ma

# Full pipeline on a subset of symbols
python src/main.py --mode all --symbols AAPL MSFT SPY
```

---

## Automated Daily Refresh

`scripts/daily_refresh.py` runs the full data cycle in one go:

1. `scripts/download_yahoo_finance.py` -- re-downloads daily OHLCV for the
   whole universe (S&P 500 + SPY), overwriting `data/raw/` with max
   available history per symbol.
2. `scripts/download_hourly_data.py` -- accumulates 1h bars into a separate
   `data/hourly/` archive (merges + de-dupes on top of what's already
   saved, since Yahoo only serves a rolling ~730-day window for hourly
   data). Not used by the current pipeline -- pure archival for later.
3. `scripts/build_features.py` -- rebuilds `data/processed/*_featured.parquet`
   from the fresh daily data.

Run it manually any time:
```bash
python scripts/daily_refresh.py
python scripts/daily_refresh.py --symbols AAPL MSFT SPY   # quick test on a few symbols
```

### Scheduling it (Windows Task Scheduler)

This needs to run somewhere with real internet access to Yahoo Finance --
your own machine (where `venv` already works), not a sandboxed environment.
One-time setup, run once in Command Prompt (adjust the time -- e.g. after US
market close in your timezone):

```cmd
schtasks /create /sc daily /tn "TradingAI-DailyRefresh" ^
  /tr "C:\Users\doruk\Desktop\AI-Project\Trading-AI\scripts\run_daily_refresh.bat" ^
  /st 23:30
```

Check it ran: `reports/refresh_logs/refresh_<timestamp>.log` (structured
summary) and `reports/refresh_logs/bat_output.log` (raw console output).
Remove it later with `schtasks /delete /tn "TradingAI-DailyRefresh" /f`.

---

## Key Design Principles

**Lookahead bias prevention:** The target uses future prices, so the last N rows of each series must be dropped before training. Features are computed causally — every feature at time T uses only data up to T. The train/test split always respects chronological order; no random shuffling.

**Survivorship bias awareness:** The v1 universe uses only currently-liquid large caps, which introduces mild survivorship bias. This is acceptable for a research baseline but must be noted when interpreting results.

**Realistic transaction costs:** Every backtest applies round-trip transaction costs (0.2% default: 0.1% one-way + 0.05% slippage each way). Strategies that only work with zero-cost assumptions are not useful.

**Evaluation hierarchy:** ML metrics (AUC, F1) come first, financial metrics (Sharpe, max drawdown) come second, and out-of-sample backtest performance is the final arbiter. A high AUC does not guarantee profitability.

---

## Current Limitations

- No live data feed
- No macro or fundamental data (pure price/volume features only)
- Single-stock backtest engine (portfolio engine is a placeholder)
- Walk-forward retraining is a placeholder (simple time split in v1)
- No short selling
- No options, futures, or other derivatives
- Small universe — results may not generalise to other stocks
- Past backtest performance does not predict future live performance

---

## Configuration

All parameters live in `configs/config.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `symbols` | 7 US stocks + SPY | Universe |
| `prediction_horizon` | 5 | Forward-looking days for target |
| `transaction_cost_pct` | 0.001 | 0.1% per trade one-way |
| `initial_capital` | 100,000 | USD |
| `train_years` | 4 | Training window |

---

*This is a research project. Nothing here constitutes financial advice.*
