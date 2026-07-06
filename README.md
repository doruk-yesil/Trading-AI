# Trading-AI

ML trading-signal research on the S&P 500 — with an honest negative result — plus the useful tool that survived it: an automated **daily market scanner**.

This repo did the experiment properly (purged cross-validation, embargo periods, sample-uniqueness weighting, realistic cost assumptions, per-regime evaluation) and reached the conclusion most quant literature predicts for daily public OHLCV data: **there is no exploitable edge here**. Rather than hide that, the repo documents it — and repurposes the data/feature pipeline into a daily anomaly scanner that produces something actually useful every morning.

---

## TL;DR — Research Findings

**Question:** Can gradient-boosted models predict 15-day forward returns across the S&P 500 from daily price/volume features well enough to trade profitably?

**Setup:** 504 symbols, pooled cross-sectional panel of **4.38M rows**, **173 engineered features**, continuous forward-return target, evaluated by Information Coefficient (Spearman rank correlation between predicted and realized returns) — not MAE/RMSE, since ranking is what a cross-sectional strategy actually uses.

**Result (held-out test period):**

| Model | Test IC | Directional acc. | Quarterly IC mean ± std |
|---|---|---|---|
| Linear | -0.022 | 52.1% | -0.040 ± 0.098 |
| Random Forest | -0.001 | 51.8% | -0.004 ± 0.032 |
| LightGBM | +0.008 | 55.3% | +0.003 ± 0.047 |
| XGBoost | **+0.010** | 55.3% | +0.009 ± 0.044 |
| Ensemble (avg of 4) | -0.010 | 53.8% | -0.021 ± 0.072 |

**Interpretation:**

- The best model's aggregate test IC (~0.01) is statistically distinguishable from zero on 4.38M rows, but **economically negligible** — well below what survives realistic transaction costs (0.2% round trip) at a 15-day horizon.
- The per-quarter breakdown is the decisive evidence: quarterly IC swings between roughly -0.08 and +0.06 with a mean statistically indistinguishable from zero. The "signal" is regime noise, not a stable edge.
- Directional accuracy (~55%) is inflated by market drift during the test period — always predicting "up" performs similarly in a rising market. IC is the honest metric.
- The ensemble underperforms its best members because it averages in the weak linear model.

**Why this outcome was likely a priori:** daily OHLCV data from Yahoo Finance is free and universally available; whatever predictive structure it once contained has been arbitraged away. The bottleneck is not the model class (gradient boosting is state of the art for tabular data) — it is the information content of the input. A negative result, properly established, is still a result.

---

## What Survived: The Daily Market Scanner

The data pipeline, feature engineering, and automation were solid — so they now power `scripts/daily_scan_report.py`, which is **detection, not prediction**. No model, no forecast; it surfaces what deviated from each stock's own recent norm:

- **Market regime gauge** (from SPY): current 20-day volatility as a percentile of the trailing year → CALM / NORMAL / TURBULENT
- **Unusual volume**: stocks trading ≥ 2x their 20-day average volume
- **Momentum breakouts / breakdowns**: closes above 50-day highs / below 50-day lows
- **52-week high/low proximity**
- **RSI extremes** (≤ 30 oversold, ≥ 75 overbought)
- **Largest single-day moves**

Output: Markdown + self-contained HTML under `reports/scans/` (dated file + `latest.*`), currently written in Turkish. Runs automatically as the final stage of the daily refresh.

```bash
python scripts/daily_scan_report.py               # full universe
python scripts/daily_scan_report.py --top 15      # longer lists
python scripts/daily_scan_report.py --symbols AAPL MSFT NVDA SPY
```

### Live report on GitHub Pages (no server needed)

`.github/workflows/daily-scan.yml` runs the whole thing on GitHub's servers every weekday after US market close — downloads the last 2 years of data (enough for every feature window), rebuilds features, generates the report, and publishes it as a static page.

One-time setup after pushing to GitHub:

1. Repo **Settings → Pages → Source: "GitHub Actions"**
2. Test it: **Actions tab → "Daily market scan" → Run workflow**
3. Report appears at `https://<username>.github.io/<repo>/`

If Yahoo rate-limits GitHub's IPs (occasional 429s), the local Task Scheduler pipeline is the fallback — it produces the identical `reports/scans/latest.html`.

---

## Pipeline Overview

```
Yahoo Finance ──> data/raw/ ──> feature engineering ──> data/processed/*_featured.parquet
   (daily, 504 symbols,             (173 features,              │
    max history)                     causal windows)            ├──> model training / evaluation (research)
                                                                └──> daily scan report (product)
```

`scripts/daily_refresh.py` runs the full cycle (download → hourly archive accumulation → feature rebuild → scan report) and is designed for Windows Task Scheduler. Hourly bars are archived separately in `data/hourly/` because Yahoo only serves a rolling ~730-day hourly window — the archive grows into a dataset you cannot re-download later.

---

## Methodology Highlights (the research part worth reusing)

1. **Purged train/val/test split with embargo** — rows whose 15-day label windows overlap a boundary are dropped, preventing label leakage across splits.
2. **Purged k-fold cross-validation** (`scripts/cross_validate.py`) — consistency check across multiple time periods, same purging logic per fold.
3. **Per-quarter performance breakdown** — a single aggregate IC hides regime dependence; quarterly IC is what exposed the instability here.
4. **Sample-uniqueness weighting** (López de Prado) — overlapping 15-day labels are not i.i.d.; heavily overlapping rows are down-weighted.
5. **Model ensemble** — average of linear / RF / LightGBM / XGBoost predictions.
6. **Feature importance analysis** (`scripts/analyze_feature_importance.py`) — rank and prune the 173 features without retraining.
7. **Meta-labeling** (`scripts/train_meta_model.py`) — a secondary model that learns when to trust the primary model's signal.

---

## Project Structure

```
trading-ai/
├── configs/config.yaml            ← all parameters
├── data/                          ← not in git
│   ├── raw/                       ← daily OHLCV per symbol
│   ├── hourly/                    ← accumulating 1h archive (future use)
│   └── processed/                 ← *_featured.parquet (182 cols)
├── reports/                       ← not in git
│   ├── scans/                     ← daily scan reports (md + html)
│   ├── experiments/               ← JSON training results
│   └── models/                    ← saved models
├── scripts/
│   ├── daily_refresh.py           ← full daily cycle (schedule this)
│   ├── daily_scan_report.py       ← market scanner (the product)
│   ├── download_yahoo_finance.py  ← daily OHLCV downloader
│   ├── download_hourly_data.py    ← hourly archive accumulator
│   ├── build_features.py          ← feature panel builder
│   ├── train_model.py             ← train + compare all models
│   ├── cross_validate.py          ← purged k-fold CV
│   ├── analyze_feature_importance.py
│   ├── train_meta_model.py        ← meta-labeling
│   └── predict.py
└── src/
    ├── data/                      ← load / validate / preprocess
    ├── features/                  ← technical indicators, target builder
    ├── models/                    ← training, evaluation, CV, meta-labeling
    ├── backtest/                  ← engine, portfolio, cost-aware metrics
    ├── strategies/                ← MA / RSI baselines, ML signal strategy
    └── utils/                     ← config, paths, universe resolution
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate                  # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python scripts/check_env.py

# First full data pull + features + scan (takes a while: 504 symbols, max history)
python scripts/daily_refresh.py
```

### Scheduling the daily refresh (Windows Task Scheduler)

```cmd
schtasks /create /sc daily /tn "TradingAI-DailyRefresh" ^
  /tr "<repo-path>\scripts\run_daily_refresh.bat" ^
  /st 23:30
```

Check runs via `reports/refresh_logs/`. Remove with `schtasks /delete /tn "TradingAI-DailyRefresh" /f`.

Useful flags: `--skip-hourly`, `--skip-features`, `--skip-scan`, `--symbols AAPL MSFT SPY` (quick test).

---

## Key Design Principles

**Lookahead bias prevention.** Every feature at time T uses only data up to T; the target uses future prices, so label-window rows at split boundaries are purged and an embargo applied. Chronological splits only — never random shuffling on time series.

**Ranking over regression error.** The strategy layer ranks stocks cross-sectionally, so evaluation is Spearman IC, not MAE/RMSE. A model can have great MAE and zero ranking power.

**Realistic costs.** Config assumes 0.1% one-way commission + 0.05% slippage. Any signal that only works at zero cost is not a signal.

**Regime honesty.** Aggregate metrics are reported alongside per-quarter breakdowns; an edge that flips sign across quarters is treated as no edge.

---

## Limitations

- Price/volume features only — no fundamentals, macro, or alternative data (this is, in effect, the finding: that's not enough).
- Survivorship bias: universe is the *current* S&P 500 membership, so delisted losers are absent. This flatters results — which makes the negative result stronger, not weaker.
- No short-selling modeling, no options/futures.
- Yahoo Finance data quality is retail-grade (splits/dividends occasionally revised).

---

## Configuration

Key settings in `configs/config.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `universe_mode` | `sp500` | Live S&P 500 constituents + SPY benchmark |
| `prediction_horizon` | 15 | Trading days forward (research target) |
| `target_type` | `continuous` | Predict forward return, evaluate by IC |
| `transaction_cost_pct` | 0.001 | 0.1% per trade one-way |
| `slippage_pct` | 0.0005 | 0.05% per side |
| `use_sample_weighting` | true | López de Prado uniqueness weights |

---

*Research project. Nothing here is financial advice; the scan report explicitly carries the same disclaimer. If anything, the headline finding is a reason **not** to trade on models like this.*
