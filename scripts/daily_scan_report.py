"""
daily_scan_report.py
--------------------
Daily S&P 500 market scanner. Reads the already-built feature panel
(data/processed/*_featured.parquet) and produces a human-readable daily
report (Turkish) in Markdown + HTML under reports/scans/.

This is DETECTION, not PREDICTION: no model, no forecast. It surfaces what
happened yesterday that deviates from each stock's own recent norm --
unusual volume, breakouts/breakdowns, RSI extremes, 52-week levels, and a
market regime gauge computed from SPY.

Rationale: the ML research phase of this repo showed that forecasting
15-day forward returns from daily OHLCV features has no exploitable edge
(see README). The feature pipeline itself, however, is a solid anomaly
scanner -- this script is the salvage of that work.

Usage:
  python scripts/daily_scan_report.py                # full universe
  python scripts/daily_scan_report.py --top 15       # longer lists
  python scripts/daily_scan_report.py --symbols AAPL MSFT NVDA SPY
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ROOT, REPORTS_DIR

PROCESSED_DIR = ROOT / "data" / "processed"
SCANS_DIR = REPORTS_DIR / "scans"

BENCHMARK = "SPY"

# Columns needed for the scan (subset of the 182-column featured parquet).
SCAN_COLUMNS = [
    "date", "symbol", "close", "volume",
    "daily_return", "return_5d", "return_20d", "return_60d",
    "volume_vs_sma20", "volume_change_1d", "volume_spike",
    "rsi_14",
    "breakout_above_20d_high", "breakdown_below_20d_low",
    "breakout_above_50d_high", "breakdown_below_50d_low",
    "close_vs_prev_252d_high", "close_vs_prev_252d_low",
    "drawdown_from_252d_high",
    "rolling_volatility_20", "atr14_pct_close", "adx_14",
    "close_gt_sma50", "close_gt_sma200",
]

DISCLAIMER = (
    "Bu rapor otomatik olarak uretilmis istatistiksel bir taramadir. "
    "Yatirim tavsiyesi degildir; alim/satim onerisi icermez."
)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_last_rows(symbols: list[str]) -> pd.DataFrame:
    """Read only the scan columns from each symbol's parquet, keep last row."""
    rows = []
    for sym in symbols:
        fp = PROCESSED_DIR / f"{sym}_featured.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp, columns=SCAN_COLUMNS)
        except Exception:
            # Column mismatch or corrupt file -- skip rather than crash.
            continue
        if df.empty:
            continue
        rows.append(df.iloc[-1])
    if not rows:
        raise SystemExit("No processed data found. Run daily_refresh.py first.")
    panel = pd.DataFrame(rows).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    # Keep only symbols whose data is current (== latest date in the panel).
    latest = panel["date"].max()
    stale = (panel["date"] != latest).sum()
    panel = panel[panel["date"] == latest].copy()
    panel.attrs["latest_date"] = latest
    panel.attrs["n_stale_dropped"] = int(stale)
    return panel


def load_regime(benchmark: str = BENCHMARK) -> dict:
    """Market regime gauge from the benchmark's full history."""
    fp = PROCESSED_DIR / f"{benchmark}_featured.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(
        fp,
        columns=["date", "close", "rolling_volatility_20",
                 "return_5d", "return_20d",
                 "close_gt_sma50", "close_gt_sma200",
                 "drawdown_from_252d_high"],
    )
    last = df.iloc[-1]
    # Percentile of current 20d volatility within the trailing year.
    trailing = df["rolling_volatility_20"].tail(252).dropna()
    vol_pct = float((trailing <= last["rolling_volatility_20"]).mean() * 100)
    if vol_pct < 40:
        label = "SAKIN"
    elif vol_pct < 75:
        label = "NORMAL"
    else:
        label = "CALKANTILI"
    return {
        "label": label,
        "vol_pct": vol_pct,
        "ret_5d": float(last["return_5d"]),
        "ret_20d": float(last["return_20d"]),
        "above_sma50": bool(last["close_gt_sma50"]),
        "above_sma200": bool(last["close_gt_sma200"]),
        "drawdown_252d": float(last["drawdown_from_252d_high"]),
    }


# ----------------------------------------------------------------------
# Scan sections
# ----------------------------------------------------------------------

def _fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def build_sections(panel: pd.DataFrame, top: int) -> list[tuple[str, pd.DataFrame, str]]:
    """Return list of (title, table, note) sections."""
    p = panel[panel["symbol"] != BENCHMARK].copy()
    sections: list[tuple[str, pd.DataFrame, str]] = []

    def table(df: pd.DataFrame, cols: dict[str, str], pct_cols: set[str]) -> pd.DataFrame:
        out = df[list(cols)].rename(columns=cols).copy()
        for c_src, c_new in cols.items():
            if c_src in pct_cols:
                out[c_new] = df[c_src].map(lambda v: _fmt_pct(float(v)))
        return out.reset_index(drop=True)

    # 1) Unusual volume
    vol = p[p["volume_vs_sma20"] >= 2.0].sort_values("volume_vs_sma20", ascending=False).head(top)
    if not vol.empty:
        t = table(vol, {"symbol": "Sembol", "volume_vs_sma20": "Hacim / 20g Ort.",
                        "daily_return": "Gunluk Getiri", "return_5d": "5g Getiri"},
                  {"daily_return", "return_5d"})
        t["Hacim / 20g Ort."] = vol["volume_vs_sma20"].map(lambda v: f"{v:.1f}x").values
        sections.append((
            "Anormal Hacim",
            t,
            "20 gunluk ortalamasinin en az 2 kati hacimle islem gorenler. "
            "Yuksek hacim + sert fiyat hareketi genelde yeni bir haber/gelisme isaretidir.",
        ))

    # 2) Breakouts (50d high with volume interest)
    br = p[p["breakout_above_50d_high"] == 1].sort_values("volume_vs_sma20", ascending=False).head(top)
    if not br.empty:
        t = table(br, {"symbol": "Sembol", "daily_return": "Gunluk Getiri",
                       "return_20d": "20g Getiri", "volume_vs_sma20": "Hacim Orani"},
                  {"daily_return", "return_20d"})
        t["Hacim Orani"] = br["volume_vs_sma20"].map(lambda v: f"{v:.1f}x").values
        sections.append((
            "Momentum: 50 Gunluk Zirve Kirilimlari",
            t,
            "Son 50 gunun en yuksek kapanisinin uzerine cikanlar.",
        ))

    # 3) Breakdowns
    bd = p[p["breakdown_below_50d_low"] == 1].sort_values("volume_vs_sma20", ascending=False).head(top)
    if not bd.empty:
        t = table(bd, {"symbol": "Sembol", "daily_return": "Gunluk Getiri",
                       "return_20d": "20g Getiri", "volume_vs_sma20": "Hacim Orani"},
                  {"daily_return", "return_20d"})
        t["Hacim Orani"] = bd["volume_vs_sma20"].map(lambda v: f"{v:.1f}x").values
        sections.append((
            "Zayiflik: 50 Gunluk Dip Kirilimlari",
            t,
            "Son 50 gunun en dusuk kapanisinin altina inenler.",
        ))

    # 4) Near 52-week high / low
    near_hi = p[p["drawdown_from_252d_high"] >= -0.01].sort_values(
        "return_20d", ascending=False).head(top)
    if not near_hi.empty:
        t = table(near_hi, {"symbol": "Sembol", "drawdown_from_252d_high": "Zirveye Uzaklik",
                            "return_20d": "20g Getiri", "return_60d": "60g Getiri"},
                  {"drawdown_from_252d_high", "return_20d", "return_60d"})
        sections.append((
            "52 Haftalik Zirvesinde / Cok Yakininda",
            t,
            "Son 1 yilin en yuksek seviyesinin %1 bandinda olanlar.",
        ))

    near_lo = p[p["close_vs_prev_252d_low"] <= 0.01].sort_values(
        "return_20d", ascending=True).head(top)
    if not near_lo.empty:
        t = table(near_lo, {"symbol": "Sembol", "drawdown_from_252d_high": "Zirveden Dusus",
                            "return_20d": "20g Getiri", "return_60d": "60g Getiri"},
                  {"drawdown_from_252d_high", "return_20d", "return_60d"})
        sections.append((
            "52 Haftalik Dibinde / Cok Yakininda",
            t,
            "Son 1 yilin en dusuk seviyesinin %1 bandinda olanlar.",
        ))

    # 5) RSI extremes
    oversold = p[p["rsi_14"] <= 30].sort_values("rsi_14").head(top)
    if not oversold.empty:
        t = table(oversold, {"symbol": "Sembol", "rsi_14": "RSI(14)",
                             "return_5d": "5g Getiri", "return_20d": "20g Getiri"},
                  {"return_5d", "return_20d"})
        t["RSI(14)"] = oversold["rsi_14"].map(lambda v: f"{v:.0f}").values
        sections.append((
            "Asiri Satim Bolgesi (RSI <= 30)",
            t,
            "Kisa vadede sert dusmus olanlar. Dusuk RSI tek basina 'al' sinyali degildir.",
        ))

    overbought = p[p["rsi_14"] >= 75].sort_values("rsi_14", ascending=False).head(top)
    if not overbought.empty:
        t = table(overbought, {"symbol": "Sembol", "rsi_14": "RSI(14)",
                               "return_5d": "5g Getiri", "return_20d": "20g Getiri"},
                  {"return_5d", "return_20d"})
        t["RSI(14)"] = overbought["rsi_14"].map(lambda v: f"{v:.0f}").values
        sections.append((
            "Asiri Alim Bolgesi (RSI >= 75)",
            t,
            "Kisa vadede cok hizli yukselmis olanlar.",
        ))

    # 6) Biggest single-day moves
    movers = p.reindex(p["daily_return"].abs().sort_values(ascending=False).index).head(top)
    t = table(movers, {"symbol": "Sembol", "daily_return": "Gunluk Getiri",
                       "volume_vs_sma20": "Hacim Orani", "rsi_14": "RSI(14)"},
              {"daily_return"})
    t["Hacim Orani"] = movers["volume_vs_sma20"].map(lambda v: f"{v:.1f}x").values
    t["RSI(14)"] = movers["rsi_14"].map(lambda v: f"{v:.0f}").values
    sections.append(("Gunun En Sert Hareketleri", t, "Mutlak gunluk getiriye gore."))

    return sections


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render_markdown(date: pd.Timestamp, regime: dict, sections, n_symbols: int) -> str:
    lines = [f"# Gunluk Piyasa Taramasi — {date.date()}", ""]
    if regime:
        trend = []
        trend.append("SMA50 ustunde" if regime["above_sma50"] else "SMA50 altinda")
        trend.append("SMA200 ustunde" if regime["above_sma200"] else "SMA200 altinda")
        lines += [
            f"**Piyasa Rejimi ({BENCHMARK}): {regime['label']}**",
            "",
            f"- 20 gunluk volatilite, son 1 yilin %{regime['vol_pct']:.0f} yuzdelik diliminde",
            f"- Son 5 gun {_fmt_pct(regime['ret_5d'])}, son 20 gun {_fmt_pct(regime['ret_20d'])}",
            f"- Trend: {', '.join(trend)}; 52h zirveden {_fmt_pct(regime['drawdown_252d'])}",
            "",
        ]
    lines.append(f"*Taranan sembol: {n_symbols} (S&P 500)*")
    lines.append("")
    for title, tbl, note in sections:
        lines += [f"## {title}", "", f"_{note}_", ""]
        lines.append(tbl.to_markdown(index=False))
        lines.append("")
    lines += ["---", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)


HTML_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:24px auto;
padding:0 16px;color:#1a1a2e;background:#fafafa}
h1{font-size:1.5em;border-bottom:2px solid #16213e;padding-bottom:8px}
h2{font-size:1.15em;color:#16213e;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:.92em;background:#fff}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}
th{background:#16213e;color:#fff}
tr:nth-child(even){background:#f4f6fa}
.regime{padding:10px 14px;border-radius:8px;display:inline-block;font-weight:600}
.SAKIN{background:#d4edda;color:#155724}
.NORMAL{background:#fff3cd;color:#856404}
.CALKANTILI{background:#f8d7da;color:#721c24}
.note{color:#666;font-style:italic;font-size:.9em}
.disclaimer{margin-top:32px;padding:10px;background:#eee;border-radius:6px;
font-size:.85em;color:#555}
"""


def render_html(date: pd.Timestamp, regime: dict, sections, n_symbols: int) -> str:
    parts = [f"<html><head><meta charset='utf-8'><style>{HTML_CSS}</style></head><body>"]
    parts.append(f"<h1>Gunluk Piyasa Taramasi — {date.date()}</h1>")
    if regime:
        parts.append(
            f"<p><span class='regime {regime['label']}'>Piyasa Rejimi: {regime['label']}</span></p>"
            f"<p class='note'>20g volatilite yillik %{regime['vol_pct']:.0f} dilimde · "
            f"5g {_fmt_pct(regime['ret_5d'])} · 20g {_fmt_pct(regime['ret_20d'])} · "
            f"52h zirveden {_fmt_pct(regime['drawdown_252d'])}</p>"
        )
    parts.append(f"<p class='note'>Taranan sembol: {n_symbols} (S&amp;P 500)</p>")
    for title, tbl, note in sections:
        parts.append(f"<h2>{title}</h2><p class='note'>{note}</p>")
        parts.append(tbl.to_html(index=False, border=0))
    parts.append(f"<div class='disclaimer'>{DISCLAIMER}</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Daily market scan report")
    ap.add_argument("--top", type=int, default=10, help="rows per section")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="override universe (default: every *_featured.parquet)")
    args = ap.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = sorted(fp.name.replace("_featured.parquet", "")
                         for fp in PROCESSED_DIR.glob("*_featured.parquet"))
    print(f"[scan] loading last rows for {len(symbols)} symbols ...")
    panel = load_last_rows(symbols)
    date = panel.attrs["latest_date"]
    if panel.attrs["n_stale_dropped"]:
        print(f"[scan] dropped {panel.attrs['n_stale_dropped']} stale symbols "
              f"(data older than {date.date()})")

    regime = load_regime()
    sections = build_sections(panel, args.top)

    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    md = render_markdown(date, regime, sections, len(panel))
    html = render_html(date, regime, sections, len(panel))
    md_path = SCANS_DIR / f"scan_{date.date()}.md"
    html_path = SCANS_DIR / f"scan_{date.date()}.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    # Stable "latest" copies for easy linking / automation.
    (SCANS_DIR / "latest.md").write_text(md, encoding="utf-8")
    (SCANS_DIR / "latest.html").write_text(html, encoding="utf-8")
    print(f"[scan] wrote {md_path}")
    print(f"[scan] wrote {html_path}")


if __name__ == "__main__":
    main()
