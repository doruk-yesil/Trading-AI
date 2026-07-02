"""
paths.py
--------
Centralised path resolution for the Trading-AI project.

All modules should import paths from here — never hardcode paths elsewhere.
This keeps the project portable across different machines and working directories.
"""

from pathlib import Path

# Project root is two levels up from this file (src/utils/paths.py → root)
ROOT = Path(__file__).resolve().parents[2]

# ── Data ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
EXTERNAL_DATA_DIR  = ROOT / "data" / "external"

# Hourly OHLCV archive — kept separate from daily raw data.
# Not used by the current pipeline; accumulated for future (deep learning /
# intraday-feature) use since Yahoo Finance only serves a rolling ~730-day
# window for 1h bars. See scripts/download_hourly_data.py.
HOURLY_DATA_DIR    = ROOT / "data" / "hourly"

# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR     = ROOT / "reports"
FIGURES_DIR     = ROOT / "reports" / "figures"
BACKTESTS_DIR   = ROOT / "reports" / "backtests"
EXPERIMENTS_DIR = ROOT / "reports" / "experiments"
MODELS_DIR      = ROOT / "reports" / "models"   # trained model artifacts, one subfolder per training run

# ── Config ────────────────────────────────────────────────────────────────────
CONFIGS_DIR  = ROOT / "configs"
CONFIG_FILE  = CONFIGS_DIR / "config.yaml"

# ── Notebooks ─────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR = ROOT / "notebooks"


def ensure_dirs() -> None:
    """Create all output directories if they don't exist yet."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        HOURLY_DATA_DIR,
        FIGURES_DIR,
        BACKTESTS_DIR,
        EXPERIMENTS_DIR,
        MODELS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {ROOT}")
    print("All directories verified.")
