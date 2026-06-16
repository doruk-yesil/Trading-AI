"""
config.py
---------
Loads and exposes the project configuration from configs/config.yaml.

Usage:
    from src.utils.config import cfg

    symbols = cfg["symbols"]
    horizon = cfg["prediction_horizon"]
"""

import yaml
from pathlib import Path
from src.utils.paths import CONFIG_FILE


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Load YAML config file and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Module-level singleton — import this directly
cfg: dict = load_config()


if __name__ == "__main__":
    import json
    print(json.dumps(cfg, indent=2, default=str))
