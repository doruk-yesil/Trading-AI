"""
check_env.py
------------
Verifies that the Python environment is correctly set up for Trading-AI.

Checks:
  1. Python version >= 3.10
  2. Required packages installed and importable
  3. CUDA / GPU availability (optional — project works on CPU)

Run:
  python scripts/check_env.py
"""

import sys
import importlib


# ── 1. Python version ─────────────────────────────────────────────────────────

MIN_PYTHON = (3, 10)

def check_python():
    v = sys.version_info
    ok = v >= MIN_PYTHON
    status = "✓" if ok else "✗"
    print(f"  {status}  Python {v.major}.{v.minor}.{v.micro}  (required >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    if not ok:
        print("     → Please upgrade Python.")
    return ok


# ── 2. Required packages ──────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("pyarrow",     "pyarrow"),
    ("yfinance",    "yfinance"),
    ("yaml",        "pyyaml"),
    ("sklearn",     "scikit-learn"),
    ("lightgbm",    "lightgbm"),
    ("xgboost",     "xgboost"),
    ("matplotlib",  "matplotlib"),
    ("seaborn",     "seaborn"),
    ("ta",          "ta"),
    ("tqdm",        "tqdm"),
]

def check_packages():
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✓  {pip_name:<20} {version}")
        except ImportError:
            print(f"  ✗  {pip_name:<20} NOT INSTALLED  →  pip install {pip_name}")
            all_ok = False
    return all_ok


# ── 3. GPU / CUDA ─────────────────────────────────────────────────────────────

def check_gpu():
    print("\n  GPU / CUDA check (optional):")

    # Check via torch if installed
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓  PyTorch CUDA available  →  {device_name}")
        else:
            print("  –  PyTorch installed but CUDA not available (CPU-only mode).")
        return cuda_available
    except ImportError:
        print("  –  PyTorch not installed (not required for v1 — CPU mode active).")
        print("     To enable GPU later:  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False


# ── 4. LightGBM GPU ───────────────────────────────────────────────────────────

def check_lightgbm_gpu():
    try:
        import lightgbm as lgb
        import numpy as np

        # Quick GPU compile check — will fail gracefully if no GPU build
        data = lgb.Dataset(np.random.rand(10, 3), label=np.random.randint(0, 2, 10))
        params = {"objective": "binary", "device": "gpu", "verbose": -1, "num_iterations": 1}
        try:
            lgb.train(params, data, num_boost_round=1)
            print("  ✓  LightGBM GPU build available.")
            return True
        except Exception:
            print("  –  LightGBM CPU-only build (GPU training unavailable).")
            return False
    except ImportError:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Trading-AI  |  Environment Check")
    print("=" * 50)

    print("\n  Python version:")
    py_ok = check_python()

    print("\n  Required packages:")
    pkg_ok = check_packages()

    check_gpu()
    check_lightgbm_gpu()

    print("\n" + "=" * 50)
    if py_ok and pkg_ok:
        print("  ✓  Environment is ready.")
    else:
        print("  ✗  Some requirements are missing. See above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
