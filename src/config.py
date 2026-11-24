"""Project paths and shared setup for analysis scripts and notebooks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
TABLES = PROJECT_ROOT / "outputs" / "tables"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

#: One seed for every random step. The modules also pass `random_state=0`
#: explicitly to each scikit-learn estimator and to the Shapiro-Wilk sample.
SEED = 0


def set_seeds(seed: int = SEED) -> None:
    """Fix the Python, NumPy and hash seeds so every run gives the same numbers."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_notebook():
    """Sensible display defaults for exploratory work in a notebook."""
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
