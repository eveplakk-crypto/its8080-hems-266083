"""Out-of-sample 7-day demand forecast, 1–7 July 2014, from the raw files.

Runs the whole pipeline (load → clean → model → rolling forecast → evaluate)
without the exploratory steps of main.py.

Usage:
    python forecast.py
"""

import sys

from src.config import set_seeds
from src.pipeline import run

if __name__ == "__main__":
    set_seeds()
    run()
    sys.exit(0)
