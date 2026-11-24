"""Normality checks and transformations for demand and weather variables.

With ~8,760 hourly values, formal normality tests reject even tiny departures
from normal, so they are reported but the decision rests on skewness,
excess kurtosis, the share of zeros and the Q-Q plots.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.weather import DEMAND, SHORT_NAMES, WEATHER

#: |skew| and |excess kurtosis| below these count as "close to normal".
SKEW_LIMIT = 0.5
KURT_LIMIT = 1.0

RADIATION = ["direct_radiation (W/m²)", "diffuse_radiation (W/m²)",
             "direct_normal_irradiance (W/m²)"]
WIND = "Wind_speed_10m (km/h)"
SHAPIRO_SAMPLE = 5000
RANDOM_STATE = 0


def normality_table(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Shape statistics and normality tests, one row per variable."""
    columns = columns or [DEMAND, *WEATHER]
    rows = []
    for col in columns:
        x = df[col].dropna()
        sample = x.sample(min(SHAPIRO_SAMPLE, len(x)), random_state=RANDOM_STATE)
        skew, kurt = x.skew(), x.kurt()
        rows.append({
            "variable": SHORT_NAMES.get(col, col), "n": int(x.size),
            "skew": skew, "excess_kurtosis": kurt,
            "pct_zeros": 100 * (x == 0).mean(),
            "dagostino_p": stats.normaltest(x).pvalue,
            "shapiro_p": stats.shapiro(sample).pvalue,
            "close_to_normal": abs(skew) < SKEW_LIMIT and abs(kurt) < KURT_LIMIT,
        })
    return pd.DataFrame(rows).set_index("variable")


def demand_transforms(demand: pd.Series) -> dict:
    """Candidate transformations of hourly demand.

    Five hours are exactly 0 (likely outages), so log and Box-Cox are fitted
    on positive values only; they are left as NaN at those hours.
    """
    positive = demand.where(demand > 0)
    boxcox, lam = stats.boxcox(positive.dropna())
    return {
        "Original": demand,
        "Square root": np.sqrt(demand),
        "Log": np.log(positive),
        f"Box-Cox (λ = {lam:.2f})": pd.Series(boxcox, index=positive.dropna().index)
                                     .reindex(demand.index),
    }


def transform_table(series: dict) -> pd.DataFrame:
    """Skew, kurtosis and D'Agostino p-value for each transformed series."""
    rows = []
    for name, s in series.items():
        x = s.dropna()
        rows.append({"transformation": name, "skew": x.skew(),
                     "excess_kurtosis": x.kurt(),
                     "dagostino_p": stats.normaltest(x).pvalue})
    return pd.DataFrame(rows).set_index("transformation")


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Transformed columns to use in modelling, next to their originals.

    - Demand: natural log (zero hours left as NaN).
    - Wind speed: square root.
    - Direct and diffuse radiation: square root, plus a daylight flag. Over
      half the hours are night zeros, which no transformation can make
      normal, so the flag carries day/night and the square root evens out
      the daytime values. Direct normal irradiance is kept as it is: its
      daytime values are already nearly symmetric and the square root
      makes them flatter, not closer to normal.
    Temperature, pressure and cloud cover are kept as they are.
    """
    out = pd.DataFrame(index=df.index)
    out["Demand"] = df[DEMAND]
    out["log_Demand"] = np.log(df[DEMAND].where(df[DEMAND] > 0))
    out["Wind_speed_10m (km/h)"] = df[WIND]
    out["sqrt_Wind_speed"] = np.sqrt(df[WIND])
    for col in RADIATION[:2]:
        short = col.split(" (")[0]
        out[col] = df[col]
        out[f"sqrt_{short}"] = np.sqrt(df[col])
    out["is_daylight"] = (df["direct_radiation (W/m²)"]
                          + df["diffuse_radiation (W/m²)"] > 0).astype(int)
    return out
