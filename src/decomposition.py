"""Additive decomposition of hourly demand: Yt = Tt + St + Rt.

Demand is hourly with a strong daily cycle and a weaker weekly one, so the
seasonal part is estimated with MSTL for both periods and St is their sum.
The training set covers a single year, so the annual swing cannot be
separated as a seasonal component and ends up in the trend.

The decomposition is additive because Demand contains exact zeros, which
rule out a log (multiplicative) model.
"""

import pandas as pd
from statsmodels.tsa.seasonal import MSTL

#: Seasonal periods in hours: daily and weekly.
PERIODS = (24, 168)


def regular_hourly(series: pd.Series) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Reindex onto a complete hourly grid, linearly filling any gaps.

    Returns the filled series and the timestamps that were filled, so that
    callers can blank them out again afterwards.
    """
    full = pd.date_range(series.index.min(), series.index.max(), freq="h")
    filled_at = full.difference(series.index)
    regular = series.reindex(full).interpolate(method="time")
    return regular, filled_at


def decompose(series: pd.Series, periods=PERIODS) -> pd.DataFrame:
    """Split a series into trend (Tt), seasonality (St) and residuals (Rt).

    Uses non-robust MSTL: demand spikes are real consumption, and robust
    weights would pull Tt towards the median and leave Rt with a positive
    mean. Individual seasonal components are kept alongside St.
    Rows that were interpolated to fill the grid are flagged in `filled`.
    """
    regular, filled_at = regular_hourly(series)
    result = MSTL(regular, periods=periods).fit()

    seasonal = result.seasonal
    out = pd.DataFrame({
        "Yt": regular,
        "Tt": result.trend,
        "St": seasonal.sum(axis=1),
        "Rt": result.resid,
    })
    for col in seasonal.columns:
        out[f"St_{col.removeprefix('seasonal_')}h"] = seasonal[col]
    out["filled"] = out.index.isin(filled_at)
    out.index.name = series.index.name
    return out


def variance_shares(components: pd.DataFrame) -> pd.Series:
    """Variance of each component relative to the variance of Yt."""
    var_y = components["Yt"].var()
    return components[["Tt", "St", "Rt"]].var() / var_y


def seasonal_strength(components: pd.DataFrame) -> float:
    """Fs = max(0, 1 - Var(Rt) / Var(St + Rt)), after Hyndman & Athanasopoulos."""
    detrended = components["St"] + components["Rt"]
    return max(0.0, 1 - components["Rt"].var() / detrended.var())


def trend_strength(components: pd.DataFrame) -> float:
    """Ft = max(0, 1 - Var(Rt) / Var(Tt + Rt))."""
    deseasoned = components["Tt"] + components["Rt"]
    return max(0.0, 1 - components["Rt"].var() / deseasoned.var())
