"""Baseline demand forecasts, issued day-ahead.

Every forecast for day D is made at 00:00 on day D (the protocol), so it may
only use demand up to 23:00 on day D-1, the forecast origin. Each baseline
fills all 24 hours of day D from data known at that origin.
"""

import pandas as pd

from src.features import lag

DEMAND = "Demand"
MOVING_AVERAGE_HOURS = 24


def forecast_origin(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last metered hour when the forecast for each hour is issued:
    23:00 on the day before."""
    return index.floor("D") - pd.Timedelta(hours=1)


def at_origin(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Value of `series` at each hour's forecast origin."""
    values = series.reindex(forecast_origin(index))
    return pd.Series(values.to_numpy(), index=index)


def naive(y: pd.Series) -> pd.Series:
    """Naive (lag-1 at the origin): the last observed hour, 23:00 yesterday,
    repeated for all 24 hours of the day."""
    return at_origin(y, y.index).rename("Naive (last hour)")


def seasonal_naive_24(y: pd.Series) -> pd.Series:
    """Seasonal naive, lag 24: the same hour yesterday."""
    return lag(y, 24).rename("Seasonal naive 24 h")


def seasonal_naive_168(y: pd.Series) -> pd.Series:
    """Seasonal naive, lag 168: the same hour one week earlier."""
    return lag(y, 168).rename("Seasonal naive 168 h")


def moving_average(y: pd.Series, hours: int = MOVING_AVERAGE_HOURS) -> pd.Series:
    """Moving average: mean of the last 24 hours before the origin
    (all of yesterday), used as a flat forecast for every hour."""
    mean = y.rolling(f"{hours}h", min_periods=hours // 2).mean()
    return at_origin(mean, y.index).rename(f"Moving average {hours} h")


BASELINES = [naive, seasonal_naive_24, seasonal_naive_168, moving_average]


def all_baselines(y: pd.Series) -> pd.DataFrame:
    """One column per baseline, aligned to the hours of `y`."""
    return pd.concat([f(y) for f in BASELINES], axis=1)
