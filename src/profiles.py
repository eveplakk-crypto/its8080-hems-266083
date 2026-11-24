"""Typical demand profiles: hour-of-day statistics grouped by day type.

Hours and days are taken from the UTC index as-is. Weekday vs weekend is
therefore split on UTC calendar days, not local time.
"""

import pandas as pd

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

#: Meteorological seasons by month.
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def day_type(index: pd.DatetimeIndex) -> pd.Index:
    """'Weekday' for Mon–Fri, 'Weekend' for Sat–Sun."""
    return pd.Index(["Weekend" if d >= 5 else "Weekday" for d in index.dayofweek])


def season(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index([SEASONS[m] for m in index.month])


def hourly_profile(series: pd.Series, by: pd.Index | None = None) -> pd.DataFrame:
    """Mean, median, quartiles and count of `series` for each hour of day.

    With `by`, one profile per group, in long format with a `group` column.
    Missing values are dropped, not filled.
    """
    s = series.dropna()
    keys = [s.index.hour] if by is None else [by[series.notna()], s.index.hour]
    names = ["hour"] if by is None else ["group", "hour"]

    grouped = s.groupby(keys)
    profile = pd.DataFrame({
        "mean": grouped.mean(),
        "median": grouped.median(),
        "p25": grouped.quantile(0.25),
        "p75": grouped.quantile(0.75),
        "n": grouped.size(),
    })
    profile.index.names = names
    return profile.reset_index()


def weekly_matrix(series: pd.Series) -> pd.DataFrame:
    """Mean demand as a 24 x 7 table: hour of day by day of week."""
    s = series.dropna()
    table = s.groupby([s.index.hour, s.index.dayofweek]).mean().unstack()
    table.index.name = "hour"
    table.columns = WEEKDAY_NAMES
    return table


def build_profiles(series: pd.Series) -> pd.DataFrame:
    """All profiles in one long table, keyed by `profile` and `group`."""
    parts = [
        hourly_profile(series).assign(profile="average_day", group="All days"),
        hourly_profile(series, day_type(series.index)).assign(profile="day_type"),
        hourly_profile(series, season(series.index)).assign(profile="season"),
    ]
    columns = ["profile", "group", "hour", "mean", "median", "p25", "p75", "n"]
    return pd.concat(parts, ignore_index=True)[columns]
