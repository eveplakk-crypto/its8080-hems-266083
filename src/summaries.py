"""Statistical summaries of the demand, PV and price columns.

Missing values are dropped per column, never filled. Totals are only given
for the energy columns (kWh); a sum of prices has no meaning.
"""

import pandas as pd

ENERGY_COLUMNS = ["Demand", "pv", "pv_mod1", "pv_mod2", "pv_mod3"]
PRICE_COLUMNS = ["Price"]
SUMMARY_COLUMNS = ENERGY_COLUMNS + PRICE_COLUMNS


def summary_table(df: pd.DataFrame, columns=SUMMARY_COLUMNS) -> pd.DataFrame:
    """One row per variable: coverage, location, spread, shape and total."""
    rows = []
    for col in columns:
        s = df[col]
        values = s.dropna()
        q = values.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        rows.append({
            "variable": col,
            "n": int(values.size),
            "n_missing": int(s.isna().sum()),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p1": q.loc[0.01],
            "p25": q.loc[0.25],
            "median": q.loc[0.50],
            "p75": q.loc[0.75],
            "p99": q.loc[0.99],
            "max": values.max(),
            "skew": values.skew(),
            "kurtosis": values.kurt(),
            "n_zeros": int((values == 0).sum()),
            "pct_zeros": 100 * (values == 0).mean(),
            "total": values.sum() if col in ENERGY_COLUMNS else float("nan"),
        })
    return pd.DataFrame(rows).set_index("variable")


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: energy totals and means, price mean and range."""
    grouped = df.resample("MS")
    table = pd.DataFrame({
        "demand_total": grouped["Demand"].sum(),
        "demand_mean": grouped["Demand"].mean(),
        "demand_max": grouped["Demand"].max(),
        "pv_total": grouped["pv"].sum(),
        "pv_mean": grouped["pv"].mean(),
        "pv_max": grouped["pv"].max(),
        "price_mean": grouped["Price"].mean(),
        "price_min": grouped["Price"].min(),
        "price_max": grouped["Price"].max(),
        "hours": grouped["Demand"].count(),
    })
    table["pv_to_demand"] = table["pv_total"] / table["demand_total"]
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "month"
    return table


def hourly_means(df: pd.DataFrame, columns=("Demand", "pv", "Price")) -> pd.DataFrame:
    """Mean of each column by hour of day (UTC), over the whole period."""
    table = df[list(columns)].groupby(df.index.hour).mean()
    table.index.name = "hour"
    return table


def pv_copies_check(df: pd.DataFrame) -> pd.Series:
    """Share of rows where each pv_mod column equals pv, where it is present."""
    out = {}
    for col in ("pv_mod1", "pv_mod2", "pv_mod3"):
        present = df[col].notna()
        out[col] = (df.loc[present, col] - df.loc[present, "pv"]).abs().lt(1e-9).mean()
    return pd.Series(out, name="share_equal_to_pv")
