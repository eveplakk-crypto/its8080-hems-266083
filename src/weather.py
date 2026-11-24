"""Demand and weather: descriptive statistics and a simple ranking analysis.

Shortwave_radiation is left out of the ranking: it is missing for 81 % of
hours, and wherever it is present it equals direct + diffuse radiation
exactly, so it carries no extra information.
"""

import pandas as pd

DEMAND = "Demand"
SHORTWAVE = "Shortwave_radiation (W/m²)"

WEATHER = [
    "Temperature", "Pressure (hPa)", "Cloud_cover (%)", "Cloud_cover_low (%)",
    "Cloud_cover_mid (%)", "Cloud_cover_high (%)", "Wind_speed_10m (km/h)",
    "direct_radiation (W/m²)", "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)",
]

#: Short display names for tables and figures.
SHORT_NAMES = {
    "Demand": "Demand (kWh)", "Temperature": "Temperature (°C)",
    "Pressure (hPa)": "Pressure (hPa)", "Cloud_cover (%)": "Cloud cover (%)",
    "Cloud_cover_low (%)": "Cloud cover low (%)",
    "Cloud_cover_mid (%)": "Cloud cover mid (%)",
    "Cloud_cover_high (%)": "Cloud cover high (%)",
    "Wind_speed_10m (km/h)": "Wind speed (km/h)",
    SHORTWAVE: "Shortwave radiation (W/m²)",
    "direct_radiation (W/m²)": "Direct radiation (W/m²)",
    "diffuse_radiation (W/m²)": "Diffuse radiation (W/m²)",
    "direct_normal_irradiance (W/m²)": "Direct normal irradiance (W/m²)",
}


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per variable: coverage, location, spread and shape."""
    rows = []
    for col in [DEMAND, *WEATHER, SHORTWAVE]:
        s = df[col]
        v = s.dropna()
        q = v.quantile([0.25, 0.50, 0.75])
        rows.append({
            "variable": SHORT_NAMES[col], "n": int(v.size),
            "pct_missing": 100 * s.isna().mean(), "mean": v.mean(),
            "std": v.std(), "min": v.min(), "p25": q.loc[0.25],
            "median": q.loc[0.50], "p75": q.loc[0.75], "max": v.max(),
            "skew": v.skew(),
        })
    return pd.DataFrame(rows).set_index("variable")


def shortwave_check(df: pd.DataFrame) -> float:
    """Largest |shortwave − (direct + diffuse)| where shortwave is present."""
    both = df[[SHORTWAVE, "direct_radiation (W/m²)",
               "diffuse_radiation (W/m²)"]].dropna()
    return float((both.iloc[:, 0] - both.iloc[:, 1] - both.iloc[:, 2]).abs().max())


def daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean of demand and every weather variable."""
    return df[[DEMAND, *WEATHER]].resample("D").mean()


def rank_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each weather variable with demand, hourly and
    daily, ranked by the absolute daily value.

    Spearman is used because demand is skewed and the relations need not be
    linear. The daily level removes the shared day/night cycle, which at the
    hourly level makes anything that peaks in daytime look related to demand.
    """
    daily = daily_means(df)
    table = pd.DataFrame({
        "hourly_rho": df[WEATHER].corrwith(df[DEMAND], method="spearman"),
        "daily_rho": daily[WEATHER].corrwith(daily[DEMAND], method="spearman"),
    })
    table["hourly_rank"] = table["hourly_rho"].abs().rank(ascending=False).astype(int)
    table["daily_rank"] = table["daily_rho"].abs().rank(ascending=False).astype(int)
    table.index = [SHORT_NAMES[c] for c in table.index]
    table.index.name = "variable"
    return table.sort_values("daily_rank")


def demand_by_temperature(df: pd.DataFrame,
                          edges=(-20, -5, 0, 5, 10, 15, 20, 30)) -> pd.DataFrame:
    """Mean daily demand in daily-mean temperature bands."""
    daily = daily_means(df)
    bands = pd.cut(daily["Temperature"], list(edges))
    grouped = daily.groupby(bands, observed=True)
    out = pd.DataFrame({
        "mean_temperature": grouped["Temperature"].mean(),
        "mean_demand_kWh": grouped[DEMAND].mean(),
        "n_days": grouped[DEMAND].size(),
    })
    out.index = out.index.astype(str)
    out.index.name = "temperature band (°C)"
    return out


def hourly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Mean demand, temperature and total radiation by hour of day."""
    radiation = df["direct_radiation (W/m²)"] + df["diffuse_radiation (W/m²)"]
    frame = pd.DataFrame({"demand": df[DEMAND], "temperature": df["Temperature"],
                          "radiation": radiation})
    return frame.groupby(df.index.hour).mean().rename_axis("hour")
