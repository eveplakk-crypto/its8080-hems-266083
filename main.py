"""Project 266083 — training data pipeline.

Loads the hourly training set, prints a summary, and builds a data
dictionary saved to outputs/tables/.

Usage:
    python main.py
"""

import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (DATA_PROCESSED, DATA_RAW, FIGURES, TABLES, set_seeds,
                        setup_notebook)
from src.imputation import (TARGET, TRUTH, distribution_summary, evaluate,
                            evaluate_live, impute_all, impute_live,
                            linear_interpolation, sun_down)
from src.weather import SHORT_NAMES
from src.weather import WEATHER as WEATHER_COLUMNS
from src.weather import (demand_by_temperature, daily_means, describe,
                         rank_drivers, shortwave_check)
from src.distributions import (RADIATION, WIND, apply_transformations,
                               demand_transforms, normality_table,
                               transform_table)
from src.features import (AVAILABILITY, HORIZON_HOURS, LAG_HOURS,
                          autocorrelation, candidate_features,
                          feature_relevance, lag, modelling_set, rank_features)
from src.evaluation import (SPLITS, predict_seasonal_naive_nrmse, score,
                            split_frame, split_summary, training_mean,
                            walk_forward_folds)
from src.baselines import all_baselines
from src.stationarity import (CANDIDATES, acf_pacf_table, difference,
                              stationarity_table, training_series)
from src.arima_models import (READ_FROM_PLOTS, aic_search, day_ahead, fit,
                               horizon_profile, hour_dummies, model_series)
from src.pipeline import run as run_forecast_pipeline
from src.decomposition import (PERIODS, decompose, seasonal_strength,
                               trend_strength, variance_shares)
from src.profiles import SEASON_ORDER, build_profiles, weekly_matrix
from src.summaries import (hourly_means, monthly_summary, pv_copies_check,
                           summary_table)

TRAINING_FILE = "266083_training.csv"

OUTPUT_NAME = "266083_training_data_dictionary.csv"

#: Units are taken only from this mapping, never inferred from values.
UNITS = {
    "Demand": "kWh", "pv": "kWh",
    "pv_mod1": "kWh", "pv_mod2": "kWh", "pv_mod3": "kWh",
    "Price": "€/kWh", "Temperature": "°C", "Pressure (hPa)": "hPa",
    "Cloud_cover (%)": "%", "Cloud_cover_low (%)": "%",
    "Cloud_cover_mid (%)": "%", "Cloud_cover_high (%)": "%",
    "Wind_speed_10m (km/h)": "km/h",
    "Shortwave_radiation (W/m²)": "W/m²", "direct_radiation (W/m²)": "W/m²",
    "diffuse_radiation (W/m²)": "W/m²", "direct_normal_irradiance (W/m²)": "W/m²",
}

UNKNOWN = "unknown"
NO_UNIT = "not stated in file"

FIGURE_NAME = "pv_demand_price_2013-12-03_05.png"
WINDOW_START = "2013-12-03 00:00"
WINDOW_END = "2013-12-05 23:59"
EXPECTED_ROWS = 72

#: Categorical slots 1, 2 and 7 from the reference palette.
COLOR_PV = "#2a78d6"
COLOR_DEMAND = "#eb6834"
COLOR_PRICE = "#4a3aa7"

DECOMPOSITION_NAME = "266083_demand_decomposition.csv"
DECOMPOSITION_FIGURE = "demand_decomposition.png"
DECOMPOSITION_ZOOM_FIGURE = "demand_decomposition_2014-01-13_26.png"
ZOOM_START = "2014-01-13 00:00"
ZOOM_END = "2014-01-26 23:59"

PROFILES_NAME = "266083_demand_profiles.csv"
WEEKLY_MATRIX_NAME = "266083_demand_hour_by_weekday.csv"
PROFILES_FIGURE = "demand_profiles.png"

#: Categorical slots 1–4 from the reference palette, in fixed order.
COLOR_WEEKDAY = "#2a78d6"
COLOR_WEEKEND = "#eb6834"
SEASON_COLORS = dict(zip(SEASON_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

SUMMARY_NAME = "266083_summary_demand_pv_price.csv"
MONTHLY_NAME = "266083_monthly_demand_pv_price.csv"
HOURLY_MEANS_NAME = "266083_hourly_means_demand_pv_price.csv"
TIMESERIES_FIGURE = "timeseries_demand_pv_price.png"

HISTOGRAM_FIGURE = "distributions_demand_pv_price.png"
BALANCE_FIGURE = "demand_vs_pv_with_price.png"
SMOOTHING_DAYS = 7
MONTHLY_BALANCE_FIGURE = "monthly_pv_vs_demand.png"
MONTHLY_BALANCE_NAME = "266083_monthly_pv_vs_demand.csv"

IMPUTATION_NAME = "266083_pv_mod1_imputation.csv"
IMPUTED_NAME = "266083_pv_mod1_imputed.csv"
IMPUTATION_SUMMARY_NAME = "266083_pv_mod1_imputed_summary.csv"
IMPUTATION_FIGURE = "pv_mod1_imputation_2013-07-14_19.png"
IMPUTATION_START = "2013-07-14 00:00"
IMPUTATION_END = "2013-07-19 23:59"
#: Categorical slots 1–4 from the reference palette, one per imputation.
IMPUTATION_COLORS = {
    "Mean imputation": "#eda100", "Linear interpolation": "#2a78d6",
    "KNN imputation (k=5)": "#1baf7a", "MICE (Extra Trees)": "#eb6834",
}

INTERPOLATION_FIGURE = "pv_mod1_interpolation_before_after.png"
#: Representative gaps in pv_mod1: (label, window start, window end).
#: Each window shows the gap with several observed hours on either side.
INTERPOLATION_CASES = [
    ("1 · Night, 05 Jul 2013 02:00", "2013-07-04 21:00", "2013-07-05 07:00"),
    ("2 · Sunrise edge, 22 Jul 2013 06:00", "2013-07-22 02:00", "2013-07-22 11:00"),
    ("3 · Clear day, 10 Apr 2014 08:00 & 10:00", "2014-04-10 04:00", "2014-04-10 14:00"),
    ("4 · Overcast winter day, 18 Nov 2013 11:00", "2013-11-18 07:00", "2013-11-18 16:00"),
    ("5 · Longest gap (3 h), 01 Mar 2014 14:00–16:00", "2014-03-01 09:00", "2014-03-01 19:00"),
    ("6 · Variable clouds, 10 Feb 2014 14:00", "2014-02-10 09:00", "2014-02-10 18:00"),
]
COLOR_FILLED = "#eb6834"

LIVE_NAME = "266083_pv_mod1_live_imputation.csv"
LIVE_IMPUTED_NAME = "266083_pv_mod1_live_imputed.csv"

WEATHER_SUMMARY_NAME = "266083_demand_weather_summary.csv"
WEATHER_RANKING_NAME = "266083_demand_weather_ranking.csv"
DEMAND_TEMPERATURE_NAME = "266083_demand_by_temperature.csv"
DEMAND_HISTOGRAM_FIGURE = "demand_histogram.png"
DEMAND_HEATMAP_FIGURE = "demand_heatmap_month_hour.png"
DEMAND_TEMPERATURE_BARS_FIGURE = "demand_by_temperature_band.png"

NORMALITY_NAME = "266083_normality_tests.csv"
DEMAND_TRANSFORM_NAME = "266083_demand_transformations.csv"
WEATHER_TRANSFORM_NAME = "266083_weather_transformations.csv"
TRANSFORMED_NAME = "266083_transformed_features.csv"
NORMALITY_FIGURE = "distributions_vs_normal.png"
DEMAND_TRANSFORM_FIGURE = "demand_transformation_qq.png"
WEATHER_TRANSFORM_FIGURE = "weather_transformations.png"
#: Grey reference curve for "what a normal distribution would look like".
COLOR_NORMAL = "#0b0b0b"

FEATURES_NAME = "266083_features.csv"
FEATURE_RELEVANCE_NAME = "266083_feature_relevance.csv"
FEATURE_AVAILABILITY_NAME = "266083_feature_availability.csv"
ACF_NAME = "266083_demand_autocorrelation.csv"
ACF_FIGURE = "demand_autocorrelation_lags.png"
FEATURE_RANKING_NAME = "266083_feature_ranking.csv"
FEATURE_RANKING_FIGURE = "feature_ranking.png"
ACF_MAX_LAG = 336

SPLIT_SUMMARY_NAME = "266083_split_summary.csv"
SPLIT_LABELS_NAME = "266083_split_labels.csv"
SPLIT_FIGURE = "train_validation_test_split.png"
WALK_FORWARD_NAME = "266083_walk_forward_folds.csv"
WALK_FORWARD_FIGURE = "walk_forward_daily_folds.png"
BASELINE_SCORES_NAME = "266083_baseline_scores.csv"
BASELINES_FIGURE = "baselines_walk_forward_week.png"
NRMSE_PREDICTION_NAME = "266083_seasonal_naive_nrmse_prediction.csv"
NRMSE_COMPARISON_NAME = "266083_seasonal_naive_nrmse_prediction_vs_actual.csv"
SEASONAL_NAIVE_NAME = "Seasonal naive 24 h"
#: Days of training history shown before the first fold in the figure.
WALK_FORWARD_CONTEXT_DAYS = 7
#: Categorical slots 1-3, one per set, in fixed order.
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}

STATIONARITY_NAME = "266083_stationarity_tests.csv"
STATIONARITY_FIGURE = "stationarity_differencing.png"
#: The differencing chosen from the tests: d = 1, D = 0.
CHOSEN_DIFFERENCING = "first difference (lag 1)"
ACF_PACF_NAME = "266083_acf_pacf_differenced.csv"
ACF_PACF_FIGURE = "acf_pacf_differenced.png"
ACF_PACF_SHORT = 48
ACF_PACF_LONG = 200
STATIONARITY_PANELS = ["level (no differencing)", "first difference (lag 1)",
                       "seasonal difference (lag 24)"]

ARIMA_SEARCH_NAME = "266083_arima_aic_search.csv"
ARIMA_PARAMS_NAME = "266083_arima_parameters.csv"
ARIMA_SCORES_NAME = "266083_arima_scores.csv"
ARIMA_FORECASTS_NAME = "266083_arima_forecasts.csv"
ARIMA_FIGURE = "arima_walk_forward_week.png"
COMPARISON_NAME = "266083_in_sample_vs_walk_forward.csv"
SARIMAX_BASE = "SARIMA(0,1,2)(1,0,0)24"
SARIMAX_NAME = "SARIMAX(0,1,2)(1,0,0)24 + hour"
SARIMAX_SCORES_NAME = "266083_sarimax_hour_scores.csv"
SARIMAX_PARAMS_NAME = "266083_sarimax_hour_parameters.csv"
SARIMAX_FIGURE = "sarimax_hour_walk_forward_week.png"
HORIZON_NAME = "266083_error_by_horizon.csv"
HORIZON_FIGURE = "error_by_horizon.png"
WINDOW_NAME = "266083_multi_output_training_window.csv"
WINDOW_FIGURE = "multi_output_training_window.png"
#: Categorical slots 3 and 4 for the two models; slot 2 is demand.
MODEL_COLORS = ["#1baf7a", "#eda100"]

IMPOSSIBLE_NAME = "266083_pv_mod1_impossible_values.csv"
IMPOSSIBLE_FIGURE = "pv_mod1_impossible_values.png"
#: (method, gap, window start, window end) for each example shown.
IMPOSSIBLE_CASES = [
    ("Mean imputation", "2013-07-05 02:00", "2013-07-04 21:00", "2013-07-05 07:00"),
    ("Linear interpolation", "2013-11-13 07:00", "2013-11-13 03:00", "2013-11-13 11:00"),
]

#: Histogram bin widths, in each variable's own unit.
BIN_WIDTHS = {"Demand": 0.05, "pv": 0.05, "Price": 0.005}
#: Demand and PV share this x-range (kWh) so their histograms compare directly.
ENERGY_X_MAX = 5.0
INK_SECONDARY = "#52514e"

#: Panel order and labels shared by the time-series figures.
SERIES_PANELS = [
    ("Demand", "Demand (kWh)", COLOR_DEMAND),
    ("pv", "PV (kWh)", COLOR_PV),
    ("Price", "Price (€/kWh)", COLOR_PRICE),
]

def load_training(filename: str = TRAINING_FILE) -> pd.DataFrame:
    """Load the hourly training set, indexed by UTC-aware timestamp.

    Columns are kept exactly as they appear in the file, and missing
    values are left in place.
    """
    path = DATA_RAW / filename
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")

def summarize(df: pd.DataFrame) -> None:
    """Print shape, index range, dtypes, missing counts and head."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Index: {df.index.min()}  ->  {df.index.max()}")
    print(f"Index dtype: {df.index.dtype}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    print(f"\nMissing values (total {int(missing.sum()):,}):")
    print(missing.to_string())

    print("\nHead:")
    print(df.head())

def sampling_resolution(index: pd.DatetimeIndex):
    """Modal gap between consecutive timestamps, computed from the index."""
    if len(index) < 2:
        return None
    gaps = pd.Series(index).diff().dropna()
    modes = gaps.mode()
    return modes.iloc[0] if len(modes) else None

def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, unit, coverage, range and distribution."""
    resolution = sampling_resolution(df.index)
    resolution_str = str(resolution) if resolution is not None else UNKNOWN
    n_rows = len(df)

    stat_fields = ["min", "max", "mean", "median", "std",
                   "p1", "p25", "p50", "p75", "p99", "n_zeros"]
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        row = {
            "variable": col,
            "dtype": str(s.dtype),
            "unit": UNITS.get(col, NO_UNIT),
            "sampling_resolution": resolution_str,
            "n_non_missing": int(s.notna().sum()),
            "n_missing": n_missing,
            "pct_missing": 100 * n_missing / n_rows if n_rows else UNKNOWN,
            "n_distinct": int(s.nunique(dropna=True)),
        }

        values = s.dropna() if pd.api.types.is_numeric_dtype(s) else None
        if values is not None and len(values):
            q = values.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
            row.update({
                "min": values.min(), "max": values.max(),
                "mean": values.mean(), "median": values.median(),
                "std": values.std(),
                "p1": q.loc[0.01], "p25": q.loc[0.25], "p50": q.loc[0.50],
                "p75": q.loc[0.75], "p99": q.loc[0.99],
                "n_zeros": int((values == 0).sum()),
            })
        else:
            row.update(dict.fromkeys(stat_fields, UNKNOWN))

        rows.append(row)

    columns = ["variable", "dtype", "unit", "sampling_resolution",
               "n_non_missing", "n_missing", "pct_missing",
               *stat_fields[:-1], "n_distinct", "n_zeros"]
    return pd.DataFrame(rows)[columns]

def dataset_summary(df: pd.DataFrame) -> dict:
    """Index-level facts: coverage, ordering, duplicates and gaps."""
    index = df.index
    resolution = sampling_resolution(index)

    if resolution is not None:
        expected = pd.date_range(index.min(), index.max(), freq=resolution)
        gaps = expected.difference(index)
    else:
        expected, gaps = None, None

    return {
        "first_timestamp": index.min(),
        "last_timestamp": index.max(),
        "duration": index.max() - index.min(),
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "sampling_resolution": resolution if resolution is not None else UNKNOWN,
        "strictly_increasing": bool(index.is_monotonic_increasing and index.is_unique),
        "n_duplicate_timestamps": int(index.duplicated().sum()),
        "n_expected_timestamps": len(expected) if expected is not None else UNKNOWN,
        "n_missing_timestamps": len(gaps) if gaps is not None else UNKNOWN,
        "missing_timestamps": list(gaps) if gaps is not None else UNKNOWN,
    }

def report(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Print the dataset summary and dictionary; optionally save to TABLES."""
    summary = dataset_summary(df)

    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    for key, value in summary.items():
        if key == "missing_timestamps" and isinstance(value, list):
            value = "none" if not value else (
                f"{value[:10]} ... (+{len(value) - 10} more)"
                if len(value) > 10 else value)
        print(f"{key:25s} {value}")

    dictionary = build_dictionary(df)

    print()
    print("=" * 70)
    print("VARIABLES")
    print("=" * 70)
    display = dictionary.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].round(3)
    print(display.to_string(index=False))

    if save:
        TABLES.mkdir(parents=True, exist_ok=True)
        out = TABLES / OUTPUT_NAME
        dictionary.to_csv(out, index=False)  # full precision
        print(f"\nSaved to {out}")

    return dictionary

def plot_pv_demand_price(df: pd.DataFrame,
                         start: str = WINDOW_START,
                         end: str = WINDOW_END):
    """Plot pv, Demand and Price over a window, in two stacked panels.

    The window is sliced from the existing UTC index. Nothing is resampled,
    filled or interpolated; missing values are left as visible gaps.
    """
    setup_notebook()
    window = df.loc[start:end]

    print(f"Window rows: {len(window)}")
    print(f"Window first: {window.index.min()}")
    print(f"Window last:  {window.index.max()}")
    if len(window) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, got {len(window)}")

    for col in ("pv", "Demand", "Price"):
        n = int(window[col].isna().sum())
        if n:
            print(f"WARNING: {col} has {n} missing value(s) in window; shown as gaps")

    fig, (ax_energy, ax_price) = plt.subplots(2, 1, sharex=True)

    ax_energy.plot(window.index, window["pv"], color=COLOR_PV, label="pv")
    ax_energy.plot(window.index, window["Demand"], color=COLOR_DEMAND, label="Demand")
    ax_energy.set_ylabel("kWh")
    ax_energy.legend()

    ax_price.plot(window.index, window["Price"], color=COLOR_PRICE)
    ax_price.set_ylabel("€/kWh")
    ax_price.set_xlabel("time (UTC)")

    ax_price.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
    ax_price.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=range(0, 24, 6), tz="UTC"))

    fig.suptitle("PV, demand and price — 3–5 December 2013 (UTC)")

    out = FIGURES / FIGURE_NAME
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def decompose_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose Demand into Tt, St and Rt; save components and figures."""
    components = decompose(df["Demand"])

    n_filled = int(components["filled"].sum())
    if n_filled:
        filled = list(components.index[components["filled"]])
        print(f"Filled {n_filled} missing timestamp(s) by interpolation: {filled}")

    check = (components["Yt"] - components[["Tt", "St", "Rt"]].sum(axis=1)).abs().max()
    print(f"Periods (h): {PERIODS}")
    print(f"Max |Yt - (Tt + St + Rt)|: {check:.2e}")
    print("Variance share of Yt:")
    print(variance_shares(components).round(3).to_string())
    print(f"Seasonal strength Fs: {seasonal_strength(components):.3f}")
    print(f"Trend strength Ft:    {trend_strength(components):.3f}")
    print("\nComponent summary (kWh):")
    print(components[["Yt", "Tt", "St", "Rt"]].describe().round(3).to_string())

    out = DATA_PROCESSED / DECOMPOSITION_NAME
    components.to_csv(out)
    print(f"\nSaved components to {out}")

    plot_decomposition(components, DECOMPOSITION_FIGURE,
                       "Demand decomposition, Jul 2013 – Jun 2014 (UTC)")
    plot_decomposition(components.loc[ZOOM_START:ZOOM_END], DECOMPOSITION_ZOOM_FIGURE,
                       "Demand decomposition, 13–26 January 2014 (UTC)")
    return components


def plot_decomposition(components: pd.DataFrame, filename: str, title: str):
    """Four stacked panels, Yt, Tt, St and Rt, each on its own y-scale."""
    setup_notebook()
    panels = [("Yt", "Demand Yt"), ("Tt", "Trend Tt"),
              ("St", "Seasonality St"), ("Rt", "Residuals Rt")]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(10, 8))
    for ax, (col, label) in zip(axes, panels):
        ax.plot(components.index, components[col], color=COLOR_DEMAND, linewidth=0.8)
        ax.set_ylabel(f"{label}\n(kWh)")
        if col in ("St", "Rt"):
            ax.axhline(0, color="#888888", linewidth=0.8)
    axes[-1].set_xlabel("time (UTC)")
    fig.suptitle(title)
    out = FIGURES / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def demand_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Build hour-of-day demand profiles; save tables and a figure."""
    profiles = build_profiles(df["Demand"])
    matrix = weekly_matrix(df["Demand"])

    wide = profiles.pivot_table(index="hour", columns="group", values="mean")
    wide = wide[["All days", "Weekday", "Weekend", *SEASON_ORDER]]
    print("Mean demand by hour of day (kWh, hours in UTC):")
    print(wide.round(3).to_string())

    daily = wide.sum()
    print("\nImplied mean daily total (kWh/day):")
    print(daily.round(2).to_string())
    for group in wide.columns:
        col = wide[group]
        print(f"{group:9s} peak {col.max():.3f} at {col.idxmax():02d}:00, "
              f"min {col.min():.3f} at {col.idxmin():02d}:00")

    print("\nMean demand, hour x weekday (kWh):")
    print(matrix.round(3).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(TABLES / PROFILES_NAME, index=False)
    matrix.to_csv(TABLES / WEEKLY_MATRIX_NAME)
    print(f"\nSaved {TABLES / PROFILES_NAME}")
    print(f"Saved {TABLES / WEEKLY_MATRIX_NAME}")

    plot_profiles(profiles)
    return profiles


def plot_profiles(profiles: pd.DataFrame):
    """Three panels: average day with IQR, weekday vs weekend, by season."""
    setup_notebook()
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.8))

    def line(ax, group, color):
        p = profiles[profiles["group"] == group]
        ax.plot(p["hour"], p["mean"], color=color, linewidth=2, label=group)
        return p

    def label_ends(ax, groups, min_gap=0.035):
        """Direct labels at the last hour, pushed apart so they don't collide."""
        ends = sorted((profiles.loc[profiles["group"] == g, "mean"].iloc[-1], g)
                      for g in groups)
        placed = []
        for y, g in ends:
            y = max(y, placed[-1] + min_gap) if placed else y
            placed.append(y)
            ax.text(23.4, y, g, va="center", color="#52514e", fontsize=9)

    ax = axes[0]
    p = line(ax, "All days", COLOR_DEMAND)
    ax.fill_between(p["hour"], p["p25"], p["p75"], color=COLOR_DEMAND,
                    alpha=0.18, linewidth=0, label="25–75th percentile")
    ax.plot(p["hour"], p["median"], color=COLOR_DEMAND, linewidth=1,
            linestyle="--", label="median")
    ax.set_title("Average day")
    ax.set_ylabel("Demand (kWh)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    line(ax, "Weekday", COLOR_WEEKDAY)
    line(ax, "Weekend", COLOR_WEEKEND)
    label_ends(ax, ["Weekday", "Weekend"])
    ax.set_title("Weekday vs weekend (mean)")
    ax.legend(loc="lower center", fontsize=9)

    ax = axes[2]
    for group in SEASON_ORDER:
        line(ax, group, SEASON_COLORS[group])
    label_ends(ax, SEASON_ORDER)
    ax.set_title("By season (mean)")
    ax.legend(loc="lower center", fontsize=9)

    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlim(0, 26)
        ax.set_xlabel("hour of day (UTC)")
        ax.set_ylim(bottom=0)

    fig.suptitle("Typical demand profiles, Jul 2013 – Jun 2014")
    out = FIGURES / PROFILES_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def summarize_demand_pv_price(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical summaries of Demand, PV and Price; save tables and figures."""
    summary = summary_table(df)
    monthly = monthly_summary(df)
    hourly = hourly_means(df)

    print("Summary statistics (Demand and pv in kWh, Price in €/kWh):")
    print(summary.round(3).T.to_string())

    print("\npv_mod1-3 against pv, where present:")
    print(pv_copies_check(df).round(3).to_string())

    print("\nMonthly summary:")
    print(monthly.round(3).to_string())

    print("\nMean by hour of day (UTC):")
    print(hourly.round(3).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    for table, name in ((summary, SUMMARY_NAME), (monthly, MONTHLY_NAME),
                        (hourly, HOURLY_MEANS_NAME)):
        table.to_csv(TABLES / name)  # full precision
        print(f"Saved {TABLES / name}")

    plot_timeseries(df)
    return summary


def plot_timeseries(df: pd.DataFrame):
    """Full year of Demand, PV and Price: hourly values with the daily mean."""
    setup_notebook()
    fig, axes = plt.subplots(len(SERIES_PANELS), 1, sharex=True, figsize=(12, 8))
    for ax, (col, label, color) in zip(axes, SERIES_PANELS):
        daily = df[col].resample("D").mean()
        ax.plot(df.index, df[col], color=color, linewidth=0.4, alpha=0.35,
                label="hourly")
        ax.plot(daily.index, daily, color=color, linewidth=1.6, label="daily mean")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(tz="UTC"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y", tz="UTC"))
    axes[-1].set_xlabel("time (UTC)")
    fig.suptitle("Demand, PV and price, hourly, Jul 2013 – Jun 2014")
    out = FIGURES / TIMESERIES_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def distribution_values(df: pd.DataFrame, col: str) -> tuple[pd.Series, str]:
    """Values to plot for one variable, and a note on what was left out.

    PV is zero for about half of all hours (night), and that spike would hide
    the shape of the producing hours, so PV is shown for pv > 0 only.
    """
    values = df[col].dropna()
    if col == "pv":
        zeros = (values == 0).mean()
        return values[values > 0], f"hours with pv > 0 only ({zeros:.0%} of hours are 0)"
    return values, "all hours"


def plot_distributions(df: pd.DataFrame):
    """Histograms of Demand, PV and Price, with mean and median marked."""
    setup_notebook()
    fig, axes = plt.subplots(1, len(SERIES_PANELS), figsize=(14, 4.6))
    # Demand and PV share the y-axis (hours) as well as the x-axis, so bar
    # heights compare directly.
    axes[1].sharey(axes[0])
    for ax, (col, label, color) in zip(axes, SERIES_PANELS):
        values, note = distribution_values(df, col)
        width = BIN_WIDTHS[col]
        top = ENERGY_X_MAX if col in ("Demand", "pv") else values.max()
        bins = np.arange(0, top + width, width)
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(values.median(), color=INK_SECONDARY, linewidth=1.5,
                   label=f"median {values.median():.3f}")
        ax.axvline(values.mean(), color=INK_SECONDARY, linewidth=1.5,
                   linestyle="--", label=f"mean {values.mean():.3f}")
        ax.set_xlabel(f"{label}, bin width {width}")
        ax.set_title(f"{label.split(' (')[0]}: {note}", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        if col in ("Demand", "pv"):
            ax.set_xlim(0, ENERGY_X_MAX)
    axes[0].set_ylabel("hours")
    axes[1].tick_params(labelleft=True)
    fig.suptitle("Distribution of hourly values, Jul 2013 – Jun 2014", y=1.02)
    out = FIGURES / HISTOGRAM_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def describe_distributions(df: pd.DataFrame) -> None:
    """Print the shape facts the distribution plots are meant to show."""
    for col, label, _ in SERIES_PANELS:
        values, note = distribution_values(df, col)
        q = values.quantile([0.05, 0.5, 0.95])
        iqr = values.quantile(0.75) - values.quantile(0.25)
        upper = values.quantile(0.75) + 1.5 * iqr
        print(f"{label} [{note}]: n={len(values):,}  p5={q.loc[0.05]:.3f}  "
              f"median={q.loc[0.5]:.3f}  p95={q.loc[0.95]:.3f}  "
              f"skew={values.skew():.2f}  "
              f"above 1.5xIQR fence ({upper:.3f}): {(values > upper).mean():.1%}")


def daily_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Daily demand and PV totals (kWh/day), daily mean price, and 7-day means.

    The 7-day means are centred: this table is for display, not a model
    input, so looking ahead three days is fine here.
    """
    daily = pd.DataFrame({
        "demand": df["Demand"].resample("D").sum(),
        "pv": df["pv"].resample("D").sum(),
        "price": df["Price"].resample("D").mean(),
    })
    smooth = daily.rolling(SMOOTHING_DAYS, center=True, min_periods=1).mean()
    return daily.join(smooth.add_suffix("_7d"))


def plot_demand_vs_pv(df: pd.DataFrame):
    """Demand and PV on one axis, with price below on the same time axis."""
    setup_notebook()
    daily = daily_balance(df)

    surplus = daily["pv_7d"] > daily["demand_7d"]
    print(f"Days with PV > demand (daily totals): {(daily['pv'] > daily['demand']).sum()} "
          f"of {len(daily)}")
    print("Share of days with PV > demand, by month:")
    by_month = (daily["pv"] > daily["demand"]).groupby(daily.index.strftime("%Y-%m")).mean()
    print(by_month.map("{:.0%}".format).to_string())

    fig, (ax, ax_price) = plt.subplots(2, 1, sharex=True, figsize=(12, 7),
                                       gridspec_kw={"height_ratios": [2.2, 1]})

    ax.plot(daily.index, daily["demand"], color=COLOR_DEMAND, linewidth=0.6, alpha=0.35)
    ax.plot(daily.index, daily["pv"], color=COLOR_PV, linewidth=0.6, alpha=0.35)
    ax.fill_between(daily.index, daily["demand_7d"], daily["pv_7d"], where=surplus,
                    interpolate=True, color=COLOR_PV, alpha=0.18, linewidth=0,
                    label="PV higher than demand")
    ax.fill_between(daily.index, daily["demand_7d"], daily["pv_7d"], where=~surplus,
                    interpolate=True, color=COLOR_DEMAND, alpha=0.18, linewidth=0,
                    label="Demand higher than PV")
    ax.plot(daily.index, daily["demand_7d"], color=COLOR_DEMAND, linewidth=2.2,
            label=f"Demand, {SMOOTHING_DAYS}-day mean")
    ax.plot(daily.index, daily["pv_7d"], color=COLOR_PV, linewidth=2.2,
            label=f"PV, {SMOOTHING_DAYS}-day mean")
    ax.set_ylabel("Energy per day (kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.set_title("Demand vs PV (thin lines: daily totals)", loc="left", fontsize=10)

    ax_price.plot(daily.index, daily["price"], color=COLOR_PRICE, linewidth=0.6,
                  alpha=0.35, label="daily mean")
    ax_price.plot(daily.index, daily["price_7d"], color=COLOR_PRICE, linewidth=2.2,
                  label=f"{SMOOTHING_DAYS}-day mean")
    ax_price.set_ylabel("Price (€/kWh)")
    ax_price.set_ylim(bottom=0)
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.set_title("Electricity price", loc="left", fontsize=10)

    ax_price.xaxis.set_major_locator(mdates.MonthLocator(tz="UTC"))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y", tz="UTC"))
    ax_price.set_xlabel("date (UTC)")
    fig.suptitle("Demand, PV and price, Jul 2013 – Jun 2014 (from hourly data)")
    out = FIGURES / BALANCE_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def monthly_pv_vs_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly PV production and demand totals (kWh), and net = PV − demand."""
    monthly = df[["pv", "Demand"]].resample("MS").sum()
    monthly["net"] = monthly["pv"] - monthly["Demand"]
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    return monthly


def plot_monthly_pv_vs_demand(df: pd.DataFrame):
    """Monthly PV production and demand side by side, with the difference above.

    The label over each month is PV − demand: positive where the system
    produces more than the household uses over that month.
    """
    setup_notebook()
    monthly = monthly_pv_vs_demand(df)
    print("Monthly PV production vs demand (kWh):")
    print(monthly.round(1).to_string())
    print(f"Months where PV falls short: {', '.join(monthly.index[monthly['net'] < 0])}")
    monthly.to_csv(TABLES / MONTHLY_BALANCE_NAME)
    print(f"Saved {TABLES / MONTHLY_BALANCE_NAME}")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(monthly))
    width = 0.38
    ax.bar(x - width / 2, monthly["pv"], width=width, color=COLOR_PV,
           label="PV production")
    ax.bar(x + width / 2, monthly["Demand"], width=width, color=COLOR_DEMAND,
           label="Demand (consumption)")
    for i, (pv, demand, net) in enumerate(zip(monthly["pv"], monthly["Demand"],
                                              monthly["net"])):
        ax.text(i, max(pv, demand) + 12, f"{net:+.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if net < 0 else "normal",
                color="#0b0b0b" if net < 0 else INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(m).strftime("%b\n%Y" if m[-2:] in ("01", "07") else "%b")
                        for m in monthly.index])
    ax.set_ylabel("kWh per month")
    ax.set_ylim(0, 1.18 * max(monthly["pv"].max(), monthly["Demand"].max()))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.set_title("Number above each month = PV − demand (kWh); "
                 "bold = PV falls short", loc="left", fontsize=10)
    fig.suptitle("Monthly PV production vs demand, Jul 2013 – Jun 2014")
    out = FIGURES / MONTHLY_BALANCE_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def impute_pv_mod1(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the gaps in pv_mod1 five ways and score each against pv."""
    results = impute_all(df)
    scores = evaluate(df, results)
    missing = df[TARGET].isna()
    print(f"{TARGET}: {int(missing.sum())} missing of {len(df):,} hours; "
          f"scored on those hours against {TRUTH}")
    print(scores.round(3).to_string())

    summary = distribution_summary(df, results)
    print("\nNumerical summary of each imputed data set vs the true PV series:")
    print(summary.round(4).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    scores.to_csv(TABLES / IMPUTATION_NAME)  # full precision
    summary.to_csv(TABLES / IMPUTATION_SUMMARY_NAME)
    print(f"Saved {TABLES / IMPUTATION_SUMMARY_NAME}")
    imputed = pd.DataFrame({name: s for name, s in results.items()
                            if name != "Listwise deletion"})
    imputed.insert(0, TARGET, df[TARGET])
    imputed.insert(1, TRUTH, df[TRUTH])
    imputed.to_csv(DATA_PROCESSED / IMPUTED_NAME)
    print(f"Saved {TABLES / IMPUTATION_NAME}")
    print(f"Saved {DATA_PROCESSED / IMPUTED_NAME}")

    plot_imputation(df, results)
    return scores


def plot_imputation(df: pd.DataFrame, results: dict):
    """Observed pv_mod1, its true values in the gaps, and each method's fill."""
    setup_notebook()
    window = df.loc[IMPUTATION_START:IMPUTATION_END]
    missing = window[TARGET].isna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window.index, window[TRUTH], color="#b5b4b0", linewidth=1,
            label="true value (pv)")
    ax.plot(window.index, window[TARGET], color="#0b0b0b", linewidth=1.6,
            label=f"{TARGET} observed")
    for name, color in IMPUTATION_COLORS.items():
        filled = results[name].loc[window.index][missing]
        ax.scatter(filled.index, filled, color=color, s=36, zorder=3, label=name)
    ax.scatter(window.index[missing], window[TRUTH][missing], marker="x",
               color="#0b0b0b", s=40, zorder=4, label="true value at gap")
    ax.set_ylabel("PV (kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
    ax.set_xlabel("time (UTC)")
    fig.suptitle(f"Imputed values for the gaps in {TARGET}, 14–19 July 2013")
    out = FIGURES / IMPUTATION_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def plot_interpolation_cases(df: pd.DataFrame,
                             cases=INTERPOLATION_CASES):
    """Before/after linear interpolation for each representative gap.

    Left column: pv_mod1 as recorded, with the gap left visible. Right column:
    the same hours after linear interpolation, with the filled hours marked and
    the true value (pv) shown at each gap for comparison.
    """
    setup_notebook()
    filled = linear_interpolation(df)

    fig, axes = plt.subplots(len(cases), 2, sharey="row",
                             figsize=(12, 2.7 * len(cases)))
    for row, (label, start, end) in zip(axes, cases):
        window = df.loc[start:end]
        after = filled.loc[start:end]
        gap = window[TARGET].isna()
        ax_before, ax_after = row

        ax_before.plot(window.index, window[TARGET], color=COLOR_PV,
                       linewidth=2, marker="o", markersize=4, label=f"{TARGET} observed")
        for t in window.index[gap]:
            ax_before.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                              color="#b5b4b0", alpha=0.35, linewidth=0)
        ax_before.set_title(f"{label} — before", loc="left", fontsize=10)

        ax_after.plot(after.index, after, color=COLOR_PV, linewidth=2,
                      marker="o", markersize=4, label=f"{TARGET} after interpolation")
        ax_after.scatter(after.index[gap], after[gap], color=COLOR_FILLED, s=70,
                         zorder=3, edgecolor="white", linewidth=1.5,
                         label="interpolated value")
        ax_after.scatter(window.index[gap], window[TRUTH][gap], marker="x",
                         color="#0b0b0b", s=60, zorder=4, label="true value (pv)")
        values = ";  ".join(f"{t:%H:%M} {after[t]:.2f} / {window[TRUTH][t]:.2f}"
                            for t in window.index[gap])
        ax_after.set_title(f"after — interpolated / true (kWh): {values}",
                           loc="left", fontsize=10)

        for ax in row:
            ax.set_ylim(bottom=0)
            ax.margins(y=0.25)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz="UTC"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
        ax_before.set_ylabel("PV (kWh)")

    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("time (UTC); grey band = missing hour")
    axes[-1, 1].set_xlabel("time (UTC)")
    fig.suptitle(f"{TARGET} before and after linear interpolation — "
                 "six representative gaps", y=1.0)
    fig.tight_layout()
    out = FIGURES / INTERPOLATION_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def impossible_values(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Imputed values that break a physical limit, one row per method.

    Limits: PV cannot be negative, cannot exceed the highest output the
    system ever recorded, and cannot be above zero while the sun is down.
    """
    missing = df[TARGET].isna()
    dark = sun_down(df)[missing]
    capacity = df[TRUTH].max()
    rows = []
    for name, series in results.items():
        if name == "Listwise deletion":
            continue
        v = series[missing]
        at_night = v[dark & (v > 0.001)]
        rows.append({
            "method": name,
            "n_negative": int((v < 0).sum()),
            "n_above_capacity": int((v > capacity).sum()),
            "n_sun_down_gaps": int(dark.sum()),
            "n_pv_while_sun_down": int(at_night.size),
            "max_pv_while_sun_down": at_night.max() if at_night.size else 0.0,
            "worst_timestamp": at_night.idxmax() if at_night.size else pd.NaT,
        })
    return pd.DataFrame(rows).set_index("method")


def plot_impossible_values(df: pd.DataFrame, results: dict,
                           cases=IMPOSSIBLE_CASES):
    """One panel per example: the imputed value against the true zero."""
    setup_notebook()
    radiation = df["direct_radiation (W/m²)"] + df["diffuse_radiation (W/m²)"]
    fig, axes = plt.subplots(1, len(cases), figsize=(12, 4.2))
    for ax, (method, gap, start, end) in zip(axes, cases):
        window = df.loc[start:end]
        filled = results[method].loc[start:end]
        t = pd.Timestamp(gap, tz="UTC")
        dark = sun_down(df).loc[start:end]

        for h in window.index[dark]:
            ax.axvspan(h - pd.Timedelta(minutes=30), h + pd.Timedelta(minutes=30),
                       color="#d9d8d4", alpha=0.5, linewidth=0)
        ax.plot(filled.index, filled, color=COLOR_PV, linewidth=2, marker="o",
                markersize=4, label=f"{TARGET} after {method.lower()}")
        ax.scatter([t], [filled[t]], color=COLOR_FILLED, s=80, zorder=3,
                   edgecolor="white", linewidth=1.5, label="imputed value")
        ax.scatter([t], [window.loc[t, TRUTH]], marker="x", color="#0b0b0b",
                   s=60, zorder=4, label="true value (pv)")
        ax.set_title(f"{method}: {t:%d %b %Y %H:%M} UTC → {filled[t]:.3f} kWh\n"
                     f"radiation {radiation[t]:.0f} W/m², true PV {window.loc[t, TRUTH]:.3f}",
                     loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.2)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz="UTC"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
        ax.set_xlabel("time (UTC); grey = sun down")
    axes[0].set_ylabel("PV (kWh)")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Physically impossible imputed values: PV produced while the sun is down")
    fig.tight_layout()
    out = FIGURES / IMPOSSIBLE_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def check_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count and plot physically impossible values from each imputation."""
    results = impute_all(df)
    table = impossible_values(df, results)
    print(table.to_string(float_format="{:.3f}".format))
    table.to_csv(TABLES / IMPOSSIBLE_NAME)
    print(f"Saved {TABLES / IMPOSSIBLE_NAME}")
    plot_impossible_values(df, results)
    return table


def live_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Score imputations that use only data available at each hour."""
    results = impute_live(df)
    scores = evaluate_live(df, results)
    print("Live (causal) methods, scored on the missing hours against pv;")
    print("offline linear interpolation is shown for reference only:")
    print(scores.to_string(float_format="{:.3f}".format))

    scores.to_csv(TABLES / LIVE_NAME)  # full precision
    imputed = pd.DataFrame(results)
    imputed.insert(0, TARGET, df[TARGET])
    imputed.insert(1, TRUTH, df[TRUTH])
    imputed.to_csv(DATA_PROCESSED / LIVE_IMPUTED_NAME)
    print(f"Saved {TABLES / LIVE_NAME}")
    print(f"Saved {DATA_PROCESSED / LIVE_IMPUTED_NAME}")
    return scores


def describe_demand_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Statistics, driver ranking and three figures for demand and weather."""
    summary = describe(df)
    ranking = rank_drivers(df)
    by_temp = demand_by_temperature(df)

    print("Summary statistics:")
    print(summary.to_string(float_format="{:.2f}".format))
    print(f"\nMax |shortwave − (direct + diffuse)| where present: "
          f"{shortwave_check(df):.3f} W/m²")
    print("\nSpearman correlation with demand, ranked by |daily rho|:")
    print(ranking.to_string(float_format="{:.3f}".format))
    print("\nMean daily demand by daily mean temperature:")
    print(by_temp.to_string(float_format="{:.3f}".format))

    TABLES.mkdir(parents=True, exist_ok=True)
    for table, name in ((summary, WEATHER_SUMMARY_NAME),
                        (ranking, WEATHER_RANKING_NAME),
                        (by_temp, DEMAND_TEMPERATURE_NAME)):
        table.to_csv(TABLES / name)  # full precision
        print(f"Saved {TABLES / name}")

    plot_demand_histogram(df)
    plot_demand_heatmap(df)
    plot_demand_temperature_bars(df)
    return ranking


def plot_demand_histogram(df: pd.DataFrame):
    """Histogram of hourly demand with the median, mean and tail marked."""
    setup_notebook()
    d = df["Demand"]
    median, mean = d.median(), d.mean()
    above_1 = (d > 1).mean()
    cutoff = 2.5
    n_beyond = int((d > cutoff).sum())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(d.clip(upper=cutoff), bins=np.arange(0, cutoff + 0.05, 0.05),
            color=COLOR_DEMAND, edgecolor="white", linewidth=0.5)
    ax.axvline(median, color="#0b0b0b", linewidth=1.8)
    ax.axvline(mean, color="#0b0b0b", linewidth=1.8, linestyle="--")
    top = ax.get_ylim()[1]
    ax.text(median - 0.03, top * 0.93, f"median {median:.2f} kWh\nhalf of all hours\nare below this",
            ha="right", va="top", fontsize=9, color="#0b0b0b")
    ax.text(mean + 0.03, top * 0.93, f"average {mean:.2f} kWh\npulled up by the\nfew high hours",
            ha="left", va="top", fontsize=9, color="#0b0b0b")
    ax.annotate(f"long tail: {above_1:.0%} of hours use more than 1 kWh,\n"
                f"the highest hour reached {d.max():.1f} kWh",
                xy=(1.6, top * 0.03), xytext=(1.25, top * 0.35), fontsize=9,
                color=INK_SECONDARY, arrowprops=dict(arrowstyle="->", color=INK_SECONDARY))
    ax.set_xlim(0, cutoff)
    ax.set_xlabel(f"electricity use in one hour (kWh); the last bar holds the "
                  f"{n_beyond} hours above {cutoff} kWh")
    ax.set_ylabel("number of hours")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{len(d):,} hours, Jul 2013 – Jun 2014", loc="left", fontsize=10)
    fig.suptitle("Most hours use little electricity; a few use a lot (right-skewed)")
    out = FIGURES / DEMAND_HISTOGRAM_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def plot_demand_heatmap(df: pd.DataFrame):
    """Mean demand for every month x hour-of-day cell, one sequential hue."""
    setup_notebook()
    from matplotlib.colors import LinearSegmentedColormap
    grid = (df["Demand"].groupby([df.index.month, df.index.hour]).mean()
            .unstack())
    cmap = LinearSegmentedColormap.from_list("demand", ["#fdf1ea", COLOR_DEMAND, "#6e2a0c"])
    month_names = [pd.Timestamp(2014, m, 1).strftime("%b") for m in grid.index]
    temps = df["Temperature"].groupby(df.index.month).mean()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=0.25, vmax=1.0)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([f"{n}  ({round(temps[m]) + 0:+d} °C)" for n, m in zip(month_names, grid.index)])
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("month (average temperature)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("average demand (kWh)")
    cbar.outline.set_visible(False)
    peak = grid.stack().idxmax()
    ax.set_title(f"Darkest = most use: cold-month evenings, Nov–Mar 17–21 h "
                 f"(peak {grid.max().max():.2f} kWh, {pd.Timestamp(2014, peak[0], 1):%b} "
                 f"{peak[1]:02d}:00). Nights are low all year.", loc="left", fontsize=10)
    fig.suptitle("When is electricity used? Average demand by month and hour")
    out = FIGURES / DEMAND_HEATMAP_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def plot_demand_temperature_bars(df: pd.DataFrame):
    """Average daily demand in each daily temperature band, labelled."""
    setup_notebook()
    bands = demand_by_temperature(df)
    labels = ["below −5", "−5 to 0", "0 to 5", "5 to 10", "10 to 15",
              "15 to 20", "above 20"]
    x = np.arange(len(bands))
    overall = daily_means(df)["Demand"].mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, bands["mean_demand_kWh"], width=0.7, color=COLOR_DEMAND)
    for xi, (v, n) in enumerate(zip(bands["mean_demand_kWh"], bands["n_days"])):
        ax.text(xi, v - 0.015, f"{v:.2f}", ha="center", va="top", fontsize=11,
                color="white", fontweight="bold")
        ax.text(xi, 0.02, f"{n} days", ha="center", va="bottom", fontsize=8,
                color="white")
    ax.axhline(overall, color="#0b0b0b", linewidth=1, linestyle="--")
    ax.text(len(bands) - 0.55, overall + 0.008, f"year average {overall:.2f} kWh",
            ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("daily average outdoor temperature (°C)")
    ax.set_ylabel("average demand per hour (kWh)")
    ax.set_ylim(0, bands["mean_demand_kWh"].max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Cool days (0–5 °C) use the most, about a third more than warm days (15–20 °C). "
                 "Extremes have few days.", loc="left", fontsize=10)
    fig.suptitle("Colder days use more electricity, up to a point")
    out = FIGURES / DEMAND_TEMPERATURE_BARS_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def analyse_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Normality checks, transformations and three figures."""
    normality = normality_table(df)
    daily = df[["Demand"]].resample("D").mean().rename(columns={"Demand": "Demand, daily mean"})
    normality = pd.concat([normality, normality_table(daily, ["Demand, daily mean"])])
    print("Normality (n = hours; tests reject at p < 0.05):")
    print(normality.to_string(float_format="{:.3g}".format))

    demand = demand_transforms(df["Demand"])
    demand_table = transform_table(demand)
    print("\nHourly demand transformations:")
    print(demand_table.to_string(float_format="{:.3g}".format))

    weather = {
        "Wind speed": df[WIND], "sqrt(Wind speed)": np.sqrt(df[WIND]),
    }
    for col in RADIATION:
        short = col.split(" (")[0].replace("_", " ")
        day = df[col].where(df[col] > 0)
        weather[f"{short}, daytime"] = day
        weather[f"sqrt({short}), daytime"] = np.sqrt(day)
    weather_table = transform_table(weather)
    print("\nWeather transformations:")
    print(weather_table.to_string(float_format="{:.3g}".format))

    transformed = apply_transformations(df)
    TABLES.mkdir(parents=True, exist_ok=True)
    normality.to_csv(TABLES / NORMALITY_NAME)
    demand_table.to_csv(TABLES / DEMAND_TRANSFORM_NAME)
    weather_table.to_csv(TABLES / WEATHER_TRANSFORM_NAME)
    transformed.to_csv(DATA_PROCESSED / TRANSFORMED_NAME)
    for path in (TABLES / NORMALITY_NAME, TABLES / DEMAND_TRANSFORM_NAME,
                 TABLES / WEATHER_TRANSFORM_NAME, DATA_PROCESSED / TRANSFORMED_NAME):
        print(f"Saved {path}")

    plot_distributions_vs_normal(df, daily, normality)
    plot_demand_transformation(df, demand)
    plot_weather_transformations(df)
    return normality


def _hist_with_normal(ax, x: pd.Series, color: str, bins=40):
    """Density histogram with the normal curve of the same mean and std."""
    x = x.dropna()
    ax.hist(x, bins=bins, density=True, color=color, edgecolor="white",
            linewidth=0.4)
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, stats.norm.pdf(grid, x.mean(), x.std()), color=COLOR_NORMAL,
            linewidth=1.4, linestyle="--")
    ax.set_yticks([])
    ax.grid(False)


def plot_distributions_vs_normal(df: pd.DataFrame, daily: pd.DataFrame,
                                 normality: pd.DataFrame):
    """One histogram per variable with the matching normal curve dashed."""
    setup_notebook()
    columns = ["Demand", *[c for c in WEATHER_COLUMNS]]
    panels = [(df[c], SHORT_NAMES[c]) for c in columns]
    panels.append((daily["Demand, daily mean"], "Demand, daily mean"))
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, (x, name) in zip(axes.flat, panels):
        row = normality.loc[name]
        color = COLOR_DEMAND if name.startswith("Demand") else COLOR_PV
        _hist_with_normal(ax, x, color)
        verdict = "≈ normal" if row["close_to_normal"] else "not normal"
        ax.set_title(f"{name}\nskew {row['skew']:+.2f}, kurt {row['excess_kurtosis']:+.2f} → {verdict}",
                     fontsize=9, loc="left")
    fig.suptitle("Each variable vs a normal curve (dashed) with the same mean and spread")
    fig.text(0.5, -0.01, "≈ normal = |skew| < 0.5 and |excess kurtosis| < 1; "
             "orange = demand, blue = weather", ha="center", fontsize=9,
             color=INK_SECONDARY)
    fig.tight_layout()
    out = FIGURES / NORMALITY_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def plot_demand_transformation(df: pd.DataFrame, demand: dict):
    """Histogram (top) and Q-Q plot (bottom) for demand before and after."""
    setup_notebook()
    names = ["Original", "Log", next(k for k in demand if k.startswith("Box-Cox"))]
    series = [demand[n] for n in names]
    daily = df["Demand"].resample("D").mean()
    names.append("Daily mean (no transform)")
    series.append(daily)

    fig, axes = plt.subplots(2, len(names), figsize=(15, 7))
    for col, (name, x) in enumerate(zip(names, series)):
        x = x.dropna()
        _hist_with_normal(axes[0, col], x, COLOR_DEMAND)
        axes[0, col].set_title(f"{name}\nskew {x.skew():+.2f}, kurt {x.kurt():+.2f}",
                               fontsize=10, loc="left")
        (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
        ax = axes[1, col]
        ax.scatter(osm, osr, s=4, color=COLOR_DEMAND, alpha=0.5, linewidth=0)
        ax.plot(osm, slope * osm + intercept, color=COLOR_NORMAL, linewidth=1.2,
                linestyle="--")
        ax.set_xlabel("normal quantiles")
        if col == 0:
            ax.set_ylabel("data quantiles")
    axes[0, 0].set_ylabel("density")
    fig.suptitle("Hourly demand before and after transformation — points on the "
                 "dashed line = normal")
    fig.tight_layout()
    out = FIGURES / DEMAND_TRANSFORM_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def plot_weather_transformations(df: pd.DataFrame):
    """Wind speed and daytime diffuse radiation, raw vs square root."""
    setup_notebook()
    diffuse = df["diffuse_radiation (W/m²)"]
    day = diffuse.where(diffuse > 0)
    pairs = [
        (df[WIND], "Wind speed (km/h)"),
        (np.sqrt(df[WIND]), "√ wind speed"),
        (day, "Diffuse radiation, daytime only (W/m²)"),
        (np.sqrt(day), "√ diffuse radiation, daytime only"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (x, name) in zip(axes, pairs):
        _hist_with_normal(ax, x, COLOR_PV)
        x = x.dropna()
        ax.set_title(f"{name}\nskew {x.skew():+.2f}, kurt {x.kurt():+.2f}",
                     fontsize=9, loc="left")
    share_zero = (diffuse == 0).mean()
    fig.suptitle(f"Square root evens out right-skewed weather variables "
                 f"(radiation: {share_zero:.0%} night zeros handled by a day/night flag)")
    fig.tight_layout()
    out = FIGURES / WEATHER_TRANSFORM_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, calendar and weather features; justify the lag with the ACF."""
    acf = autocorrelation(df["Demand"], range(1, ACF_MAX_LAG + 1))
    usable = acf.loc[HORIZON_HOURS:]
    print(f"Spearman autocorrelation of demand at selected lags (hours):")
    print(acf.loc[[1, 2, 23, 24, 25, 48, 72, 168, 336]].to_string(float_format="{:.3f}".format))
    print(f"Strongest lag usable day-ahead (>= {HORIZON_HOURS} h): "
          f"{usable.idxmax()} h, rho = {usable.max():.3f}")

    candidates = candidate_features(df)
    features = modelling_set(candidates)
    print("\nWhen each feature becomes known (prediction time = 00:00 on day D):")
    print(AVAILABILITY[["known_at", "lead_time_h", "available"]].to_string())
    dropped = [c for c in candidates.columns if c not in features.columns]
    print(f"Removed from the modelling set (0 h lead time): {dropped}")
    relevance = feature_relevance(candidates)
    print("\nSpearman correlation of each feature with demand:")
    print(relevance.to_string(float_format="{:.3f}".format))
    print("\nMissing values per feature (start of series and the missing hour):")
    print(features.isna().sum().to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    features.to_csv(DATA_PROCESSED / FEATURES_NAME)
    relevance.to_csv(TABLES / FEATURE_RELEVANCE_NAME)
    AVAILABILITY.to_csv(TABLES / FEATURE_AVAILABILITY_NAME)
    print(f"Saved {TABLES / FEATURE_AVAILABILITY_NAME}")
    acf.rename_axis("lag_hours").to_csv(TABLES / ACF_NAME)
    for path in (DATA_PROCESSED / FEATURES_NAME, TABLES / FEATURE_RELEVANCE_NAME,
                 TABLES / ACF_NAME):
        print(f"Saved {path}")

    plot_autocorrelation(acf)
    return features


def plot_autocorrelation(acf: pd.Series):
    """Autocorrelation by lag, with the lags not usable day-ahead shaded."""
    setup_notebook()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axvspan(0.5, HORIZON_HOURS - 0.5, color="#d9d8d4", alpha=0.6, linewidth=0)
    ax.text(HORIZON_HOURS / 2 + 2, 0.32, "not known\nday-ahead", ha="center", va="top",
            fontsize=9, color=INK_SECONDARY)
    ax.vlines(acf.index, 0, acf.values, color=COLOR_DEMAND, linewidth=1.2)
    for k, label in ((1, "1 h"), (LAG_HOURS, "24 h (chosen)"),
                     (48, "48 h"), (168, "168 h (1 week)"),
                     (336, "336 h")):
        ax.scatter([k], [acf[k]], color="#0b0b0b", s=30, zorder=3)
        ax.annotate(f"{label}\nρ = {acf[k]:.2f}", (k, acf[k]), textcoords="offset points",
                    xytext=(8, -4) if k == 1 else (6, 6), fontsize=9,
                    fontweight="bold" if k == LAG_HOURS else "normal")
    ax.set_xlim(0, ACF_MAX_LAG + 8)
    ax.set_ylim(min(0, acf.min() - 0.03), 0.8)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(range(0, ACF_MAX_LAG + 1, 24))
    ax.set_xlabel("lag (hours); ticks every 24 h")
    ax.set_ylabel("Spearman correlation of demand\nwith itself k hours earlier")
    ax.grid(axis="x", visible=False)
    ax.set_title("Peaks every 24 h = the daily routine repeats; the 24 h peak is the "
                 "highest usable one", loc="left", fontsize=10)
    fig.suptitle("Demand autocorrelation: which lag to use for a day-ahead forecast")
    out = FIGURES / ACF_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def rank_modelling_features(features: pd.DataFrame) -> pd.DataFrame:
    """Rank the modelling features by three measures and plot them."""
    ranking = rank_features(features)
    print("Feature ranking (higher = more relevant; final rank = mean of three ranks):")
    print(ranking.to_string(float_format="{:.4f}".format))
    ranking.to_csv(TABLES / FEATURE_RANKING_NAME)
    print(f"Saved {TABLES / FEATURE_RANKING_NAME}")
    plot_feature_ranking(ranking)
    return ranking


def plot_feature_ranking(ranking: pd.DataFrame):
    """One panel per ranking method, features in final-rank order."""
    setup_notebook()
    panels = [("spearman_abs_rho", "Spearman |ρ|\n(straight-line-like link)"),
              ("mutual_information", "Mutual information\n(any kind of link)"),
              ("permutation_mae_increase", "Permutation importance\n(MAE increase, kWh, unseen months)")]
    order = ranking.index[::-1]
    labels = [f"{ranking.loc[f, 'final_rank']}. {f}" for f in order]
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(13, 3.6))
    for ax, (col, title) in zip(axes, panels):
        values = ranking.loc[order, col]
        ax.barh(range(len(order)), values, color=COLOR_DEMAND, height=0.6)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, values.max() * 1.3)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#888888", linewidth=0.8)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels)
    fig.suptitle("Feature ranking: hour of day and yesterday's demand lead, "
                 "the heating feature trails")
    fig.tight_layout()
    out = FIGURES / FEATURE_RANKING_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def define_split(df: pd.DataFrame) -> pd.DataFrame:
    """Label each hour train / validation / test and summarise the sets."""
    frame = split_frame(df)
    summary = split_summary(df, frame)
    print("Train / validation / test split (time order, whole months):")
    print(summary.to_string(float_format="{:.2f}".format))
    unassigned = int(frame["split"].isna().sum())
    print(f"Hours not in any set: {unassigned}")

    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / SPLIT_SUMMARY_NAME)
    frame.to_csv(DATA_PROCESSED / SPLIT_LABELS_NAME)
    print(f"Saved {TABLES / SPLIT_SUMMARY_NAME}")
    print(f"Saved {DATA_PROCESSED / SPLIT_LABELS_NAME}")

    plot_split(df, frame)
    return frame


def plot_split(df: pd.DataFrame, frame: pd.DataFrame):
    """Daily mean demand over the year, coloured by set."""
    setup_notebook()
    daily = df["Demand"].resample("D").mean()
    split = frame["split"].resample("D").first()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for name, start, end in SPLITS:
        part = daily[split == name]
        ax.plot(part.index, part, color=SPLIT_COLORS[name], linewidth=1.6)
        ax.axvspan(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                   color=SPLIT_COLORS[name], alpha=0.08, linewidth=0)
        mid = pd.Timestamp(start, tz="UTC") + (pd.Timestamp(end, tz="UTC")
                                               - pd.Timestamp(start, tz="UTC")) / 2
        hours = int((frame["split"] == name).sum())
        ax.text(mid, 0.98, f"{name}\n{pd.Timestamp(start):%d %b %Y}\n"
                f"– {pd.Timestamp(end):%d %b %Y}\n{hours:,} h",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color="#0b0b0b")
    ax.set_ylabel("daily mean demand (kWh)")
    ax.set_ylim(0, daily.max() * 1.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y", tz="UTC"))
    ax.set_xlabel("date (UTC)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Train / validation / test split in time order: 9 / 2 / 1 months")
    out = FIGURES / SPLIT_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def define_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Daily walk-forward folds over the last week of the training file."""
    folds = walk_forward_folds(df.index)
    print("Walk-forward validation, last week, one fold per day (expanding window):")
    print(folds.to_string())
    folds.to_csv(TABLES / WALK_FORWARD_NAME)
    print(f"Saved {TABLES / WALK_FORWARD_NAME}")
    plot_walk_forward(df, folds)
    return folds


def plot_walk_forward(df: pd.DataFrame, folds: pd.DataFrame):
    """One row per fold: training hours in blue, the forecast day in orange."""
    setup_notebook()
    first_test = folds["test_start"].iloc[0]
    view_start = first_test - pd.Timedelta(days=WALK_FORWARD_CONTEXT_DAYS)
    view_end = folds["test_end"].iloc[-1] + pd.Timedelta(hours=1)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for fold, row in folds.iterrows():
        y = len(folds) - fold
        ax.barh(y, row["train_end"] + pd.Timedelta(hours=1) - view_start,
                left=view_start, height=0.6, color=COLOR_PV)
        ax.barh(y, pd.Timedelta(hours=24), left=row["test_start"], height=0.6,
                color=COLOR_DEMAND)
        ax.text(view_end + pd.Timedelta(hours=3), y,
                f"forecast {row['test_start']:%a %d %b}  "
                f"(train {row['train_hours']:,} h)", va="center", fontsize=9)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"fold {f}" for f in folds.index[::-1]])
    ax.set_xlim(view_start, view_end)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b", tz="UTC"))
    ax.set_xlabel("date (UTC); training continues back to 1 Jul 2013 (cut off on the left)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR_PV, label="training data (all earlier hours)"),
                       Patch(color=COLOR_DEMAND, label="forecast day (24 h test)")],
              loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2, fontsize=9,
              frameon=False)
    fig.suptitle("Walk-forward validation: last week of the training file, one fold per day")
    out = FIGURES / WALK_FORWARD_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def predict_baseline_nrmse(df: pd.DataFrame) -> dict:
    """Predict the seasonal naive (24 h) nRMSE from training data only,
    before any baseline is scored."""
    prediction = predict_seasonal_naive_nrmse(df["Demand"], split_frame(df))
    print("Predicted nRMSE of the seasonal naive 24 h baseline (training set only):")
    print(f"  mean = {prediction['train_mean']:.4f} kWh, σ = {prediction['train_std']:.4f} kWh, "
          f"σ/mean = {prediction['cv']:.3f}")
    print(f"  Pearson ρ at lag 24 h = {prediction['pearson_rho']:.3f}, "
          f"√(2(1−ρ)) = {prediction['sqrt_2_1_minus_rho']:.3f}")
    print(f"  predicted nRMSE = {prediction['cv']:.3f} × "
          f"{prediction['sqrt_2_1_minus_rho']:.3f} = {prediction['predicted_nRMSE']:.3f}")
    pd.Series(prediction).to_csv(TABLES / NRMSE_PREDICTION_NAME, header=["value"])
    print(f"Saved {TABLES / NRMSE_PREDICTION_NAME}")
    return prediction


def compare_nrmse_prediction(prediction: dict, scores: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs actual seasonal naive nRMSE, with the prediction error."""
    actual = scores[(scores["baseline"] == SEASONAL_NAIVE_NAME)
                    & scores["unit"].str.contains(r"\(all")].set_index("group")["nRMSE"]
    table = pd.DataFrame({"predicted_nRMSE": prediction["predicted_nRMSE"],
                          "actual_nRMSE": actual})
    table["error"] = table["predicted_nRMSE"] - table["actual_nRMSE"]
    table["relative_error_pct"] = 100 * table["error"] / table["actual_nRMSE"]
    print("Seasonal naive 24 h: predicted vs actual nRMSE (error = predicted − actual):")
    print(table.to_string(float_format="{:.3f}".format))
    table.to_csv(TABLES / NRMSE_COMPARISON_NAME)
    print(f"Saved {TABLES / NRMSE_COMPARISON_NAME}")
    return table


def evaluation_units(df: pd.DataFrame, frame: pd.DataFrame) -> list:
    """(unit name, set, hours) for every scored unit of the protocol:
    validation weeks, test weeks, walk-forward days, and each set as a whole."""
    units = []
    for name in ("validation", "test"):
        in_set = frame["split"] == name
        for block, idx in frame.index[in_set].groupby(frame.loc[in_set, "week_block"]).items():
            units.append((f"{name} week {block}", name, idx))
        units.append((f"{name} (all)", name, frame.index[in_set]))
    folds = walk_forward_folds(df.index)
    for fold, f in folds.iterrows():
        idx = df.index[(df.index >= f["test_start"]) & (df.index <= f["test_end"])]
        units.append((f"walk-forward day {fold} ({f['test_start']:%a %d %b})",
                      "walk-forward", idx))
    wf = df.index[(df.index >= folds["test_start"].min())
                  & (df.index <= folds["test_end"].max())]
    units.append(("walk-forward (all 7 days)", "walk-forward", wf))
    return units


def baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score every baseline with MAE and nRMSE on every unit of the protocol.

    nRMSE is RMSE divided by the training-set mean demand, everywhere.
    Outage hours are not scored.
    """
    frame = split_frame(df)
    usable = ~frame["outage"]
    actual = df["Demand"].where(usable)
    forecasts = all_baselines(df["Demand"])
    normaliser = training_mean(df["Demand"], frame)
    print(f"nRMSE denominator = training-set mean demand = {normaliser:.4f} kWh "
          "(same value for every score)")

    rows = []
    for unit, group, idx in evaluation_units(df, frame):
        normal = idx[~frame.loc[idx, "absent_day"].to_numpy()]
        for name in forecasts.columns:
            f = forecasts[name].where(usable)
            rows.append({"unit": unit, "group": group, "baseline": name,
                         **score(actual[idx], f[idx], normaliser),
                         "MAE_normal_days_kWh": score(actual[normal], f[normal],
                                                      normaliser)["MAE_kWh"]})
    table = pd.DataFrame(rows)
    table["normaliser_kWh"] = normaliser
    table.to_csv(TABLES / BASELINE_SCORES_NAME, index=False)

    overall = table[table["unit"].str.contains(r"\(all")].pivot_table(
        index="baseline", columns="unit", values=["MAE_kWh", "nRMSE"], sort=False)
    print("Baselines, overall per evaluation set:")
    print(overall.to_string(float_format="{:.3f}".format))
    wf_days = table[table["unit"].str.startswith("walk-forward day")].pivot_table(
        index="unit", columns="baseline", values="MAE_kWh", sort=False)
    print("\nWalk-forward MAE (kWh) per day:")
    print(wf_days.to_string(float_format="{:.3f}".format))
    print(f"Saved {TABLES / BASELINE_SCORES_NAME}")

    plot_baselines_walk_forward(df, forecasts)
    return table


def plot_baselines_walk_forward(df: pd.DataFrame, forecasts: pd.DataFrame):
    """Actual demand vs each baseline over the walk-forward week, one panel each."""
    setup_notebook()
    folds = walk_forward_folds(df.index)
    start, end = folds["test_start"].min(), folds["test_end"].max()
    actual = df["Demand"].loc[start:end]
    fig, axes = plt.subplots(len(forecasts.columns), 1, sharex=True, sharey=True,
                             figsize=(12, 9))
    for ax, name in zip(axes, forecasts.columns):
        f = forecasts[name].loc[start:end]
        err = (f - actual).abs().mean()
        ax.plot(actual.index, actual, color="#0b0b0b", linewidth=1.2, label="actual")
        ax.plot(f.index, f, color=COLOR_DEMAND, linewidth=1.8, label=name)
        ax.set_title(f"{name} — MAE {err:.3f} kWh", loc="left", fontsize=10)
        ax.set_ylabel("kWh")
        ax.set_ylim(bottom=0)
        for day in folds["test_start"]:
            ax.axvline(day, color="#d9d8d4", linewidth=0.8, zorder=0)
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz="UTC"))
    axes[-1].set_xlabel("walk-forward week, one fold per day (UTC)")
    fig.suptitle("Baselines vs actual demand, last week of the training file")
    fig.tight_layout()
    out = FIGURES / BASELINES_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def test_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """ADF + KPSS on training-set demand and its differences; pick d and D."""
    y = training_series(df["Demand"], split_frame(df))
    table = stationarity_table(y)
    cols = ["ADF_stat", "ADF_p", "KPSS_stat", "KPSS_p", "both_agree_stationary",
            "variance", "acf_1", "acf_24", "acf_168"]
    print(f"Training set only, {len(y):,} hours (outage and missing hour interpolated).")
    print("ADF: H0 unit root (p < 0.05 → stationary). KPSS: H0 stationary "
          "(p < 0.05 → not stationary). KPSS p-values are capped at 0.01–0.10.")
    print(table[cols].to_string(float_format="{:.3f}".format))
    print(f"Chosen: {CHOSEN_DIFFERENCING} → d = 1, D = 0")
    table.to_csv(TABLES / STATIONARITY_NAME)
    print(f"Saved {TABLES / STATIONARITY_NAME}")
    plot_stationarity(y, table)
    return table


def plot_stationarity(y: pd.Series, table: pd.DataFrame):
    """Each candidate: a two-week slice (left) and its autocorrelation (right)."""
    from statsmodels.tsa.stattools import acf
    setup_notebook()
    fig, axes = plt.subplots(len(STATIONARITY_PANELS), 2, figsize=(13, 8),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    window = slice("2014-02-03", "2014-02-16")
    for (ax_ts, ax_acf), name in zip(axes, STATIONARITY_PANELS):
        s = difference(y, CANDIDATES[name])
        row = table.loc[name]
        chosen = name == CHOSEN_DIFFERENCING
        color = COLOR_DEMAND if chosen else COLOR_PV
        ax_ts.plot(s.loc[window].index, s.loc[window], color=color, linewidth=1)
        ax_ts.axhline(0 if name != STATIONARITY_PANELS[0] else s.mean(),
                      color="#888888", linewidth=0.8)
        verdict = "both tests: stationary" if row["both_agree_stationary"] else "tests disagree"
        ax_ts.set_title(f"{name}{'  ← chosen' if chosen else ''}\n"
                        f"ADF {row['ADF_stat']:.1f} (p {row['ADF_p']:.3f}), "
                        f"KPSS {row['KPSS_stat']:.3f} (p {row['KPSS_p']:.2f}) → {verdict}; "
                        f"variance {row['variance']:.3f}",
                        loc="left", fontsize=9, fontweight="bold" if chosen else "normal")
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
        r = acf(s, nlags=192)
        ax_acf.vlines(range(1, 193), 0, r[1:], color=color, linewidth=1)
        ax_acf.axhline(0, color="#888888", linewidth=0.8)
        for k in (24, 168):
            ax_acf.annotate(f"lag {k}: {r[k]:+.2f}", (k, r[k]), textcoords="offset points",
                            xytext=(4, 6 if r[k] >= 0 else -12), fontsize=8)
        ax_acf.set_ylim(-0.65, 0.65)
        ax_acf.set_xticks(range(0, 193, 24))
        ax_acf.set_title("autocorrelation (lags 1–192 h)", loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("two-week example, Feb 2014")
    axes[-1, 1].set_xlabel("lag (hours)")
    fig.suptitle("Stationarity of demand, training set: level vs differenced")
    fig.tight_layout()
    out = FIGURES / STATIONARITY_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def analyse_acf_pacf(df: pd.DataFrame) -> pd.DataFrame:
    """ACF and PACF of the first-differenced training demand, to read (p, q)."""
    y = training_series(df["Demand"], split_frame(df))
    dy = difference(y, CANDIDATES[CHOSEN_DIFFERENCING])
    table = acf_pacf_table(dy, ACF_PACF_LONG)
    bound = table["bound"].iloc[0]
    print(f"First-differenced training demand, n = {len(dy):,}; "
          f"95 % bound ±1.96/√n = ±{bound:.3f}")
    print(table.loc[[1, 2, 3, 4, 5, 6, 12, 23, 24, 25, 48, 72, 168], ["acf", "pacf"]]
          .to_string(float_format="{:+.3f}".format))
    table.to_csv(TABLES / ACF_PACF_NAME)
    print(f"Saved {TABLES / ACF_PACF_NAME}")
    plot_acf_pacf(table, len(dy))
    return table


def plot_acf_pacf(table: pd.DataFrame, n: int):
    """ACF and PACF, short lags (top) and long lags (bottom), with ±1.96/√n."""
    setup_notebook()
    bound = table["bound"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for row, max_lag in enumerate((ACF_PACF_SHORT, ACF_PACF_LONG)):
        for col, kind in enumerate(("acf", "pacf")):
            ax = axes[row, col]
            t = table.loc[:max_lag]
            ax.vlines(t.index, 0, t[kind], color=COLOR_DEMAND,
                      linewidth=2 if max_lag == ACF_PACF_SHORT else 1)
            ax.axhspan(-bound, bound, color="#d9d8d4", alpha=0.6, linewidth=0)
            ax.axhline(0, color="#888888", linewidth=0.8)
            ax.set_ylim(-0.3, 0.12)
            step = 6 if max_lag == ACF_PACF_SHORT else 24
            ax.set_xticks(range(0, max_lag + 1, step))
            ax.set_xlim(0, max_lag + 1)
            ax.set_title(f"{kind.upper()}, lags 1–{max_lag} h", loc="left", fontsize=10)
            ax.grid(axis="x", visible=False)
        axes[row, 0].set_ylabel("correlation")
    for k in (1, 2):
        axes[0, 0].annotate(f"lag {k}: {table.loc[k, 'acf']:+.2f}", (k, table.loc[k, "acf"]),
                            textcoords="offset points", xytext=(8, -4), fontsize=9)
    axes[0, 0].annotate("cuts off after lag 2", (3, -0.03), textcoords="offset points",
                        xytext=(20, -40), fontsize=9, color=INK_SECONDARY,
                        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY))
    axes[0, 1].annotate("decays slowly over ~24 lags", (8, table.loc[8, "pacf"]),
                        textcoords="offset points", xytext=(30, -45), fontsize=9,
                        color=INK_SECONDARY,
                        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY))
    axes[1, 0].annotate("small spikes every 24 h\n(+0.03 to +0.08)", (48, table.loc[48, "acf"]),
                        textcoords="offset points", xytext=(30, 10), fontsize=9,
                        color=INK_SECONDARY,
                        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY))
    for ax in axes[1]:
        ax.set_xlabel("lag (hours)")
    fig.suptitle(f"ACF and PACF of first-differenced demand (training set, n = {n:,}); "
                 f"grey band = ±1.96/√n = ±{bound:.3f}")
    fig.tight_layout()
    out = FIGURES / ACF_PACF_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def train_arima_models(df: pd.DataFrame) -> pd.DataFrame:
    """Fit ARMA-family models (orders from ACF/PACF and from AIC search) and
    score their day-ahead forecasts under the protocol."""
    frame = split_frame(df)
    y = model_series(df["Demand"], frame["outage"].reindex(df.index))
    train_end = pd.Timestamp(SPLITS[0][2], tz="UTC")
    val_end = pd.Timestamp(SPLITS[1][2], tz="UTC")

    search = aic_search(y.loc[:train_end])
    best = search.iloc[0]
    print("AIC search over ARIMA(p,1,q), p,q = 0..3, training set only (best 5):")
    print(search.head().to_string(index=False, float_format="{:.1f}".format))
    models = dict(READ_FROM_PLOTS)
    auto_name = f"ARIMA({int(best.p)},1,{int(best.q)})"
    models[auto_name] = ((int(best.p), 1, int(best.q)), (0, 0, 0, 0), "AIC search")
    search.to_csv(TABLES / ARIMA_SEARCH_NAME, index=False)

    days = {name: pd.DatetimeIndex(frame.index[frame["split"] == name].floor("D").unique())
            for name in ("validation", "test")}
    folds = walk_forward_folds(df.index)
    forecasts, params = {}, []
    for name, (order, seasonal, source) in models.items():
        fitted_train = fit(y.loc[:train_end], order, seasonal)
        fitted_trval = fit(y.loc[:val_end], order, seasonal)
        params.append({"model": name, "source": source, "AIC_train": fitted_train.aic,
                       **fitted_train.params.round(4).to_dict()})
        pieces = [day_ahead(fitted_train, y, days["validation"]),
                  day_ahead(fitted_trval, y, days["test"])]
        wf = []
        for _, f in folds.iterrows():
            refit = fit(y.loc[:f["train_end"]], order, seasonal)
            wf.append(day_ahead(refit, y, pd.DatetimeIndex([f["test_start"]])))
        forecasts[name] = (pd.concat(pieces).reindex(df.index), pd.concat(wf).reindex(df.index))
        print(f"Fitted {name} ({source})")

    params = pd.DataFrame(params).set_index("model")
    print("\nParameters fitted on the training set:")
    print(params.to_string(float_format="{:.3f}".format))
    params.to_csv(TABLES / ARIMA_PARAMS_NAME)

    usable = ~frame["outage"]
    actual = df["Demand"].where(usable)
    normaliser = training_mean(df["Demand"], frame)
    rows = []
    for unit, group, idx in evaluation_units(df, frame):
        if not unit.endswith("(all 7 days)") and "(all)" not in unit:
            continue
        for name, (split_fc, wf_fc) in forecasts.items():
            fc = (wf_fc if group == "walk-forward" else split_fc).where(usable)
            rows.append({"set": group, "model": name, "source": models[name][2],
                         **score(actual[idx], fc[idx], normaliser)})
    scores = pd.DataFrame(rows)
    baselines = baseline_scores_quiet(df)
    scores = pd.concat([scores, baselines], ignore_index=True)
    table = scores.pivot_table(index=["model", "source"], columns="set",
                               values=["MAE_kWh", "nRMSE"], sort=False)
    print("\nDay-ahead scores (nRMSE = RMSE / 0.5374 kWh):")
    print(table.to_string(float_format="{:.3f}".format))
    scores.to_csv(TABLES / ARIMA_SCORES_NAME, index=False)
    out = pd.DataFrame({f"{n} (split)": v[0] for n, v in forecasts.items()}
                       | {f"{n} (walk-forward)": v[1] for n, v in forecasts.items()})
    out.insert(0, "Demand", df["Demand"])
    out.to_csv(DATA_PROCESSED / ARIMA_FORECASTS_NAME)
    for path in (TABLES / ARIMA_SEARCH_NAME, TABLES / ARIMA_PARAMS_NAME,
                 TABLES / ARIMA_SCORES_NAME, DATA_PROCESSED / ARIMA_FORECASTS_NAME):
        print(f"Saved {path}")
    plot_arima_walk_forward(df, {n: v[1] for n, v in forecasts.items()}, folds)
    return scores


def baseline_scores_quiet(df: pd.DataFrame) -> pd.DataFrame:
    """Best baseline per set and metric, in the same layout as the model scores."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        table = baseline_scores(df)
    table = table[table["unit"].str.contains(r"\(all")]
    table = table.assign(set=table["group"], model="Best baseline", source="baseline")
    best = table.groupby("set")[["MAE_kWh", "nRMSE"]].min().reset_index()
    return best.assign(model="Best baseline (per metric)", source="baseline")


def plot_arima_walk_forward(df: pd.DataFrame, forecasts: dict, folds: pd.DataFrame,
                            filename: str = ARIMA_FIGURE,
                            title: str = "ARMA-family models, day-ahead forecasts vs actual demand"):
    """Actual demand and each model's day-ahead forecast, walk-forward week."""
    setup_notebook()
    start, end = folds["test_start"].min(), folds["test_end"].max()
    actual = df["Demand"].loc[start:end]
    fig, axes = plt.subplots(len(forecasts), 1, sharex=True, sharey=True,
                             figsize=(12, 2.5 * len(forecasts) + 0.5))
    for ax, (name, fc), color in zip(axes, forecasts.items(), MODEL_COLORS):
        f = fc.loc[start:end]
        ax.plot(actual.index, actual, color="#0b0b0b", linewidth=1.1, label="actual")
        ax.plot(f.index, f, color=color, linewidth=1.8, label=name)
        ax.set_title(f"{name} — MAE {(f - actual).abs().mean():.3f} kWh",
                     loc="left", fontsize=10)
        ax.set_ylabel("kWh")
        ax.set_ylim(bottom=0)
        for day in folds["test_start"]:
            ax.axvline(day, color="#d9d8d4", linewidth=0.8, zorder=0)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz="UTC"))
    axes[-1].set_xlabel("walk-forward week; each day forecast at 00:00 from data up to 23:00 the day before")
    fig.suptitle(title)
    fig.tight_layout()
    out = FIGURES / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def compare_in_sample_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample fit vs walk-forward result for the two models and the baselines.

    In-sample (training set, parameters fitted on it):
    - one-step: the model's fitted values, each hour forecast from the hour before;
    - day-ahead: 24-hour forecasts issued at 00:00, as in the protocol.
    Walk-forward: the 7 daily folds at the end of the file, refitted each day.
    Baselines have no parameters; their "in-sample" score is their error on the
    training period.
    """
    frame = split_frame(df)
    y = model_series(df["Demand"], frame["outage"].reindex(df.index))
    usable = ~frame["outage"]
    actual = df["Demand"].where(usable)
    normaliser = training_mean(df["Demand"], frame)
    train_end = pd.Timestamp(SPLITS[0][2], tz="UTC")
    train = frame["split"] == "train"
    # Skip the first week so every baseline (incl. the 168 h lag) has a value.
    train_scored = df.index[train & (df.index >= df.index.min() + pd.Timedelta(days=7))]
    train_days = pd.DatetimeIndex(train_scored.floor("D").unique())
    folds = walk_forward_folds(df.index)
    wf = df.index[(df.index >= folds["test_start"].min()) & (df.index <= folds["test_end"].max())]

    search = pd.read_csv(TABLES / ARIMA_SEARCH_NAME).iloc[0]
    models = dict(READ_FROM_PLOTS)
    models[f"ARIMA({int(search.p)},1,{int(search.q)})"] = (
        (int(search.p), 1, int(search.q)), (0, 0, 0, 0), "AIC search")

    rows = []
    for name, (order, seasonal, source) in models.items():
        fitted = fit(y.loc[:train_end], order, seasonal)
        one_step = fitted.fittedvalues.reindex(df.index)
        day = day_ahead(fitted, y, train_days).reindex(df.index)
        wf_fc = pd.concat([
            day_ahead(fit(y.loc[:f["train_end"]], order, seasonal), y,
                      pd.DatetimeIndex([f["test_start"]]))
            for _, f in folds.iterrows()]).reindex(df.index)
        for label, fc, idx in (("in-sample, one-step (1 h)", one_step, train_scored),
                               ("in-sample, day-ahead (24 h)", day, train_scored),
                               ("walk-forward, day-ahead (24 h)", wf_fc, wf)):
            rows.append({"model": name, "source": source, "evaluation": label,
                         **score(actual[idx], fc.where(usable)[idx], normaliser)})
        print(f"Scored {name}")
    baselines = all_baselines(df["Demand"])
    for name in baselines.columns:
        fc = baselines[name].where(usable)
        for label, idx in (("in-sample, day-ahead (24 h)", train_scored),
                           ("walk-forward, day-ahead (24 h)", wf)):
            rows.append({"model": name, "source": "baseline", "evaluation": label,
                         **score(actual[idx], fc[idx], normaliser)})

    table = pd.DataFrame(rows)
    wide = table.pivot_table(index=["model", "source"], columns="evaluation",
                             values=["MAE_kWh", "nRMSE"], sort=False)
    print("In-sample fit vs walk-forward (nRMSE = RMSE / 0.5374 kWh):")
    print(wide.to_string(float_format="{:.3f}".format))
    table.to_csv(TABLES / COMPARISON_NAME, index=False)
    print(f"Saved {TABLES / COMPARISON_NAME}")
    return table


def train_sarimax_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour of day (23 dummies) to the SARIMA read from the ACF/PACF and
    score the change in MAE and nRMSE under the same protocol."""
    frame = split_frame(df)
    y = model_series(df["Demand"], frame["outage"].reindex(df.index))
    exog = hour_dummies(pd.date_range(y.index.min(), y.index.max() + pd.Timedelta(hours=24),
                                      freq="h"))
    order, seasonal, _ = READ_FROM_PLOTS[SARIMAX_BASE]
    train_end = pd.Timestamp(SPLITS[0][2], tz="UTC")
    val_end = pd.Timestamp(SPLITS[1][2], tz="UTC")
    days = {name: pd.DatetimeIndex(frame.index[frame["split"] == name].floor("D").unique())
            for name in ("validation", "test")}
    folds = walk_forward_folds(df.index)

    fitted_train = fit(y.loc[:train_end], order, seasonal, exog)
    print(f"Fitted {SARIMAX_NAME} on train, AIC {fitted_train.aic:.1f}")
    split_fc = pd.concat([
        day_ahead(fitted_train, y, days["validation"], exog),
        day_ahead(fit(y.loc[:val_end], order, seasonal, exog), y, days["test"], exog),
    ]).reindex(df.index)
    wf_fc = pd.concat([
        day_ahead(fit(y.loc[:f["train_end"]], order, seasonal, exog), y,
                  pd.DatetimeIndex([f["test_start"]]), exog)
        for _, f in folds.iterrows()]).reindex(df.index)

    params = fitted_train.params.rename("value")
    params.to_csv(TABLES / SARIMAX_PARAMS_NAME)
    print("Parameters fitted on the training set:")
    print(params.to_frame().T.to_string(float_format="{:+.3f}".format))

    usable = ~frame["outage"]
    actual = df["Demand"].where(usable)
    normaliser = training_mean(df["Demand"], frame)
    rows = []
    for unit, group, idx in evaluation_units(df, frame):
        if "(all" not in unit:
            continue
        fc = (wf_fc if group == "walk-forward" else split_fc).where(usable)
        rows.append({"set": group, "model": SARIMAX_NAME,
                     **score(actual[idx], fc[idx], normaliser)})
    before = pd.read_csv(TABLES / ARIMA_SCORES_NAME)
    before = before[before["model"] == SARIMAX_BASE][["set", "model", "MAE_kWh", "nRMSE"]]
    best = baseline_scores_quiet(df)[["set", "model", "MAE_kWh", "nRMSE"]]
    scores = pd.concat([before, pd.DataFrame(rows), best], ignore_index=True)
    wide = scores.pivot_table(index="model", columns="set", values=["MAE_kWh", "nRMSE"],
                              sort=False)
    print("\nSame protocol, before vs after adding hour (nRMSE = RMSE / 0.5374 kWh):")
    print(wide.to_string(float_format="{:.3f}".format))
    change = (scores.set_index(["model", "set"])[["MAE_kWh", "nRMSE"]]
              .unstack("model"))
    for metric in ("MAE_kWh", "nRMSE"):
        b, a = change[(metric, SARIMAX_BASE)], change[(metric, SARIMAX_NAME)]
        scores_change = pd.DataFrame({"before": b, "after": a, "change": a - b,
                                      "change_pct": 100 * (a - b) / b})
        print(f"\nChange in {metric}:")
        print(scores_change.to_string(float_format="{:+.3f}".format))
    scores.to_csv(TABLES / SARIMAX_SCORES_NAME, index=False)
    print(f"Saved {TABLES / SARIMAX_SCORES_NAME}")
    print(f"Saved {TABLES / SARIMAX_PARAMS_NAME}")

    sarima_wf = pd.read_csv(DATA_PROCESSED / ARIMA_FORECASTS_NAME, index_col=0,
                            parse_dates=True)[f"{SARIMAX_BASE} (walk-forward)"]
    plot_arima_walk_forward(df, {SARIMAX_BASE: sarima_wf, SARIMAX_NAME: wf_fc}, folds,
                            filename=SARIMAX_FIGURE,
                            title="Adding hour of day: SARIMA vs SARIMAX, day-ahead forecasts")
    return scores


def error_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Error over the 24 forecast steps: multi-output vs recursive.

    Both fitted on train; every validation day (Apr-May) forecast at 00:00,
    as in the protocol.
    """
    from src import multi_output
    frame = split_frame(df)
    y = model_series(df["Demand"], frame["outage"].reindex(df.index))
    days = pd.DatetimeIndex(frame.index[frame["split"] == "validation"].floor("D").unique())
    actual = df["Demand"].where(~frame["outage"])
    train_end = pd.Timestamp(SPLITS[0][2], tz="UTC")

    inputs, targets = multi_output.design(multi_output.daily_matrix(y))
    observed = multi_output.daily_matrix(actual.reindex(y.index))
    train_days = inputs.index <= train_end
    mimo = multi_output.fit(inputs[train_days], targets[train_days])
    mimo_table = multi_output.horizon_errors(mimo, inputs, observed, days)

    exog = hour_dummies(pd.date_range(y.index.min(), y.index.max() + pd.Timedelta(hours=24),
                                      freq="h"))
    order, seasonal, _ = READ_FROM_PLOTS[SARIMAX_BASE]
    recursive = fit(y.loc[:train_end], order, seasonal, exog)
    rec_table = horizon_profile(recursive, y, actual, days, exog)

    table = pd.concat({"multi-output": mimo_table,
                       "recursive": rec_table.drop(columns="clock_hour")}, axis=1)
    print(f"Error by step, {len(days)} validation days (step 1 = 00:00):")
    print(table.to_string(float_format="{:.3f}".format))
    for name, t in (("multi-output", mimo_table), ("recursive", rec_table)):
        print(f"{name:12s} whole day: MAE {t['MAE_kWh'].mean():.3f}; "
              f"steps 1–2 MAE {t['MAE_kWh'].iloc[:2].mean():.3f}, "
              f"steps 3–24 MAE {t['MAE_kWh'].iloc[2:].mean():.3f}")
    table.to_csv(TABLES / HORIZON_NAME)
    print(f"Saved {TABLES / HORIZON_NAME}")
    plot_error_by_horizon(mimo_table, rec_table)
    return table


def plot_error_by_horizon(mimo: pd.DataFrame, recursive: pd.DataFrame):
    """MAE per step for both strategies; recursive model uncertainty for reference."""
    setup_notebook()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(mimo.index, mimo["MAE_kWh"], color="#1baf7a", linewidth=2.2, marker="o",
            markersize=4, label="multi-output ridge: MAE")
    ax.plot(recursive.index, recursive["MAE_kWh"], color=COLOR_DEMAND, linewidth=1.6,
            marker="o", markersize=4, label="recursive SARIMAX + hour: MAE")
    ax.plot(recursive.index, recursive["model_forecast_se_kWh"], color="#0b0b0b",
            linewidth=1.2, linestyle="--",
            label="recursive model's own forecast standard error")
    ax.set_xticks(mimo.index)
    ax.set_xticklabels([f"{s}\n{h:02d}h" for s, h in zip(mimo.index, mimo["clock_hour"])],
                       fontsize=8)
    ax.set_xlabel("forecast step (top) and clock hour (bottom); forecast made at 00:00")
    ax.set_ylabel("kWh")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Multi-output: no error carried between steps; both follow the time of day",
                 loc="left", fontsize=10)
    fig.suptitle("Error by forecast step: multi-output vs recursive (validation, Apr–May 2014)")
    out = FIGURES / HORIZON_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def compare_training_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-output model: train on all earlier days or only a recent window?

    Rolling day-by-day refit over the validation days (Apr-May), scored with
    MAE and nRMSE (RMSE / 0.5374 kWh), for several windows and ridge strengths.
    """
    from src import multi_output
    frame = split_frame(df)
    y = model_series(df["Demand"], frame["outage"].reindex(df.index))
    actual = multi_output.daily_matrix(df["Demand"].where(~frame["outage"]).reindex(y.index))
    inputs, targets = multi_output.design(multi_output.daily_matrix(y))
    days = inputs.index[frame["split"].reindex(inputs.index) == "validation"]
    normaliser = training_mean(df["Demand"], frame)

    table = multi_output.window_comparison(inputs, targets, actual, days, normaliser)
    print(f"Training window × ridge strength, {len(days)} validation days, "
          "refitted every day:")
    print(table.pivot_table(index="window", columns="alpha", values=["MAE_kWh", "nRMSE"],
                            sort=False).to_string(float_format="{:.3f}".format))
    chosen = table[table["alpha"] == multi_output.ALPHA].set_index("window")
    best = chosen["MAE_kWh"].idxmin()
    print(f"At alpha = {multi_output.ALPHA:g}: lowest MAE with '{best}' "
          f"({chosen.loc[best, 'MAE_kWh']:.3f}); 30 days gives "
          f"{chosen.loc['30 days', 'MAE_kWh']:.3f}")
    table.to_csv(TABLES / WINDOW_NAME, index=False)
    print(f"Saved {TABLES / WINDOW_NAME}")
    plot_training_windows(chosen)
    return table


def plot_training_windows(chosen: pd.DataFrame):
    """MAE and nRMSE per training window, at the model's ridge strength."""
    setup_notebook()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(chosen))
    for ax, metric, label in zip(axes, ("MAE_kWh", "nRMSE"), ("MAE (kWh)", "nRMSE")):
        values = chosen[metric]
        colors = ["#1baf7a" if v == values.min() else "#9a9893" for v in values]
        ax.plot(x, values, color="#d9d8d4", linewidth=1, zorder=1)
        ax.scatter(x, values, color=colors, s=90, zorder=3)
        for xi, v in zip(x, values):
            ax.text(xi, v, f"  {v:.3f}", ha="left", va="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(chosen.index)
        ax.set_xlim(-0.5, len(x) - 0.2)
        pad = (values.max() - values.min()) * 0.6
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_xlabel("training window (days before the forecast day)")
    axes[1].set_xlabel("training window (days before the forecast day)")
    fig.suptitle("Multi-output model: training on all earlier days beats recent windows "
                 "(validation, Apr–May 2014; green = best)")
    fig.tight_layout()
    out = FIGURES / WINDOW_FIGURE
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


def main() -> int:
    set_seeds()
    print("STEP 1 — load")
    df = load_training()
    print(f"Loaded {TRAINING_FILE}: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("STEP 2 — summary")
    summarize(df)
    print(f"Summarised {df.shape[1]} columns over {len(df):,} rows\n")

    print("STEP 3 — data dictionary")
    dictionary = report(df)
    print(f"Built data dictionary: {len(dictionary)} variables\n")

    print("STEP 4 — plot")
    figure_path = plot_pv_demand_price(df)
    print(f"Plotted pv, Demand and Price to {figure_path.name}\n")

    print("STEP 5 — decompose Demand")
    components = decompose_demand(df)
    print(f"Decomposed Demand into Tt, St, Rt over {len(components):,} hours\n")

    print("STEP 6 — demand profiles")
    profiles = demand_profiles(df)
    print(f"Built {profiles['group'].nunique()} hourly demand profiles\n")

    print("STEP 7 — demand, PV and price summaries")
    summary = summarize_demand_pv_price(df)
    print(f"Summarised {len(summary)} variables and plotted their time series\n")

    print("STEP 8 — distributions")
    describe_distributions(df)
    plot_distributions(df)
    print("Plotted distributions of Demand, PV and Price\n")

    print("STEP 9 — demand vs PV, with price")
    plot_demand_vs_pv(df)
    print("Plotted demand and PV on one axis with price below\n")

    print("STEP 10 — monthly PV production vs demand")
    plot_monthly_pv_vs_demand(df)
    print("Plotted monthly PV production against demand\n")

    print("STEP 11 — handle missing values in pv_mod1")
    impute_pv_mod1(df)
    print("Compared deletion, univariate and multivariate imputation\n")

    print("STEP 12 — before/after linear interpolation for representative gaps")
    plot_interpolation_cases(df)
    print(f"Plotted {len(INTERPOLATION_CASES)} representative gaps\n")

    print("STEP 13 — physically impossible imputed values")
    check_impossible_values(df)
    print("Checked imputations against physical limits\n")

    print("STEP 14 — live (causal) imputation of pv_mod1")
    live_imputation(df)
    print("Compared imputations that use no information from after each hour\n")

    print("STEP 15 — demand and weather: statistics and driver ranking")
    describe_demand_weather(df)
    print("Described demand and weather and ranked weather drivers\n")

    print("STEP 16 — distributions: normality and transformations")
    analyse_distributions(df)
    print("Checked normality and applied transformations\n")

    print("STEP 17 — feature engineering: lag, calendar and weather features")
    features = engineer_features(df)
    print("Built lag, calendar and weather features; kept only those known in time\n")

    print("STEP 18 — rank features by relevance")
    rank_modelling_features(features)
    print("Ranked the modelling features\n")

    print("STEP 19 — train / validation / test split")
    define_split(df)
    print("Defined the time-ordered split\n")

    print("STEP 20 — walk-forward validation, last week in daily folds")
    define_walk_forward(df)
    print("Defined the daily walk-forward folds\n")

    print("STEP 21 — predict the seasonal naive nRMSE before scoring (training data only)")
    prediction = predict_baseline_nrmse(df)
    print("Wrote down the predicted nRMSE\n")

    print("STEP 22 — baselines scored with MAE and nRMSE (normalised by training-set mean)")
    scores = baseline_scores(df)
    compare_nrmse_prediction(prediction, scores)
    print("Scored the baselines and compared the prediction with the actual nRMSE\n")

    print("STEP 23 — stationarity: ADF + KPSS, differencing choice")
    test_stationarity(df)
    print("Tested stationarity and chose the differencing\n")

    print("STEP 24 — ACF and PACF of the differenced series, reading (p, q)")
    analyse_acf_pacf(df)
    print("Plotted ACF and PACF\n")

    print("STEP 25 — train ARMA-family models and score them day-ahead")
    train_arima_models(df)
    print("Trained and scored the autoregressive models\n")

    print("STEP 26 — in-sample fit vs walk-forward, models and baselines")
    compare_in_sample_walk_forward(df)
    print("Compared in-sample and walk-forward performance\n")

    print("STEP 27 — exogenous extension: SARIMA + hour of day (SARIMAX)")
    train_sarimax_hour(df)
    print("Scored the SARIMAX extension\n")

    print("STEP 28 — out-of-sample rolling 7-day forecast (266083_forecast.csv)")
    run_forecast_pipeline()
    print("Forecast 1–7 July 2014 day by day\n")

    print("STEP 29 — multi-step strategy: error by forecast horizon (multi-output vs recursive)")
    error_by_horizon(df)
    print("Analysed the error over the 24 forecast steps\n")

    print("STEP 30 — multi-output training window: all days vs recent windows")
    compare_training_windows(df)
    print("Compared training windows on validation\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
