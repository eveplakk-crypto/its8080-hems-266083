"""Reproducible pipeline: raw files → cleaning → model → rolling 7-day forecast.

    python forecast.py

1. Load   data/raw/266083_training.csv (history) and 266083_forecast.csv
          (the 7 days to forecast, 1–7 Jul 2014).
2. Clean  the same rules for both files: hourly UTC grid, no duplicates,
          outage hours (zero demand on 2014-03-03 14:00–18:00) and missing
          timestamps set to missing, then filled by linear interpolation for
          the model input. Every filled hour is logged.
3. Models two multi-step strategies:
          - RECURSIVE: SARIMAX(0,1,2)(1,0,0)24 with hour of day (23 dummies),
            one one-step model iterated 24 times on its own forecasts;
          - MULTI-OUTPUT: ridge regression predicting all 24 hours of day D
            at once from the 24 hours of day D-1 (src/multi_output.py).
4. Forecast rolling, day by day: for each day D both models are refitted on
          all cleaned demand up to 23:00 on day D-1, then the 24 hours of
          day D are forecast at 00:00 on day D (24 h horizon, 0 h lead time).
          The actual demand of day D is added to the history only after the
          forecast for day D is made.
5. Evaluate MAE and nRMSE for each day and the week, next to the baselines
          run the same way. nRMSE = RMSE / mean demand of the training split
          (Jul 2013 – Mar 2014, outage excluded) = 0.5374 kWh, the same fixed
          normaliser used everywhere in the project.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.arima_models import day_ahead, fit, hour_dummies
from src import multi_output
from src.baselines import all_baselines
from src.config import DATA_PROCESSED, DATA_RAW, FIGURES, TABLES
from src.evaluation import OUTAGE, score, split_frame, training_mean

TRAINING_FILE = "266083_training.csv"
FORECAST_FILE = "266083_forecast.csv"
TARGET = "Demand"
HORIZON_HOURS = 24

MODEL_NAME = "SARIMAX(0,1,2)(1,0,0)24 + hour"
ORDER = (0, 1, 2)
SEASONAL_ORDER = (1, 0, 0, 24)

FORECASTS_NAME = "266083_forecast_week_predictions.csv"
SCORES_NAME = "266083_forecast_week_scores.csv"
CLEANING_LOG_NAME = "266083_forecast_week_cleaning_log.csv"
FIGURE_NAME = "forecast_week_rolling.png"
STEP_SCORES_NAME = "266083_forecast_week_error_by_step.csv"
STEP_FIGURE_NAME = "forecast_week_error_by_step.png"
DAILY_VS_SN_NAME = "266083_forecast_week_best_vs_seasonal_naive.csv"
SEASONAL_NAIVE = "Seasonal naive 24 h"

INTERVALS_NAME = "266083_forecast_week_intervals.csv"
INTERVAL_OFFSETS_NAME = "266083_interval_width_by_step.csv"
INTERVAL_FIGURE_NAME = "forecast_day_prediction_interval.png"
#: The forecast day shown with its prediction interval: the first out-of-sample day.
INTERVAL_DAY = "2014-07-01"

#: The four baselines fixed in Task 7, in the order they were defined.
TASK7_BASELINES = ["Naive (last hour)", "Seasonal naive 24 h",
                   "Seasonal naive 168 h", "Moving average 24 h"]
#: Categorical slots 1, 4, 5 and 7 for the baselines; slot 3 is the forecast.
BASELINE_COLORS = ["#2a78d6", "#eda100", "#9a6fd0", "#4a3aa7"]


# --- 1. load -----------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    """Raw file, indexed by UTC timestamp, columns as in the file."""
    df = pd.read_csv(DATA_RAW / filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


# --- 2. clean ----------------------------------------------------------------

def clean_demand(df: pd.DataFrame, source: str) -> tuple[pd.Series, pd.DataFrame]:
    """Demand on a full hourly grid, with a log of every changed hour.

    Returns the observed series (problems set to NaN, used for scoring) and
    the log. Model input is made from it with `model_input`.
    """
    df = df[~df.index.duplicated(keep="first")]
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h")
    y = df[TARGET].reindex(grid).astype(float)
    log = [{"timestamp": t, "source": source, "issue": "missing timestamp"}
           for t in grid.difference(df.index)]

    start, end = (pd.Timestamp(t, tz="UTC") for t in OUTAGE)
    outage = (y.index >= start) & (y.index <= end)
    log += [{"timestamp": t, "source": source, "issue": "metering outage (0 kWh)"}
            for t in y.index[outage]]
    y[outage] = np.nan

    negative = y < 0
    log += [{"timestamp": t, "source": source, "issue": "negative demand"}
            for t in y.index[negative]]
    y[negative] = np.nan
    return y.rename(TARGET), pd.DataFrame(log, columns=["timestamp", "source", "issue"])


def model_input(y: pd.Series) -> pd.Series:
    """Gap-free series for the state-space model: missing hours interpolated."""
    return y.interpolate(limit_direction="both")


# --- 4. rolling forecast -----------------------------------------------------

def rolling_forecast(history: pd.Series, actual: pd.Series) -> pd.Series:
    """Day-by-day SARIMAX forecasts over the days of `actual`.

    Day D is forecast at 00:00 from everything up to 23:00 on day D-1; the
    model is refitted each day on that history.
    """
    full = pd.concat([history, actual])
    exog = hour_dummies(pd.date_range(full.index.min(),
                                      full.index.max() + pd.Timedelta(hours=HORIZON_HOURS),
                                      freq="h"))
    pieces = []
    for day in pd.DatetimeIndex(actual.index.floor("D").unique()):
        known = full.loc[:day - pd.Timedelta(hours=1)]
        result = fit(known, ORDER, SEASONAL_ORDER, exog)
        pieces.append(day_ahead(result, known, pd.DatetimeIndex([day]), exog))
        print(f"  forecast {day:%a %d %b %Y} from data up to "
              f"{known.index.max():%d %b %H:%M} ({len(known):,} h)")
    return pd.concat(pieces).rename(MODEL_NAME)


# --- 5. evaluate -------------------------------------------------------------

def evaluate(actual: pd.Series, forecasts: pd.DataFrame, normaliser: float) -> pd.DataFrame:
    """MAE and nRMSE per day and for the whole week, for every forecast."""
    rows = []
    days = pd.DatetimeIndex(actual.index.floor("D").unique())
    for name in forecasts.columns:
        for day in days:
            idx = actual.index[actual.index.floor("D") == day]
            rows.append({"forecast": name, "period": f"{day:%a %d %b}",
                         **score(actual[idx], forecasts.loc[idx, name], normaliser)})
        rows.append({"forecast": name, "period": "week (1–7 Jul)",
                     **score(actual, forecasts[name], normaliser)})
    return pd.DataFrame(rows)


def error_by_step(actual: pd.Series, forecasts: pd.DataFrame,
                  normaliser: float) -> pd.DataFrame:
    """MAE and nRMSE for each horizon step h = 1..24 over the forecast days,
    for the multi-output forecast and the four Task 7 baselines only.

    Every forecast is issued at 00:00, so step h covers clock hour h-1.
    Each step is scored on one value per day (7 values for the week).
    """
    forecasts = forecasts[[multi_output.MODEL_NAME, *TASK7_BASELINES]]
    step = pd.Series(actual.index.hour + 1, index=actual.index, name="h")
    errors = forecasts.sub(actual, axis=0)
    mae = errors.abs().groupby(step).mean()
    nrmse = (errors ** 2).groupby(step).mean() ** 0.5 / normaliser
    table = pd.concat({"MAE_kWh": mae, "nRMSE": nrmse}, axis=1)
    table.insert(0, ("actual", "std_kWh"), actual.groupby(step).std())
    table.insert(0, ("actual", "mean_kWh"), actual.groupby(step).mean())
    table.insert(0, ("clock_hour", ""), step.groupby(step).first() - 1)
    return table


def best_vs_seasonal_naive(actual: pd.Series, forecasts: pd.DataFrame, history: pd.Series,
                           normaliser: float) -> pd.DataFrame:
    """For each day: did the best model (multi-output) beat the seasonal naive?

    Also returns the facts used to explain each day: the day's mean and peak
    demand, the same for the day before (what the seasonal naive copies), and
    the model's forecast mean and peak.
    """
    full = pd.concat([history, actual])
    rows = []
    for day, today in actual.groupby(actual.index.floor("D")):
        idx = today.index
        yesterday = full.loc[day - pd.Timedelta(days=1):day - pd.Timedelta(hours=1)]
        model = forecasts.loc[idx, multi_output.MODEL_NAME]
        naive = forecasts.loc[idx, SEASONAL_NAIVE]
        m, n = score(today, model, normaliser), score(today, naive, normaliser)
        rows.append({
            "day": f"{day:%a %d %b}",
            "model_MAE": m["MAE_kWh"], "seasonal_naive_MAE": n["MAE_kWh"],
            "model_nRMSE": m["nRMSE"], "seasonal_naive_nRMSE": n["nRMSE"],
            "beat_on_MAE": m["MAE_kWh"] < n["MAE_kWh"],
            "beat_on_nRMSE": m["nRMSE"] < n["nRMSE"],
            "actual_mean": today.mean(), "actual_peak": today.max(),
            "actual_peak_hour": today.idxmax().hour,
            "yesterday_mean": yesterday.mean(), "yesterday_peak": yesterday.max(),
            "forecast_mean": model.mean(), "forecast_peak": model.max(),
        })
    return pd.DataFrame(rows).set_index("day")


def calibrate_intervals(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Residual quantiles of the multi-output model, from the protocol split:
    fitted on train (Jul-Mar), residuals on validation (Apr-May)."""
    frame = split_frame(train_raw)
    y = model_input(clean_demand(train_raw, TRAINING_FILE)[0])
    observed = clean_demand(train_raw, TRAINING_FILE)[0]
    inputs, targets = multi_output.design(multi_output.daily_matrix(y))
    actual = multi_output.daily_matrix(observed)
    split = frame["split"].reindex(inputs.index)
    model = multi_output.fit(inputs[split == "train"], targets[split == "train"])
    return multi_output.residual_quantiles(model, inputs, actual,
                                           inputs.index[split == "validation"])


def plot_interval_day(band: pd.DataFrame, actual: pd.Series, offsets: pd.DataFrame) -> Path:
    """One forecast day with its 80 % interval (top) and the width per step (bottom)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from src.config import setup_notebook
    setup_notebook()
    inside = (actual >= band["lower"]) & (actual <= band["upper"])
    fig, (ax, ax_w) = plt.subplots(2, 1, figsize=(12, 7.5),
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax.fill_between(band.index, band["lower"], band["upper"], color="#1baf7a", alpha=0.22,
                    linewidth=0, step=None, label="80 % prediction interval")
    ax.plot(band.index, band["forecast"], color="#1baf7a", linewidth=2.4,
            label=f"forecast: {multi_output.MODEL_NAME}")
    ax.plot(actual.index, actual, color="#0b0b0b", linewidth=1.3, marker="o", markersize=4,
            label="actual demand")
    ax.scatter(actual.index[~inside], actual[~inside], color="#d03b3b", s=60, zorder=5,
               label=f"outside the interval ({int((~inside).sum())} of {len(actual)} h)")
    ax.set_ylabel("demand (kWh)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(band.index[0] - pd.Timedelta(minutes=30),
                band.index[-1] + pd.Timedelta(minutes=30))
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2), tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_title(f"{actual.index[0]:%a %d %b %Y}, forecast issued at 00:00 "
                 f"({inside.mean():.0%} of hours inside)", loc="left", fontsize=10)

    ax_w.bar(offsets.index, offsets["width"], color="#1baf7a", width=0.6)
    ax_w.set_xticks(offsets.index)
    ax_w.set_xticklabels([f"{s}\n{h:02d}h" for s, h in
                          zip(offsets.index, offsets["clock_hour"])], fontsize=8)
    ax_w.set_ylabel("interval width\n(kWh)")
    ax_w.set_xlabel("horizon step h (top) and clock hour (bottom)")
    fig.suptitle("Multi-output forecast with 80 % prediction interval "
                 "(calibrated on validation residuals, Apr–May 2014)")
    fig.tight_layout()
    out = FIGURES / INTERVAL_FIGURE_NAME
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_error_by_step(table: pd.DataFrame) -> Path:
    """MAE per step for both models and the baselines, with demand variability."""
    import matplotlib.pyplot as plt

    from src.config import setup_notebook
    setup_notebook()
    mae = table["MAE_kWh"]
    fig, (ax, ax_sd) = plt.subplots(2, 1, sharex=True, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [2.4, 1]})
    for name, color in zip(TASK7_BASELINES, BASELINE_COLORS):
        ax.plot(mae.index, mae[name], color=color, linewidth=1.2, linestyle="--",
                marker="o", markersize=3, label=f"baseline: {name}")
    ax.plot(mae.index, mae[multi_output.MODEL_NAME], color="#1baf7a", linewidth=2.6,
            marker="o", markersize=5, label=f"forecast: {multi_output.MODEL_NAME}")
    ax.set_ylabel("MAE (kWh)")
    ax.set_ylim(0, mae.max().max() * 1.4)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax.set_title("Multi-output forecast vs the four Task 7 baselines; the curve follows "
                 "the time of day, not the distance ahead", loc="left", fontsize=10)
    ax_sd.bar(table.index, table[("actual", "std_kWh")], color="#0b0b0b", width=0.6)
    ax_sd.set_ylabel("actual demand\nstd across days\n(kWh)")
    ax_sd.set_xticks(table.index)
    ax_sd.set_xticklabels([f"{h}\n{c:02d}h" for h, c in
                           zip(table.index, table[("clock_hour", "")])], fontsize=8)
    ax_sd.set_xlabel("horizon step h (top) and clock hour (bottom); forecast issued at 00:00")
    fig.suptitle("Error per horizon step h, out-of-sample week 1–7 July 2014 (7 forecasts per step)")
    fig.tight_layout()
    out = FIGURES / STEP_FIGURE_NAME
    fig.savefig(out)
    plt.close(fig)
    return out


def plot(actual: pd.Series, forecasts: pd.DataFrame, scores: pd.DataFrame) -> Path:
    """Actual demand and the model forecast over the week, best baseline dashed."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from src.config import setup_notebook
    setup_notebook()
    week = scores[scores["period"].str.startswith("week")].set_index("forecast")
    models = [multi_output.MODEL_NAME, MODEL_NAME]
    best_baseline = week.drop(index=models)["MAE_kWh"].idxmin()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(actual.index, actual, color="#0b0b0b", linewidth=1.3, label="actual demand")
    for name, color in zip(models, ("#1baf7a", "#eb6834")):
        ax.plot(forecasts.index, forecasts[name], color=color, linewidth=2,
                label=f"{name} (MAE {week.loc[name, 'MAE_kWh']:.3f}, "
                      f"nRMSE {week.loc[name, 'nRMSE']:.3f})")
    ax.plot(forecasts.index, forecasts[best_baseline], color="#2a78d6", linewidth=1.2,
            linestyle="--", label=f"best baseline: {best_baseline} "
                                  f"(MAE {week.loc[best_baseline, 'MAE_kWh']:.3f})")
    for day in pd.DatetimeIndex(actual.index.floor("D").unique()):
        ax.axvline(day, color="#d9d8d4", linewidth=0.8, zorder=0)
    ax.set_ylabel("demand (kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz="UTC"))
    ax.set_xlabel("each day forecast at 00:00 from data up to 23:00 the day before (UTC)")
    fig.suptitle("Out-of-sample rolling forecast, 1–7 July 2014 (24 h horizon, 0 h lead time)")
    out = FIGURES / FIGURE_NAME
    fig.savefig(out)
    plt.close(fig)
    return out


# --- run ---------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Run every stage and save forecasts, scores, cleaning log and figure."""
    print("1. Load")
    train_raw, week_raw = load(TRAINING_FILE), load(FORECAST_FILE)
    print(f"  {TRAINING_FILE}: {len(train_raw):,} rows, "
          f"{train_raw.index.min():%Y-%m-%d} – {train_raw.index.max():%Y-%m-%d}")
    print(f"  {FORECAST_FILE}: {len(week_raw):,} rows, "
          f"{week_raw.index.min():%Y-%m-%d} – {week_raw.index.max():%Y-%m-%d}")

    print("2. Clean")
    y_train, log_train = clean_demand(train_raw, TRAINING_FILE)
    y_week, log_week = clean_demand(week_raw, FORECAST_FILE)
    gap = y_week.index.min() - y_train.index.max()
    assert gap == pd.Timedelta(hours=1), f"files do not join up (gap {gap})"
    log = pd.concat([log_train, log_week], ignore_index=True)
    print(log.groupby(["source", "issue"]).size().to_string() if len(log) else "  no issues")
    normaliser = training_mean(train_raw[TARGET], split_frame(train_raw))
    print(f"  normaliser (training-split mean demand) = {normaliser:.4f} kWh")

    print(f"3–4. Recursive: {MODEL_NAME}, rolling day-by-day forecast")
    model = rolling_forecast(model_input(y_train), model_input(y_week))
    print(f"3–4. Multi-output: {multi_output.MODEL_NAME}, rolling day-by-day forecast")
    mimo = multi_output.rolling_forecast(model_input(y_train), model_input(y_week))

    combined = pd.concat([model_input(y_train), model_input(y_week)])
    baselines = all_baselines(combined).loc[y_week.index]
    forecasts = pd.concat([mimo, model, baselines], axis=1)
    forecasts.insert(0, "actual", y_week)

    print("5. Evaluate")
    scores = evaluate(y_week, forecasts.drop(columns="actual"), normaliser)
    print(scores.pivot_table(index="period", columns="forecast", values="MAE_kWh", sort=False)
          .to_string(float_format="{:.3f}".format))
    week = scores[scores["period"].str.startswith("week")]
    print("\nWhole week:")
    print(week.set_index("forecast")[["hours", "MAE_kWh", "nRMSE"]]
          .to_string(float_format="{:.3f}".format))

    steps = error_by_step(y_week, forecasts.drop(columns="actual"), normaliser)
    print("\nMAE (kWh) per horizon step h (forecast issued at 00:00):")
    print(steps["MAE_kWh"].to_string(float_format="{:.3f}".format))

    daily = best_vs_seasonal_naive(y_week, forecasts.drop(columns="actual"),
                                   model_input(y_train), normaliser)
    print(f"\nBest model ({multi_output.MODEL_NAME}) vs {SEASONAL_NAIVE}, per day:")
    print(daily.to_string(float_format="{:.3f}".format))

    offsets = calibrate_intervals(train_raw)
    bands = multi_output.interval(forecasts[multi_output.MODEL_NAME], offsets)
    bands.insert(0, "actual", y_week)
    inside = (bands["actual"] >= bands["lower"]) & (bands["actual"] <= bands["upper"])
    print(f"\n80 % prediction interval width by step (kWh): "
          f"min {offsets['width'].min():.2f} (h={offsets['width'].idxmin()}), "
          f"max {offsets['width'].max():.2f} (h={offsets['width'].idxmax()})")
    print("Coverage per day: " + ", ".join(
        f"{d:%d %b} {c:.0%}" for d, c in inside.groupby(inside.index.floor("D")).mean().items())
        + f"; week {inside.mean():.1%}")
    day = bands.loc[INTERVAL_DAY]

    TABLES.mkdir(parents=True, exist_ok=True)
    offsets.to_csv(TABLES / INTERVAL_OFFSETS_NAME)
    bands.assign(inside=inside).to_csv(DATA_PROCESSED / INTERVALS_NAME)
    interval_figure = plot_interval_day(day, day["actual"], offsets)
    for path in (TABLES / INTERVAL_OFFSETS_NAME, DATA_PROCESSED / INTERVALS_NAME,
                 interval_figure):
        print(f"Saved {path}")
    daily.to_csv(TABLES / DAILY_VS_SN_NAME)
    print(f"Saved {TABLES / DAILY_VS_SN_NAME}")
    steps.to_csv(TABLES / STEP_SCORES_NAME)
    step_figure = plot_error_by_step(steps)
    print(f"Saved {TABLES / STEP_SCORES_NAME}")
    print(f"Saved {step_figure}")
    forecasts.to_csv(DATA_PROCESSED / FORECASTS_NAME)
    scores.to_csv(TABLES / SCORES_NAME, index=False)
    log.to_csv(TABLES / CLEANING_LOG_NAME, index=False)
    figure = plot(y_week, forecasts, scores)
    for path in (DATA_PROCESSED / FORECASTS_NAME, TABLES / SCORES_NAME,
                 TABLES / CLEANING_LOG_NAME, figure):
        print(f"Saved {path}")
    return scores
