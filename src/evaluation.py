"""Train / validation / test split of the training file, in time order.

Split by whole calendar months, roughly 75 / 17 / 8 %:
- Train (9 months) covers the whole winter, the highest-demand period.
- Validation (2 months, ~8-9 weeks) compares models and tunes settings.
- Test (1 month, the most recent data) gives one final score, used once.

Validation and test are scored in consecutive 7-day blocks.
"""

import pandas as pd

DEMAND = "Demand"

#: (name, first hour, last hour) of each set, inclusive, in UTC.
SPLITS = [
    ("train", "2013-07-01 00:00", "2014-03-31 23:00"),
    ("validation", "2014-04-01 00:00", "2014-05-31 23:00"),
    ("test", "2014-06-01 00:00", "2014-06-30 23:00"),
]

#: Hours left out of fitting and scoring: zero demand, likely a metering outage.
OUTAGE = ("2014-03-03 14:00", "2014-03-03 18:00")

BLOCK_DAYS = 7
#: A day counts as "absent" when its mean demand is below this (kWh per hour).
ABSENCE_THRESHOLD = 0.3


def assign_split(index: pd.DatetimeIndex) -> pd.Series:
    """Label every hour with the set it belongs to."""
    labels = pd.Series(pd.NA, index=index, dtype="object", name="split")
    for name, start, end in SPLITS:
        labels[(index >= pd.Timestamp(start, tz="UTC"))
               & (index <= pd.Timestamp(end, tz="UTC"))] = name
    return labels


def outage_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True for the hours excluded as a metering outage."""
    start, end = (pd.Timestamp(t, tz="UTC") for t in OUTAGE)
    return pd.Series((index >= start) & (index <= end), index=index, name="outage")


def week_blocks(index: pd.DatetimeIndex, labels: pd.Series) -> pd.Series:
    """Number consecutive 7-day blocks from the first day of each set.

    Training hours get no block; the last block of a set may be shorter.
    """
    blocks = pd.Series(pd.NA, index=index, dtype="Int64", name="week_block")
    for name, start, _ in SPLITS[1:]:
        in_set = labels == name
        days = (index[in_set] - pd.Timestamp(start, tz="UTC")).days
        blocks[in_set] = days // BLOCK_DAYS + 1
    return blocks


def absent_days(demand: pd.Series) -> pd.Series:
    """True for every hour of a day whose mean demand is below the threshold."""
    daily = demand.resample("D").mean() < ABSENCE_THRESHOLD
    return daily.reindex(demand.index.floor("D")).set_axis(demand.index).rename("absent_day")


def split_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour split label, week block, outage and absence flags."""
    labels = assign_split(df.index)
    return pd.DataFrame({
        "split": labels,
        "week_block": week_blocks(df.index, labels),
        "outage": outage_mask(df.index),
        "absent_day": absent_days(df[DEMAND]),
    }, index=df.index)


def split_summary(df: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per set: dates, size and what it contains."""
    rows = []
    for name, start, end in SPLITS:
        m = frame["split"] == name
        rows.append({
            "set": name, "first_hour": start, "last_hour": end,
            "hours": int(m.sum()), "share_pct": 100 * m.mean(),
            "days": int(m.sum() // 24),
            "full_week_blocks": int(frame.loc[m, "week_block"]
                                    .value_counts().eq(BLOCK_DAYS * 24).sum())
                                if name != "train" else 0,
            "outage_hours": int(frame.loc[m, "outage"].sum()),
            "absent_days": int(frame.loc[m, "absent_day"].sum() // 24),
            "mean_temperature": df.loc[m, "Temperature"].mean(),
            "mean_demand": df.loc[m, DEMAND].mean(),
        })
    return pd.DataFrame(rows).set_index("set")


# --- walk-forward validation ------------------------------------------------
#: The last 7 days of the file, one fold per day.
WALK_FORWARD_DAYS = 7
FOLD_HOURS = 24


def walk_forward_folds(index: pd.DatetimeIndex,
                       n_days: int = WALK_FORWARD_DAYS) -> pd.DataFrame:
    """Daily walk-forward folds over the last `n_days` of the data.

    Fold k forecasts day D_k (00:00-23:00). Its training window is every
    hour from the start of the file up to 23:00 on the day before, so it
    grows by one day per fold (expanding window). The forecast is issued at
    00:00 on day D_k, once that last training hour has been metered.
    """
    last_day = index.max().floor("D")
    first_test_day = last_day - pd.Timedelta(days=n_days - 1)
    rows = []
    for k in range(n_days):
        test_start = first_test_day + pd.Timedelta(days=k)
        test_end = test_start + pd.Timedelta(hours=FOLD_HOURS - 1)
        train_end = test_start - pd.Timedelta(hours=1)
        rows.append({
            "fold": k + 1,
            "train_start": index.min(), "train_end": train_end,
            "train_hours": int(((index >= index.min()) & (index <= train_end)).sum()),
            "forecast_issued": test_start,
            "test_start": test_start, "test_end": test_end,
            "test_hours": int(((index >= test_start) & (index <= test_end)).sum()),
        })
    return pd.DataFrame(rows).set_index("fold")


# --- metrics ----------------------------------------------------------------
# nRMSE = RMSE / mean of the target over the TRAINING set (outage excluded).
# One fixed denominator is used for every fold, block and model, so scores
# stay comparable and no information from the scored period leaks into it.


def training_mean(series: pd.Series, frame: pd.DataFrame) -> float:
    """Mean of the target over the training set, outage hours excluded."""
    return float(series[(frame["split"] == "train") & ~frame["outage"]].mean())


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error, in the target's unit (kWh)."""
    return float((forecast - actual).abs().mean())


def nrmse(actual: pd.Series, forecast: pd.Series, normaliser: float) -> float:
    """Root mean squared error divided by the fixed training-set mean."""
    rmse = float((((forecast - actual) ** 2).mean()) ** 0.5)
    return rmse / normaliser


def score(actual: pd.Series, forecast: pd.Series, normaliser: float) -> dict:
    """MAE and nRMSE on the hours where both values exist."""
    both = pd.concat([actual, forecast], axis=1).dropna()
    a, f = both.iloc[:, 0], both.iloc[:, 1]
    return {"hours": len(both), "MAE_kWh": mae(a, f),
            "nRMSE": nrmse(a, f, normaliser)}


# --- predicted nRMSE of the seasonal naive baseline -------------------------
# Written down before the baseline is scored, from training data only:
#   nRMSE ≈ (σ / mean) · √(2 · (1 − ρ_k))
# where ρ_k is the Pearson autocorrelation at the seasonal lag k. It follows
# from Var(y_t − y_{t−k}) = 2σ²(1 − ρ_k) for a series whose mean and spread
# do not change over time.


def predict_seasonal_naive_nrmse(series: pd.Series, frame: pd.DataFrame,
                                 lag_hours: int = 24) -> dict:
    """Predicted nRMSE of 'same hour k hours earlier', training set only."""
    from src.features import lag

    usable = ~frame["outage"]
    y = series.where(usable)
    train = frame["split"] == "train"
    y_train = y[train]
    lagged = lag(y, lag_hours)[train]
    mean, sigma = float(y_train.mean()), float(y_train.std())
    rho = float(y_train.corr(lagged))
    factor = (2 * (1 - rho)) ** 0.5
    return {"lag_hours": lag_hours, "train_mean": mean, "train_std": sigma,
            "cv": sigma / mean, "pearson_rho": rho, "sqrt_2_1_minus_rho": factor,
            "predicted_nRMSE": sigma / mean * factor}
