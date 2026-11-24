"""Multi-output (MIMO) day-ahead demand model.

Multi-step strategy: MULTI-OUTPUT. One model predicts all 24 hours of day D
at once, as a vector, from information known at 00:00 on day D: the 24
hourly values of day D-1. No forecast is fed back in, and the 24 outputs
are fitted jointly, so each output position is one clock hour (hour of day
is built into the structure).

Model: ridge regression (linear, 24 inputs -> 24 outputs). Inputs and the
ridge strength were chosen on the validation set (Apr-May 2014):
yesterday only, alpha = 100 (MAE 0.221, nRMSE 0.647). Adding the same day
last week did not help (0.222 / 0.650).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HOURS = 24
ALPHA = 100.0
MODEL_NAME = "Multi-output ridge (yesterday's 24 h)"


def daily_matrix(y: pd.Series) -> pd.DataFrame:
    """One row per day, columns 0..23 = demand at each clock hour.

    `y` must be a gap-free hourly series starting at 00:00.
    """
    assert y.index[0].hour == 0 and len(y) % HOURS == 0, "series must cover whole days"
    days = y.index[::HOURS]
    return pd.DataFrame(y.to_numpy().reshape(-1, HOURS), index=days)


def design(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs (previous day's 24 values) and targets (this day's 24 values)."""
    inputs = days.shift(1).add_prefix("yesterday_h")
    return inputs, days


def fit(inputs: pd.DataFrame, targets: pd.DataFrame) -> Ridge:
    """Joint ridge fit of all 24 outputs on complete rows."""
    ok = inputs.notna().all(axis=1) & targets.notna().all(axis=1)
    return Ridge(alpha=ALPHA).fit(inputs[ok], targets[ok])


def forecast_day(model: Ridge, yesterday: pd.Series, day: pd.Timestamp) -> pd.Series:
    """All 24 hours of `day` in one prediction."""
    values = model.predict(yesterday.to_frame().T)[0]
    return pd.Series(values, index=pd.date_range(day, periods=HOURS, freq="h"))


def rolling_forecast(history: pd.Series, actual: pd.Series) -> pd.Series:
    """Day-by-day: refit on every complete day before D, forecast D at 00:00."""
    inputs, targets = design(daily_matrix(pd.concat([history, actual])))
    pieces = []
    for day in pd.DatetimeIndex(actual.index.floor("D").unique()):
        before = inputs.index < day
        model = fit(inputs[before], targets[before])
        pieces.append(forecast_day(model, inputs.loc[day], day))
    return pd.concat(pieces).rename(MODEL_NAME)


def horizon_errors(model: Ridge, inputs: pd.DataFrame, actual: pd.DataFrame,
                   days: pd.DatetimeIndex) -> pd.DataFrame:
    """MAE and RMSE for each of the 24 output positions over `days`."""
    errors = model.predict(inputs.loc[days]) - actual.loc[days].to_numpy()
    return pd.DataFrame({
        "step": range(1, HOURS + 1), "clock_hour": range(HOURS),
        "MAE_kWh": np.nanmean(np.abs(errors), axis=0),
        "RMSE_kWh": np.sqrt(np.nanmean(errors ** 2, axis=0)),
    }).set_index("step")


# --- prediction intervals ------------------------------------------------------
# The ridge model gives point forecasts only. Intervals are built from its
# errors on held-out data (split-conformal / empirical residual quantiles):
# fit on train, collect the validation residuals (actual - forecast) for each
# output position, and add their 10th and 90th percentiles to new forecasts.
# Separate quantiles per position allow the width (and asymmetry) to differ
# by step.

INTERVAL_LEVEL = 0.80


def residual_quantiles(model: Ridge, inputs: pd.DataFrame, actual: pd.DataFrame,
                       days: pd.DatetimeIndex, level: float = INTERVAL_LEVEL) -> pd.DataFrame:
    """Lower and upper residual quantiles for each of the 24 output positions."""
    residuals = actual.loc[days].to_numpy() - model.predict(inputs.loc[days])
    tail = (1 - level) / 2
    return pd.DataFrame({
        "step": range(1, HOURS + 1), "clock_hour": range(HOURS),
        "lower_offset": np.nanquantile(residuals, tail, axis=0),
        "upper_offset": np.nanquantile(residuals, 1 - tail, axis=0),
    }).set_index("step").assign(width=lambda t: t["upper_offset"] - t["lower_offset"])


def interval(point: pd.Series, offsets: pd.DataFrame) -> pd.DataFrame:
    """Prediction interval around a forecast, by clock hour; demand cannot be < 0."""
    by_hour = offsets.set_index("clock_hour").loc[point.index.hour]
    return pd.DataFrame({
        "forecast": point,
        "lower": (point + by_hour["lower_offset"].to_numpy()).clip(lower=0),
        "upper": point + by_hour["upper_offset"].to_numpy(),
    }, index=point.index)


# --- training-window choice --------------------------------------------------
# Should the model learn from all earlier days (expanding window) or only the
# most recent ones (sliding window)? Compared on the validation set with the
# same rolling day-by-day refit as the forecast. Result: all days win
# (MAE 0.221 vs 0.235 for 30 days), so the expanding window is kept.

WINDOWS_DAYS = (14, 30, 60, 90, 180, None)   # None = all earlier days
ALPHAS = (10.0, 100.0, 1000.0)


def rolling_scores(inputs: pd.DataFrame, targets: pd.DataFrame, actual: pd.DataFrame,
                   days: pd.DatetimeIndex, window_days: int | None, alpha: float,
                   normaliser: float) -> dict:
    """MAE and nRMSE of day-by-day forecasts, each refitted on the window
    of complete days before the forecast day."""
    complete = inputs.notna().all(axis=1) & targets.notna().all(axis=1)
    forecasts = []
    for day in days:
        use = complete & (inputs.index < day)
        if window_days is not None:
            use &= inputs.index >= day - pd.Timedelta(days=window_days)
        model = Ridge(alpha=alpha).fit(inputs[use], targets[use])
        forecasts.append(model.predict(inputs.loc[[day]])[0])
    errors = np.array(forecasts) - actual.loc[days].to_numpy()
    return {"MAE_kWh": float(np.nanmean(np.abs(errors))),
            "nRMSE": float(np.sqrt(np.nanmean(errors ** 2)) / normaliser)}


def window_comparison(inputs: pd.DataFrame, targets: pd.DataFrame, actual: pd.DataFrame,
                      days: pd.DatetimeIndex, normaliser: float) -> pd.DataFrame:
    """Every training window × ridge strength, scored on `days`."""
    rows = []
    for window in WINDOWS_DAYS:
        for alpha in ALPHAS:
            rows.append({"window": "all days" if window is None else f"{window} days",
                         "window_days": window, "alpha": alpha,
                         **rolling_scores(inputs, targets, actual, days, window, alpha,
                                          normaliser)})
    return pd.DataFrame(rows)
