"""Autoregressive (ARMA-family) demand models, forecast day-ahead.

Univariate: every model uses only past demand. Two models:
- SARIMA(0,1,2)(1,0,0)24: read by hand from the ACF/PACF of the
  first-differenced training series (ACF cuts off after lag 2, PACF tails
  off; small non-decaying ACF spikes every 24 h).
- ARIMA(p,1,q) with the lowest AIC over p, q = 0..3: automatic search,
  training set only.

Multi-step strategy: RECURSIVE (iterated). One model of the one-hour-ahead
dynamics is used for all 24 steps; for steps 2..24 the unknown earlier hours
are replaced by the model's own forecasts, so errors carry forward.

Forecasts follow the protocol: for day D, the model state is updated with
demand up to 23:00 on day D-1 (parameters fixed), then 24 steps are forecast
at 00:00. Parameters are fitted once per evaluation set: on train for
validation, on train + validation for test, and on all earlier hours for
each walk-forward fold.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HOURS = 24
SEARCH_MAX_ORDER = 3

#: name → (order, seasonal_order, source)
READ_FROM_PLOTS = {
    "SARIMA(0,1,2)(1,0,0)24": ((0, 1, 2), (1, 0, 0, 24), "ACF/PACF reading"),
}


def model_series(demand: pd.Series, outage: pd.Series) -> pd.Series:
    """Demand on a full hourly grid; outage and missing hours interpolated
    so the state-space filter has an unbroken input."""
    return demand.where(~outage).asfreq("h").interpolate(limit_direction="both")


def fit(y: pd.Series, order, seasonal_order=(0, 0, 0, 0), exog=None):
    """Maximum-likelihood fit, quietly. `exog` is aligned to `y` if given."""
    if exog is not None:
        exog = exog.loc[y.index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=order,
                       seasonal_order=seasonal_order).fit(disp=0)


def hour_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour of day as 23 indicator columns (00:00 is the reference hour).

    The calendar is known in advance, so the value for every forecast hour
    is available at prediction time (lead time unlimited).
    """
    hours = pd.Categorical(index.hour, categories=range(24))
    return (pd.get_dummies(hours, prefix="hour", drop_first=True, dtype=float)
            .set_axis(index))


def aic_search(y: pd.Series, d: int = 1, max_order: int = SEARCH_MAX_ORDER) -> pd.DataFrame:
    """AIC of every ARIMA(p, d, q) with p, q = 0..max_order."""
    rows = []
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            try:
                aic = fit(y, (p, d, q)).aic
            except Exception:  # a non-converging order is simply skipped
                aic = np.nan
            rows.append({"p": p, "d": d, "q": q, "AIC": aic})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def day_ahead(result, y: pd.Series, days: pd.DatetimeIndex,
              exog: pd.DataFrame | None = None) -> pd.Series:
    """24-hour forecasts issued at 00:00 for each day, parameters fixed.

    For each day the fitted parameters are applied to demand up to 23:00
    the day before, and the next 24 hours are forecast. With `exog`, its
    values for the 24 forecast hours are passed in as well.
    """
    pieces = []
    for day in days:
        history = y.loc[:day - pd.Timedelta(hours=1)]
        future = pd.date_range(day, periods=HOURS, freq="h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if exog is None:
                state = result.apply(history)
                pieces.append(state.forecast(HOURS))
            else:
                state = result.apply(history, exog=exog.loc[history.index])
                pieces.append(state.forecast(HOURS, exog=exog.loc[future]))
    return pd.concat(pieces)


def horizon_profile(result, y: pd.Series, actual: pd.Series, days: pd.DatetimeIndex,
                    exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Error and model uncertainty by forecast step (1 = 00:00 ... 24 = 23:00).

    Every forecast starts at 00:00, so step h is also clock hour h-1: the
    observed error mixes horizon and time of day. The model's own forecast
    standard error isolates the horizon effect.
    """
    errors, std_errors = [], []
    for day in days:
        history = y.loc[:day - pd.Timedelta(hours=1)]
        future = pd.date_range(day, periods=HOURS, freq="h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if exog is None:
                fc = result.apply(history).get_forecast(HOURS)
            else:
                fc = (result.apply(history, exog=exog.loc[history.index])
                      .get_forecast(HOURS, exog=exog.loc[future]))
        errors.append(fc.predicted_mean.to_numpy() - actual.reindex(future).to_numpy())
        std_errors.append(fc.se_mean.to_numpy())
    errors, std_errors = np.array(errors), np.array(std_errors)
    return pd.DataFrame({
        "step": range(1, HOURS + 1),
        "clock_hour": range(HOURS),
        "MAE_kWh": np.nanmean(np.abs(errors), axis=0),
        "RMSE_kWh": np.sqrt(np.nanmean(errors ** 2, axis=0)),
        "model_forecast_se_kWh": std_errors.mean(axis=0),
    }).set_index("step")
