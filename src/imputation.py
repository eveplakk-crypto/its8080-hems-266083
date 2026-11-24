"""Missing-value handling for pv_mod1: deletion, univariate and multivariate.

pv_mod1 is equal to the complete `pv` column wherever it is present, so `pv`
holds the true value behind every gap. It is used only to score the methods
and is never an input to any of them. The other modules (pv_mod2, pv_mod3)
are left out as inputs for the same reason: they are copies of the target.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

TARGET = "pv_mod1"
TRUTH = "pv"

#: Observed weather columns used by the multivariate methods.
#: Shortwave_radiation is left out: it is missing for 81 % of hours.
WEATHER = [
    "direct_radiation (W/m²)", "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)", "Temperature", "Cloud_cover (%)",
    "Cloud_cover_low (%)", "Cloud_cover_mid (%)", "Cloud_cover_high (%)",
]

RANDOM_STATE = 0


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour of day and day of year as sine/cosine pairs, so 23 h sits next to 0 h."""
    hour = 2 * np.pi * index.hour / 24
    doy = 2 * np.pi * index.dayofyear / 365.25
    return pd.DataFrame({"hour_sin": np.sin(hour), "hour_cos": np.cos(hour),
                         "doy_sin": np.sin(doy), "doy_cos": np.cos(doy)},
                        index=index)


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Target plus weather and time features, in that column order."""
    return pd.concat([df[[TARGET] + WEATHER], time_features(df.index)], axis=1)


# --- deletion ---------------------------------------------------------------

def listwise_deletion(df: pd.DataFrame) -> pd.Series:
    """Drop every hour where pv_mod1 is missing."""
    return df[TARGET].dropna()


# --- univariate -------------------------------------------------------------

def mean_imputation(df: pd.DataFrame) -> pd.Series:
    """Fill every gap with the overall mean of the observed values."""
    s = df[TARGET]
    return s.fillna(s.mean())


def linear_interpolation(df: pd.DataFrame) -> pd.Series:
    """Straight line between the neighbouring observed hours, using real time gaps."""
    return df[TARGET].interpolate(method="time", limit_direction="both")


# --- multivariate -----------------------------------------------------------

def knn_imputation(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """Mean of the k hours with the most similar weather and time of day/year.

    Features are standardised first so radiation (hundreds of W/m²) does not
    swamp the time features; the target is put back on its own scale after.
    """
    X = design_matrix(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    filled = KNNImputer(n_neighbors=k, weights="distance").fit_transform(scaled)
    target = filled[:, 0] * scaler.scale_[0] + scaler.mean_[0]
    return pd.Series(target, index=df.index, name=TARGET).clip(lower=0)


def mice_imputation(df: pd.DataFrame) -> pd.Series:
    """Iterative (MICE-style) imputation with a tree model for each column.

    Trees are used instead of the default linear model because PV is
    non-linear in radiation and is zero for every night hour.
    """
    X = design_matrix(df)
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=-1, random_state=RANDOM_STATE),
        max_iter=10, random_state=RANDOM_STATE)
    filled = imputer.fit_transform(X)
    return pd.Series(filled[:, 0], index=df.index, name=TARGET).clip(lower=0)


METHODS = {
    "Listwise deletion": ("Deletion", listwise_deletion),
    "Mean imputation": ("Univariate", mean_imputation),
    "Linear interpolation": ("Univariate", linear_interpolation),
    "KNN imputation (k=5)": ("Multivariate", knn_imputation),
    "MICE (Extra Trees)": ("Multivariate", mice_imputation),
}


def evaluate(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Score each method on the missing hours and on the whole series."""
    missing = df[TARGET].isna()
    truth = df[TRUTH]
    rows = []
    for name, series in results.items():
        family = METHODS[name][0]
        row = {"family": family, "method": name, "n": int(series.size),
               "mean": series.mean(), "std": series.std(),
               "total": series.sum()}
        if family == "Deletion":
            row.update(MAE=np.nan, RMSE=np.nan, bias=np.nan, R2=np.nan)
        else:
            err = series[missing] - truth[missing]
            row.update(
                MAE=err.abs().mean(),
                RMSE=np.sqrt((err ** 2).mean()),
                bias=err.mean(),
                R2=1 - (err ** 2).sum()
                / ((truth[missing] - truth[missing].mean()) ** 2).sum())
        rows.append(row)
    rows.append({"family": "Reference", "method": "True values (pv)",
                 "n": int(truth.size), "mean": truth.mean(),
                 "std": truth.std(), "total": truth.sum()})
    return pd.DataFrame(rows).set_index("method")


def sun_down(df: pd.DataFrame) -> pd.Series:
    """Hours when the sun is down: radiation is 0 and, in that month, no day
    ever recorded PV at that hour of day.

    Both conditions are taken from the data, so no site location is assumed.
    This is a stand-in for solar position: a live system would compute sun
    elevation from the site coordinates, which is known in advance.
    """
    radiation = df["direct_radiation (W/m²)"] + df["diffuse_radiation (W/m²)"]
    never_produces = (df[TRUTH].groupby([df.index.month, df.index.hour])
                      .transform("max") == 0)
    return never_produces & (radiation == 0)


def impute_all(df: pd.DataFrame) -> dict:
    """Run every method on pv_mod1 and return the resulting series."""
    return {name: func(df) for name, (_, func) in METHODS.items()}


def distribution_summary(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Numerical summary of each imputed data set next to the true PV series.

    Each row describes a whole series (all hours it contains), so the rows
    can be compared directly with the `pv` reference row.
    """
    series = {"True values (pv)": df[TRUTH], **results}
    rows = []
    for name, s in series.items():
        s = s.dropna()
        q = s.quantile([0.25, 0.50, 0.75])
        rows.append({
            "method": name, "n": int(s.size), "mean": s.mean(),
            "variance": s.var(), "std": s.std(), "min": s.min(),
            "p25": q.loc[0.25], "median": q.loc[0.50], "p75": q.loc[0.75],
            "max": s.max(), "skew": s.skew(), "kurtosis": s.kurt(),
            "n_zeros": int((s == 0).sum()), "total": s.sum(),
        })
    return pd.DataFrame(rows).set_index("method")


# --- live (causal) methods --------------------------------------------------
# Each value at hour t is filled using only pv_mod1 up to t and the weather
# at t, as a system cleaning data on arrival would have to.

def live_forward_fill(df: pd.DataFrame) -> pd.Series:
    """Repeat the last observed value."""
    return df[TARGET].ffill()


def live_same_hour(df: pd.DataFrame) -> pd.Series:
    """Value at the same hour on the last day it was observed."""
    s = df[TARGET]
    return s.fillna(s.groupby(df.index.hour).ffill())


def live_hour_mean(df: pd.DataFrame) -> pd.Series:
    """Mean of all past observed values at this hour of day."""
    s = df[TARGET]
    past = s.groupby(df.index.hour).transform(
        lambda x: x.expanding().mean().shift(1))
    return s.fillna(past)


def live_weather_model(df: pd.DataFrame, min_train_hours: int = 24 * 7) -> pd.Series:
    """Extra Trees on weather and time features, retrained at the start of
    each month on all earlier hours only.

    Until enough history exists (the first month), gaps fall back to the
    same hour on the last observed day.
    """
    s = df[TARGET]
    X = pd.concat([df[WEATHER], time_features(df.index)], axis=1)
    month = df.index.tz_convert(None).to_period("M")
    predicted = pd.Series(np.nan, index=df.index)
    for period in month.unique():
        current = month == period
        train = (df.index < df.index[current][0]) & s.notna().to_numpy()
        gaps = current & s.isna().to_numpy()
        if train.sum() < min_train_hours or not gaps.any():
            continue
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                    n_jobs=-1, random_state=RANDOM_STATE)
        model.fit(X[train], s[train])
        predicted[gaps] = model.predict(X[gaps])
    return s.fillna(predicted).fillna(live_same_hour(df))


def with_sun_down_rule(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    """Set filled values to 0 wherever the sun is down."""
    filled = series.copy()
    filled[df[TARGET].isna() & sun_down(df)] = 0.0
    return filled


LIVE_METHODS = {
    "Forward fill": ("Univariate", live_forward_fill),
    "Same hour, last observed day": ("Univariate", live_same_hour),
    "Past mean at this hour": ("Univariate", live_hour_mean),
    "Weather model, past data only": ("Multivariate", live_weather_model),
}


def impute_live(df: pd.DataFrame) -> dict:
    """Run every live method on pv_mod1, each followed by the sun-down rule."""
    return {name: with_sun_down_rule(df, func(df))
            for name, (_, func) in LIVE_METHODS.items()}


def evaluate_live(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Error on the missing hours for each live method, with offline linear
    interpolation as the (non-live) reference."""
    missing = df[TARGET].isna()
    truth = df[TRUTH][missing]
    series = {name: (LIVE_METHODS[name][0], s) for name, s in results.items()}
    series["Linear interpolation (offline reference)"] = (
        "Reference", linear_interpolation(df))
    rows = []
    for name, (family, s) in series.items():
        err = s[missing] - truth
        rows.append({"family": family, "method": name,
                     "n_filled": int(s[missing].notna().sum()),
                     "MAE": err.abs().mean(),
                     "RMSE": np.sqrt((err ** 2).mean()),
                     "bias": err.mean(), "total": s.sum()})
    return pd.DataFrame(rows).set_index("method")
