"""Feature engineering for day-ahead demand forecasting.

The battery schedule for a day is planned the day before, so the forecast
horizon is 24 h: at forecast time, demand is known only up to 24 hours
before the hour being predicted. Every lag is therefore at least 24 h.

One function builds the features, so training, the forecast week and the
optimization day are all treated the same way.

Prediction time: the forecast for day D is issued at 00:00 UTC on day D,
once the last hour of day D-1 (23:00-24:00) has been metered. Each
timestamp labels the hour that starts at it, so the value at t is known
at t + 1 h.
"""

import pandas as pd

DEMAND = "Demand"
HORIZON_HOURS = 24
LAG_HOURS = 24

#: Heating base temperature (°C). Below this, heating is assumed to be on.
#: The daily correlation with demand is flat between 10 and 18 °C
#: (Spearman ρ 0.33–0.34), so the common 15 °C convention is kept.
HEATING_BASE_C = 15.0
TEMPERATURE_WINDOW_HOURS = 24


def lag(series: pd.Series, hours: int) -> pd.Series:
    """Value `hours` earlier, by timestamp rather than by row.

    Shifting by timestamp keeps the lag exact across the missing hour
    2013-12-31 23:00, where a row-based shift would be off by one hour.
    """
    shifted = series.shift(hours, freq="h")
    return shifted.reindex(series.index)


def autocorrelation(series: pd.Series, lags) -> pd.Series:
    """Spearman correlation of the series with itself at each lag (hours).

    Spearman is used because demand is right-skewed with spikes, which
    pull the ordinary (Pearson) autocorrelation down.
    """
    return pd.Series({k: series.corr(lag(series, k), method="spearman")
                      for k in lags}, name="spearman_acf")


def heating_degree_hours(temperature: pd.Series,
                         base: float = HEATING_BASE_C,
                         window: int = TEMPERATURE_WINDOW_HOURS) -> pd.Series:
    """How far the last 24 h mean temperature is below the heating base.

    The 24 h mean stands for the building's slow response to outdoor
    temperature; it uses the current and past hours only.
    """
    mean_24h = temperature.rolling(f"{window}h", min_periods=window // 2).mean()
    return (base - mean_24h).clip(lower=0).rename("heating_degree_hours")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Demand target plus one lag, one calendar and one weather feature.

    - demand_lag_24h: demand at the same hour the day before.
    - hour: hour of day (0-23), the main daily pattern in demand.
    - heating_degree_hours: how far the last 24 h mean temperature is
      below the heating base.
    """
    y = df[DEMAND]
    return pd.DataFrame({
        DEMAND: y,
        f"demand_lag_{LAG_HOURS}h": lag(y, LAG_HOURS),
        "hour": df.index.hour,
        "heating_degree_hours": heating_degree_hours(df["Temperature"]),
    }, index=df.index)


#: When each candidate feature's value for target hour t becomes known.
#: lead_time_h = how many hours before the target hour starts the value is
#: known (0 h = not known in advance). A feature is available only if it is
#: known by the prediction time (00:00 on day D) for every target hour of
#: day D; the binding case is the last hour, 23:00.
AVAILABILITY = pd.DataFrame([
    {"feature": "demand_lag_24h",
     "built_from": "Demand at t − 24 h",
     "known_at": "t − 23 h (end of the same hour yesterday)",
     "latest_known_for_day_D": "00:00 on day D (for target 23:00)",
     "lead_time_h": 23.0, "available": True},
    {"feature": "hour",
     "built_from": "the timestamp t itself",
     "known_at": "always (calendar, fixed in advance)",
     "latest_known_for_day_D": "any time before day D",
     "lead_time_h": float("inf"), "available": True},
    {"feature": "heating_degree_hours",
     "built_from": "mean Temperature over t − 23 h … t",
     "known_at": "t + 1 h (needs the measured temperature of the target hour)",
     "latest_known_for_day_D": "24:00 on day D, after the day has passed",
     "lead_time_h": 0.0, "available": False},
    {"feature": "heating_degree_hours_lag_24h",
     "built_from": "mean Temperature over t − 47 h … t − 24 h",
     "known_at": "t − 23 h",
     "latest_known_for_day_D": "00:00 on day D (for target 23:00)",
     "lead_time_h": 23.0, "available": True},
]).set_index("feature")


def candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature built so far, before the availability check."""
    features = build_features(df)
    features["heating_degree_hours_lag_24h"] = lag(
        features["heating_degree_hours"], LAG_HOURS)
    return features


def modelling_set(features: pd.DataFrame) -> pd.DataFrame:
    """Keep only the target and the features known at prediction time."""
    keep = AVAILABILITY.index[AVAILABILITY["available"]]
    return features[[DEMAND, *[c for c in features.columns if c in keep]]]


def feature_relevance(features: pd.DataFrame) -> pd.Series:
    """Spearman correlation of each numeric feature with demand."""
    cols = [c for c in features.columns if c != DEMAND]
    return (features[cols].corrwith(features[DEMAND], method="spearman")
            .rename("spearman_rho"))


#: Train on Jul 2013 – Mar 2014, score permutation importance on Apr – Jun
#: 2014, so the importance is measured on hours the model has not seen.
RANKING_SPLIT = "2014-04-01"
RANDOM_STATE = 0


def rank_features(features: pd.DataFrame) -> pd.DataFrame:
    """Rank the modelling features with three simple, complementary methods.

    - Spearman ρ: monotonic relation with demand (misses U- or peak-shaped
      ones, such as hour of day).
    - Mutual information: any relation, linear or not.
    - Permutation importance: how much a random forest's error (MAE) on
      unseen months grows when the feature is shuffled; this also accounts
      for overlap between features.
    The final rank is the mean of the three ranks.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.inspection import permutation_importance

    data = features.dropna()
    X, y = data.drop(columns=DEMAND), data[DEMAND]
    discrete = [c == "hour" for c in X.columns]

    spearman = X.corrwith(y, method="spearman").abs()
    mi = pd.Series(mutual_info_regression(X, y, discrete_features=discrete,
                                          random_state=RANDOM_STATE), index=X.columns)
    train = X.index < pd.Timestamp(RANKING_SPLIT, tz="UTC")
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                   n_jobs=-1, random_state=RANDOM_STATE)
    forest.fit(X[train], y[train])
    perm = permutation_importance(forest, X[~train], y[~train],
                                  scoring="neg_mean_absolute_error",
                                  n_repeats=20, random_state=RANDOM_STATE)
    permutation = pd.Series(perm.importances_mean, index=X.columns)

    table = pd.DataFrame({"spearman_abs_rho": spearman,
                          "mutual_information": mi,
                          "permutation_mae_increase": permutation})
    ranks = table.rank(ascending=False)
    table["mean_rank"] = ranks.mean(axis=1)
    table["final_rank"] = table["mean_rank"].rank(method="min").astype(int)
    return table.sort_values("final_rank")
