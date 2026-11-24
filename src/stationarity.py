"""Stationarity tests and differencing choice for demand (training set only).

Two tests with opposite null hypotheses are used together:
- ADF:  H0 = unit root (non-stationary). p < 0.05 → reject → stationary.
- KPSS: H0 = stationary.                 p < 0.05 → reject → non-stationary.
A series is treated as stationary only when both agree. Over-differencing
is checked with the variance and the autocorrelation at the differenced
lag (close to −0.5 = over-differenced).
"""

import warnings

import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss

ALPHA = 0.05
ACF_LAGS = (1, 24, 168)

#: Candidate transformations: name → (ordinary lag, seasonal lag).
CANDIDATES = {
    "level (no differencing)": (),
    "first difference (lag 1)": (1,),
    "seasonal difference (lag 24)": (24,),
    "seasonal difference (lag 168)": (168,),
    "lag 24 + lag 1": (24, 1),
}


def training_series(demand: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Training-set demand on a full hourly grid.

    Outage hours are set to missing, then the outage and the one missing
    timestamp are filled by linear interpolation, because the tests and
    ARIMA models need an unbroken series.
    """
    y = demand.where(~frame["outage"])[frame["split"] == "train"]
    return y.asfreq("h").interpolate(limit_direction="both")


def difference(y: pd.Series, lags: tuple) -> pd.Series:
    """Apply the differences in `lags` one after another."""
    for k in lags:
        y = y.diff(k)
    return y.dropna()


def run_tests(y: pd.Series) -> dict:
    """ADF and KPSS (level-stationary) statistics, p-values and verdicts."""
    with warnings.catch_warnings():
        # KPSS p-values are interpolated from a table bounded at 0.01-0.10.
        warnings.simplefilter("ignore")
        adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(y, autolag="AIC")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(y, regression="c", nlags="auto")
    adf_ok, kpss_ok = adf_p < ALPHA, kpss_p >= ALPHA
    return {
        "ADF_stat": adf_stat, "ADF_crit_5pct": adf_crit["5%"], "ADF_p": adf_p,
        "ADF_lags": adf_lags,
        "KPSS_stat": kpss_stat, "KPSS_crit_5pct": kpss_crit["5%"], "KPSS_p": kpss_p,
        "KPSS_lags": kpss_lags,
        "ADF_says_stationary": adf_ok, "KPSS_says_stationary": kpss_ok,
        "both_agree_stationary": adf_ok and kpss_ok,
    }


def stationarity_table(y: pd.Series) -> pd.DataFrame:
    """Tests, variance and key autocorrelations for every candidate."""
    rows = []
    for name, lags in CANDIDATES.items():
        s = difference(y, lags)
        r = acf(s, nlags=max(ACF_LAGS))
        rows.append({"series": name, "n": len(s), **run_tests(s),
                     "variance": s.var(),
                     **{f"acf_{k}": r[k] for k in ACF_LAGS}})
    return pd.DataFrame(rows).set_index("series")


def acf_pacf_table(y: pd.Series, nlags: int = 200) -> pd.DataFrame:
    """ACF and PACF by lag with the ±1.96/√n significance bound."""
    from statsmodels.tsa.stattools import pacf

    r_acf = acf(y, nlags=nlags)
    r_pacf = pacf(y, nlags=nlags, method="ywm")
    bound = 1.96 / len(y) ** 0.5
    table = pd.DataFrame({"acf": r_acf[1:], "pacf": r_pacf[1:]},
                         index=pd.RangeIndex(1, nlags + 1, name="lag"))
    table["bound"] = bound
    table["acf_significant"] = table["acf"].abs() > bound
    table["pacf_significant"] = table["pacf"].abs() > bound
    return table
