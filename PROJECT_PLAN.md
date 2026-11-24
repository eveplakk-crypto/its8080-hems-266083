# Project Plan — HEMS data science workflow, dataset 266083

ITS8080 Energy Data Science. The plan follows the Data Science Lifecycle and is
written against the actual files in this project and the analysis done so far.

## Timeline

```text
WEEK       4    5    6    7    8    9    10   11   12   13   14
           │    │    │    │    │    │    │    │    │    │    │

Business   ███
understanding

Exploration     ███

Cleaning             ███────██

Feature                   ███────██
engineering

Modelling                           ███  ███  ███  ███
                                   └─────────────────┘
                                     MOST EFFORT

Evaluation                         ███  ███  ███  ███  ███

Visualisation   ███────────────────────────────────────███
```

**Sources this plan is based on**

- Project context (HEMS brief, as provided): reduce the annual electricity bill
  and maximise self-consumption of solar PV, using PV, a battery (BESS) and an
  optimising supervisor.
- `data/raw/266083_training.csv`, `266083_forecast.csv`, `266083_optimization.csv`.
- Analysis in this repo (`main.py`, `src/`, `outputs/`).
- The parallel Task 1–2 workspace `~/ITS8080 Energy Data Science/` (its
  `README.md`, `time_index_audit.md`, `demand_structure.md`, data dictionary).
  That README quotes the brief's Task 11: **5 kW PV, 10 kWh battery, 5 kW grid
  connection**.


---

## The three datasets

| File | Rows | Period (UTC) | Role in the project |
|---|---|---|---|
| `266083_training.csv` | 8,759 | 2013-07-01 → 2014-06-30, hourly | Fit and validate everything |
| `266083_forecast.csv` | 168 | 2014-07-01 → 2014-07-07 | Hold-out week: score demand and PV forecasts |
| `266083_optimization.csv` | 24 | 2014-07-08 | One day to schedule the battery, with two PV scenarios (`pv_low` 1.22, `pv_high` 12.94 kWh/day) |

- **Training columns:** `Demand`, `pv`, `pv_mod1–3`, `Price`, 12 weather columns.
- **Forecast file:** has no `pv_mod*` columns.
- **Optimization file:** replaces `pv` with `pv_low` / `pv_high`. Its `Demand` totals 11.63 kWh and `Price` ranges 0.050–0.102.

---

## 1. Business Understanding

**Goal.** Build the data pipeline and models a HEMS needs to decide, hour by hour:
- whether to use PV directly,
- whether to charge or discharge the battery,
- whether to import from the grid.

The two objectives are:
- a lower electricity bill,
- more of the site's own PV consumed on site.

**Decisions that the data science feeds**

| HEMS decision | Needs | Delivered by |
|---|---|---|
| When to charge or discharge the battery | Next-day demand and PV forecasts, price | Modelling (forecasts) → optimization |
| How much PV is surplus | PV forecast minus demand forecast | Modelling |
| Cost and self-consumption impact | Price, energy balance | Optimization and evaluation |

**Success criteria to fix up front**
- **Forecasts:** MAE and RMSE on the hold-out week, compared against baselines (seasonal naïve at 24 h and 168 h).
- **Optimization:** cost (€) and self-consumption rate (%), compared against a no-battery case and a simple rule-based battery. Report both PV scenarios.

**Site profile (from Tasks 1–2)**
- **A single household, not an aggregate.** Base load is about 0.28 kWh/h and single hours peak at 4.4. Saturday is the highest day and Sunday the lowest.
- **Heating in winter.** Demand is highest in February and lowest in July. Daily mean demand vs temperature: r = −0.25.
- **PV sized just below 5 kW.** The maximum is 4.89, which matches the brief's 5 kW system.

**Decided**
- **Units: kWh.** `Demand`, `pv`, `pv_low` and `pv_high` are read as energy per hour (kWh), so cost is simply Price (€/kWh) × energy (kWh). The equipment ratings stay in kW because they are power limits: the 5 kW PV system and 5 kW grid connection. Over a 1 h step, a 5 kW limit means at most 5 kWh per hour. The battery's 10 kWh is its storage capacity.
- **Decomposition: non-robust MSTL** (daily + weekly, additive). Demand spikes are real consumption. The robust fit down-weights them, which pulls the trend down to the median and leaves residuals averaging +0.089 kWh instead of 0. The trade-off: the spikes it absorbs into St leave some residual autocorrelation at lags 24 and 168 (−0.26 and −0.20). The forecasting model has to handle that.

**Open points to settle from the brief**
- **Country and time zone.** Estonia (Europe/Tallinn).
- **What `Price` is.** It could be a retail import price or a spot price. Also, is export paid, and at what rate? This sets the value of sending PV to the grid.
- **Battery details.** Efficiency, allowed state-of-charge range, and the starting and ending state of charge for the 24 h run.

---

## 2. Data Collection

**Done**
- All three CSVs are in `data/raw/`, untouched, and git-ignored.
- `load_training()` in `main.py` parses the timestamps as UTC.

**To do**
- Loaders for `forecast` and `optimization`. They must handle the different column sets, and `pv_low` / `pv_high` in place of `pv`.
- Record provenance in the README: which file each result comes from.
- Decide on external data (see the dedicated section below). Current conclusion: none is required, and a holiday calendar is optional.

---

## 3. Data Cleaning

**Done (Tasks 1–2)**
- **Missing row:** `2013-12-31 23:00` is missing, a row rather than a `NaN`. The series is reindexed onto a full 8,760-hour grid and that hour is interpolated only where a method needs no gaps. It is flagged in the `filled` column of the decomposition output.
- **Time index:** sorted, no duplicates, UTC. Daylight-saving time is not an issue in UTC (`time_index_audit.md`).
- **Data dictionary:** `outputs/tables/266083_training_data_dictionary.csv`.

**Observations that still need a cleaning decision**

| Issue | Evidence | Proposed handling |
|---|---|---|
| `Shortwave_radiation` is 81% missing | 1,664 of 8,759 rows present | Drop it; use `direct_radiation` + `diffuse_radiation` (complete) |
| `pv_mod1–3` have 5–6% missing | Wherever present, each equals `pv` exactly (r = 1.0) | Treat them as redundant copies; model `pv` only. Note that the modules are not independent measurements. |
| `pv` > 0 while radiation = 0 | 91 hours | Check for a timestamp offset between the PV and weather sources, e.g. by lagging PV against radiation. Fix before any PV model. |
| Five hours of zero demand in a row | 2014-03-03 14:00–18:00 | Probably a metering outage. Flag it, and exclude it from training and evaluation rather than learning it as demand. |
| `Price` = 0 in 12 hours | Data dictionary | Check whether these are genuine (spot prices can reach 0) or missing values |
| Demand spikes 2–4 kWh/h | Isolated hours all year | Keep them: they are real consumption. Decide how they are treated in the loss function (Modelling). |
| Vacancy periods | 33 days with a daily maximum < 0.45, e.g. 9–11 & 19–26 Jul 2013, 23–27 Dec, 23–28 Jan, 25–30 May | Keep the data but add an occupancy flag (Feature Engineering) |

**Output:** one cleaning function in `src/` producing `data/processed/training_clean.csv`, plus a written log of every rule and how many rows it touched.

---

## 4. Data Exploration

**Done (Tasks 1–2)**
- Summary statistics, missing-value counts and the data dictionary (`main.py` steps 1–3).
- PV, demand and price plot for 3–5 Dec 2013 (`outputs/figures/pv_demand_price_2013-12-03_05.png`).
- **Decomposition** (step 5, `src/decomposition.py`). Additive MSTL with daily and weekly periods; the zeros in demand rule out a multiplicative model. Output is `data/processed/266083_demand_decomposition.csv`.
  - **Rt carries 50%** of the variance, St 37%, Tt 5%.
  - Seasonal strength is 0.46; trend strength is 0.12.
  - The trend's big moves are the vacancy periods, not slow drift.
- **Typical profiles** (step 6, `src/profiles.py`):
  - **Two daily peaks:** 08:00 (0.68) and 18:00–20:00 (0.74); night floor about 0.28.
  - **Weekday vs weekend:** the same daily total (12.7 kWh), but weekends start later and peak higher.
  - **Seasons:** winter has the highest evening peak (0.89); summer has the lowest total (11.1 kWh/day).
- **ITS8080 workspace:** lag correlations (1 h, 24 h, 168 h); rolling statistics, including the demonstration that a trailing window leaks unless `shift(1)` is applied; holiday and day-of-month profiles (day of month is noise, p = 0.67).

**Still to do**
- **PV and weather:** PV vs `direct` + `diffuse` radiation. The correlation is only 0.67, which is low for PV and points to the timing issue above. Also explore cloud cover and the seasonal shape of PV.
- **Price:**
  - daily and weekly price profile;
  - how price lines up with demand (r = 0.23; both peak in the evening);
  - how price lines up with PV (the spread between midday and evening decides whether the battery pays off).
- **Energy balance:** hours where PV > demand (the surplus available to the battery), and monthly self-sufficiency without a battery. This is the baseline for the business case.
- **Residual diagnostics:** ACF of Rt and a Ljung–Box test. The result decides whether lagged demand can recover some of the 50% residual.
- **Hold-out week vs training:** compare the July 2014 week (mean 0.43) with July 2013 (mean 0.41, which includes 11 vacancy days).

---

## 5. Feature Engineering

**Done:** lag and rolling features were prototyped in the ITS8080 workspace, with correct `shift(1)` handling for leakage.

**Planned feature set (demand model)**

| Group | Features | Why (evidence) |
|---|---|---|
| Calendar | hour, day of week, weekend flag, month or season; cyclical sin/cos of hour | Daily cycle is the main structure; weekend changes morning timing |
| Lags | `y(t−24)`, `y(t−168)`, and `y(t−1)` only if the forecast horizon allows it | Seasonal-naïve structure. `lag168` is contaminated after vacancy weeks. |
| Rolling | `y.shift(k).rolling(24 / 168)` mean and max | Level of recent use; must be shifted by at least the horizon |
| Occupancy | "vacant" flag (daily maximum < threshold), or recent-level feature | Vacancy periods drive the trend and break the daily profile |
| Weather | Temperature, heating degree-hours, rolling daily temperature | Heating signal: r = −0.25 on daily means |
| Holidays | Public holiday flag (Estonia, pending confirmation) | Small effect; overlaps with vacancies |

**PV model features:** `direct_radiation`, `diffuse_radiation`, `direct_normal_irradiance`, cloud cover (total, low, mid, high), hour and day of year (sun position), and temperature. Apply the timing fix from Data Cleaning first.

**The forecast horizon decides the lags.** Planning for the next day means no lags shorter than 24 h. Define the horizon from the brief before building features. All features are built by one function applied the same way to training, the forecast week and the optimization day.

---

## 6. Modelling

**Demand forecast (hold-out: `266083_forecast.csv`, 168 h)**
1. **Baselines:** seasonal naïve at 24 h and at 168 h, and the average hour-by-weekday profile from `outputs/tables/266083_demand_hour_by_weekday.csv`.
2. **Statistical:** SARIMAX or MSTL + ARIMA on the residual, with temperature as an exogenous input.
3. **Machine learning:** gradient boosting or random forest on the feature set above (scikit-learn is already installed).
4. **Validation:** time-series splits only (expanding window), never shuffled. Include summer folds, because the test week is July.
5. **Metrics:** MAE and RMSE. MAPE is not usable because demand has zeros. Optionally add quantile (pinball) loss to produce ranges.

**PV forecast (same hold-out week)**
- Regression of `pv` on radiation, cloud cover and sun-position features.
- Baselines: persistence at 24 h and clear-sky-shaped climatology.

**Optimization (`266083_optimization.csv`, 24 h)**
- **System:** 5 kW PV, 10 kWh battery, 5 kW grid limit (brief, Task 11).
- **Objective:** minimise Σ price × grid import, net of any export revenue if the brief allows export.
- **Constraints:**
  - energy balance each hour;
  - battery state-of-charge limits and charge/discharge limits;
  - grid limit.
- **Method:** a linear programme (`scipy.optimize.linprog` is enough).
- **Runs:** both PV scenarios (`pv_low` and `pv_high`), each compared against no battery and a rule-based battery.
- **Outputs:** cost (€), grid import (kWh) and self-consumption (%).

---

## 7. Data Visualisation

**Done:**
- the PV, demand and price window;
- decomposition figures for the full year and a two-week close-up;
- the typical-profiles figure (average day, weekday vs weekend, seasons).

**To do**
- **PV vs radiation scatter**, before and after the timing fix.
- **Price profile**, and price shown alongside PV surplus.
- **Forecast vs actual** for the hold-out week, one chart each for demand and PV, with a residual panel; plus a table of errors against the baselines.
- **Optimization:**
  - hourly stacked energy flows (PV → load, PV → battery, battery → load, grid → load);
  - battery state of charge;
  - price;
  - one set per PV scenario.
- **Summary figure for the business case:** cost and self-consumption for each strategy.

Conventions already in use (keep them): UTC axes labelled as such, one y-axis per panel, fixed categorical colours, figures saved from `main.py` into `outputs/figures/`.

---

## Status summary

| Lifecycle stage | Status | Notes |
|---|---|---|
| 1 Business Understanding | Partly done | Site profile known and units fixed (kWh); price type and battery details still open |
| 2 Data Collection | Mostly done | Loaders for forecast and optimization files missing |
| 3 Data Cleaning | Partly done | Time index handled. PV timing, outage hours, price zeros and redundant columns still open. |
| 4 Data Exploration | Demand done | PV, price and energy balance not yet explored |
| 5 Feature Engineering | Prototype only | Lags and rolling features in the ITS8080 workspace |
| 6 Modelling | Not started | Demand and PV forecasts, then the linear programme |
| 7 Data Visualisation | Demand done | Forecast and optimization figures to come |

**Completed in Tasks 1–2**
- Loading, summary, time-index audit, the missing hour, data dictionary and a first multi-variable plot.
- Demand structure: daily, weekly and annual profiles, holidays, lags, rolling features and leakage.
- MSTL decomposition into Tt, St and Rt, with variance shares.
- Typical demand profiles.

**Consistency between the two workspaces**
- **Units: kWh in both.** The ITS8080 workspace was switched over: `demand_kWh`, axis labels, data dictionary and write-ups.
- **Decomposition fit: non-robust in both.** ITS8080 section 33 now keeps the non-robust fit, and the robust one is still computed for comparison. The two workspaces give the same trend strength (0.12) and seasonal strength (0.46).
- **Still open, where the code lives:** the ITS8080 `main.py` has 33 sections in one script; this repo is modular (`src/`). Pick one as the submission code base.

---

## Where the most effort is expected, and why

I expect the most effort to be required in data cleaning, feature engineering, and modelling. From the initial analysis in Tasks 1 and 2, the time-series data appeared relatively noisy, and patterns such as trend and seasonality were not immediately clear. Therefore, the data needs to be carefully explored and prepared before modelling.

Data cleaning can also require significant effort, especially when domain knowledge and business understanding are limited. Missing or incorrect values, unusual observations, or problems in the time series can easily be overlooked, and these may affect the whole analysis and the final results.

However, I expect modelling to be the most challenging part. The modelling results depend strongly on which variables and features are selected. Different feature combinations, models, and scenarios may need to be tested. Even predicting one variable accurately can be difficult, and several variables may influence the results in this dataset.

Therefore, although considerable effort will be needed for cleaning and understanding the data, I expect the largest challenge to be selecting appropriate features, testing different modelling approaches, and evaluating the results.

**Observations from the data behind this**

1. **Demand forecasting (Feature Engineering + Modelling): the largest effort.**
   - **Half of the variation is residual.** The residuals carry 50% of the variance, so a calendar-only model tops out at about half.
   - **The residual is spiky and right-skewed.** Median −0.02, maximum +2.99. Point forecasts will miss peaks, and a peak forecast an hour early or late is penalised twice.
   - **Vacancy periods are regime switches.** Over the 33 days they affect, they drive the trend's big moves and contaminate the 168 h lag. The daily profile is wrong during them unless there is an occupancy feature.
   - **The test week is summer, and training has only one July,** which itself contains 11 vacancy days. That leaves little representative data for the test period.

2. **PV data quality and modelling: medium effort, with a risk that must be cleared early.**
   - There are 91 hours with PV output at zero radiation, and PV vs radiation correlates at only 0.67. Both point to a timestamp offset between sources.
   - This must be fixed before any PV model is trusted.
   - Once aligned, PV should be much easier to predict than demand, because it is driven by radiation, not behaviour.

3. **Optimization: moderate effort, mostly in getting the formulation right.**
   - The linear programme is small (24 h).
   - The effort is in stating the assumptions: export pricing, efficiency, and starting and ending state of charge.
   - It also needs a fair comparison against no battery and a rule-based battery.

4. **Data cleaning: low to moderate.** The problems are known and few; the work is documenting the decisions.

---

## Is additional external data required?

**No external data is required for the core tasks.**
- **Weather** is already included, hourly, in all three files, including the forecast week and the optimization day. That covers the drivers the PV and demand models need.
- **Price** is included for every hour, including the optimization day.
- **System specifications** (5 kW PV, 10 kWh battery, 5 kW grid) come from the brief.

**Optional, low cost:**
- **Public holiday calendar** (the Python `holidays` package, already used in the ITS8080 workspace). The effect is small and mostly overlaps with vacancies. It needs the country confirmed.
- **Sun position** (solar elevation and azimuth), computed from latitude and longitude rather than downloaded. It would help the PV model if the site location is known; if not, day of year and hour stand in for it.

**Not obtainable, and the main limit on accuracy:**
- **Occupancy or appliance-level data.** The residual and vacancy findings show that household behaviour drives most of the unexplained variance. No public source provides it, so it has to be approximated from the demand series itself.

**Caveat to state in the report:**
- The weather in `266083_forecast.csv` is observed, not a real forecast issued in advance. Using it as model input gives an optimistic view of accuracy compared with a live HEMS, which would work from day-ahead weather forecasts.
