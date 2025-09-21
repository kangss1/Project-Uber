# Week 3 — Feature Engineering Summary

**Inputs**
- Cleaned file: `ncr_ride_bookings_cleaned.csv`

**Row-level revenue datasets**
- Full (all columns + one-hot): `ncr_rowlevel_revenue_full.csv`  (rows: 148,767)
- Modeling subset (features + target): `ncr_rowlevel_revenue_model.csv`  (rows: 148,767)
- Numeric features: ['Ride Distance', 'Avg VTAT', 'Avg CTAT', 'Hour']
- One-hot columns (Vehicle Type / Weekday): 15

**Daily demand datasets**
- Soft (keeps early NA lags): `ncr_daily_demand_soft.csv` (rows: 365)
- Hard (drops NA lags/rolls): `ncr_daily_demand_hard.csv` (rows: 351)
- Added features: lag_1, lag_7, lag_14, roll_7, roll_14, roll_28 (+ revenue lags/rolls)

**Notes**
- Parsed Date/Time and added Hour/Weekday/Month.
- De-duplicated by `Booking ID` (kept earliest by date).
- Next (Week 4): Baselines
  - Demand: ARIMA on `rides` + RF/XGB using lags/rolls (compare MAE/RMSE).
  - Revenue: Linear Regression (log(Booking Value) recommended) vs RF/XGB + importances.

---

## Preview — Revenue Modeling Frame (`rev_model`)
|   Ride Distance |   Avg VTAT |   Avg CTAT |   Hour |   Booking Value | Date_parsed         |
|----------------:|-----------:|-----------:|-------:|----------------:|:--------------------|
|          38.69  |        3.8 |      20.1  |     18 |           142   | 2024-01-01 00:00:00 |
|          26.14  |        7.5 |      33.2  |     18 |           616   | 2024-01-01 00:00:00 |
|          26.51  |        7.8 |      23.54 |     17 |           384.6 | 2024-01-01 00:00:00 |
|          36.6   |       11.7 |      18.1  |     20 |           739   | 2024-01-01 00:00:00 |
|          19.672 |        3.4 |      32.48 |      5 |           466   | 2024-01-01 00:00:00 |
|          31.04  |        2.2 |      21.7  |     19 |           147   | 2024-01-01 00:00:00 |
|          41.48  |       13.8 |      35.7  |     17 |           304   | 2024-01-01 00:00:00 |
|          23.23  |        2.8 |      26    |     16 |           434   | 2024-01-01 00:00:00 |

## Preview — Daily Demand Frame (`daily_hard`)
| Date_parsed         |   rides |   revenue |   Weekday_num |   Month |   lag_1 |   lag_7 |   lag_14 |   roll_7 |   roll_14 |   roll_28 |   rev_lag_1 |   rev_lag_7 |   rev_lag_14 |   rev_roll_7 |   rev_roll_14 |   rev_roll_28 |
|:--------------------|--------:|----------:|--------------:|--------:|--------:|--------:|---------:|---------:|----------:|----------:|------------:|------------:|-------------:|-------------:|--------------:|--------------:|
| 2024-01-15 00:00:00 |     410 |    253598 |             0 |       1 |     428 |     387 |      414 |  417.286 |   411.143 |   411.333 |      267962 |      179332 |       202958 |       224105 |        218329 |        217305 |
| 2024-01-16 00:00:00 |     437 |    206667 |             1 |       1 |     410 |     434 |      389 |  417.714 |   414.571 |   412.938 |      253598 |      208943 |       192805 |       223780 |        219319 |        216640 |
| 2024-01-17 00:00:00 |     415 |    193182 |             2 |       1 |     437 |     407 |      384 |  418.857 |   416.786 |   413.059 |      206667 |      191800 |       181049 |       223978 |        220186 |        215260 |
| 2024-01-18 00:00:00 |     422 |    200509 |             3 |       1 |     415 |     435 |      414 |  417     |   417.357 |   413.556 |      193182 |      207280 |       191391 |       223010 |        220837 |        214440 |
| 2024-01-19 00:00:00 |     429 |    204396 |             4 |       1 |     422 |     388 |      416 |  422.857 |   418.286 |   414.368 |      200509 |      180939 |       214409 |       226361 |        220122 |        213912 |
| 2024-01-20 00:00:00 |     437 |    264673 |             5 |       1 |     429 |     419 |      408 |  425.429 |   420.357 |   415.5   |      204396 |      258215 |       253808 |       227284 |        220898 |        216450 |
| 2024-01-21 00:00:00 |     449 |    272018 |             6 |       1 |     437 |     428 |      437 |  428.429 |   421.214 |   417.095 |      264673 |      267962 |       275079 |       227863 |        220680 |        219096 |
| 2024-01-22 00:00:00 |     379 |    175877 |             0 |       1 |     449 |     410 |      387 |  424     |   420.643 |   415.364 |      272018 |      253598 |       179332 |       216760 |        220433 |        217131 |