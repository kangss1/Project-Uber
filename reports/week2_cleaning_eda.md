# Week 2 — Cleaning & Core EDA

**Input**: `ncr_ride_bookings_week1_with_timecols.csv`  
**Output**: `ncr_ride_bookings_cleaned.csv`  
**Rows after de-dup**: 148,767 (removed 1,233 from 150,000)

## Steps Applied
- **Deduplication**: Dropped duplicate `Booking ID`s using a transparent priority:
  (1) keep non-null `Booking Status`, (2) keep the row with the fewest missing values, (3) preserve original order on ties.
- **Imputation**:
  - Events (counts) → 0; event reasons → "Not Applicable".
  - Core categoricals (e.g., Payment Method) → mode.
  - Core numerics → Median for core numerics (baseline).
- **Outliers**:
- Ride Distance: capped to [2.43, 49.20] (1st–99th pct)
- Booking Value: capped to [63.00, 1678.00] (1st–99th pct)
  - Ratings clipped to [0, 5].

## Post-clean Audit
- Exact duplicate rows remaining: **0**
- Missing values (top 15):

|                                   |   n_missing |   pct_missing |
|:----------------------------------|------------:|--------------:|
| Date                              |           0 |             0 |
| Time                              |           0 |             0 |
| Booking ID                        |           0 |             0 |
| Booking Status                    |           0 |             0 |
| Customer ID                       |           0 |             0 |
| Vehicle Type                      |           0 |             0 |
| Pickup Location                   |           0 |             0 |
| Drop Location                     |           0 |             0 |
| Avg VTAT                          |           0 |             0 |
| Avg CTAT                          |           0 |             0 |
| Cancelled Rides by Customer       |           0 |             0 |
| Reason for cancelling by Customer |           0 |             0 |
| Cancelled Rides by Driver         |           0 |             0 |
| Driver Cancellation Reason        |           0 |             0 |
| Incomplete Rides                  |           0 |             0 |


## Visuals
- `week2_distributions.png` — numeric distributions (after cleaning)
- `week2_corr_matrix.png` — numeric correlation heatmap
- `week2_daily_rides.png` — daily ride counts (raw)
- `week2_daily_rides_7day.png` — 7-day rolling average
- `week2_daily_rides_28day.png` — 28-day rolling average
