# Week 2 — Cleaning & Core EDA
- Input file: ncr_ride_bookings_week1_with_timecols.csv
- Output file: ncr_ride_bookings_cleaned.csv

## Cleaning Steps Applied
- Missing event columns imputed (0 / Not Applicable)
- Numeric columns imputed with KNN
- Outliers clipped (Distance >200km, Value >10,000, Ratings → [0,5])

## Generated Visuals
- week2_distributions.png (numeric distributions)
- week2_corr_matrix.png (correlation heatmap)
- week2_daily_rides.png (daily rides)
- week2_daily_rides_7day.png (7-day rolling)
- week2_daily_rides_28day.png (28-day rolling)
