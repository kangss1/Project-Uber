# Week 4 — Baseline Results

## Demand (Daily Rides)
Test window: last 30 days
Models evaluated:
- ARIMA(7,1,1): MAE=17.31, RMSE=21.47
- SARIMA(1,1,1)(1,0,1,7): MAE=17.33, RMSE=21.50
- RF_lags: MAE=15.81, RMSE=19.40
- XGB_lags: MAE=15.35, RMSE=19.04

Plots:
- `week4_demand_arima.png` (if ARIMA available)
- `week4_demand_rf_lags.png`
- `week4_demand_xgb_lags.png` (if XGBoost available)

## Revenue (Row-level Booking Value; evaluated in original currency units)
Models evaluated:
- Linear(logY): MAE=248.82, RMSE=394.74
- RF(logY): MAE=192.58, RMSE=351.66
- XGB(logY): MAE=199.63, RMSE=349.34

Artifacts:
- `week4_revenue_rf_residuals_log.png`
- `week4_revenue_rf_importances.csv`