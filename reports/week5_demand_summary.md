# Week 5 — Advanced Demand Models

**Data**
- Source: `ncr_daily_demand_hard.csv`
- Train size: 321 days
- Test size: 30 days
- Features: ['lag_1', 'lag_7', 'lag_14', 'roll_7', 'roll_14', 'roll_28', 'Weekday_num', 'Month']

**Results (30-day holdout)**
- RF_tuned: MAE=14.84, RMSE=18.45
- XGB_tuned: MAE=14.86, RMSE=18.69

**Artifacts**
- `week5_demand_rf_tuned.png`
- `week5_demand_xgb_tuned.png` (if available)
- `week5_demand_xgb_importance.png` (if available)
- `week5_demand_xgb_shap_summary.png` (if available)

**Models**
- `models/week5_best_rf.pkl`
- `models/week5_best_xgb.json` (if available)
- `models/week5_best_xgb_params.json` (if available)