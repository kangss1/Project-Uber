# Week 5 — Advanced Demand Models

**Data**
- Source: `ncr_daily_demand_hard.csv`
- Train size: 321 days
- Test size: 30 days
- Features: ['lag_1', 'lag_7', 'lag_14', 'roll_7', 'roll_14', 'roll_28', 'Weekday_num', 'Month', 'is_weekend', 'fourier_sin_1', 'fourier_cos_1', 'fourier_sin_2', 'fourier_cos_2']

**Holdout (last 30 days)**
- Naive_last: MAE=22.77, RMSE=28.28
- Naive_7: MAE=25.57, RMSE=30.68
- RF_tuned: MAE=15.25, RMSE=19.55
- XGB_tuned: MAE=17.15, RMSE=21.84

**Artifacts**
- `week5_demand_rf_tuned.png`
- `week5_demand_xgb_tuned.png` (if available)
- `week5_demand_xgb_importance.png` (if available)
- `week5_demand_leaderboard.csv` (all CV trials, lower is better)
- `week5_demand_holdout_metrics.csv`
- `week5_demand_xgb_shap_summary.png` (if available)

**Models**
- `models/week5_best_rf.pkl`
- `models/week5_best_xgb.json` (if available)
- `models/week5_best_xgb_params.json` (if available)