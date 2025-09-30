# Week 6 — Advanced Revenue Models (Summary, no leakage)

**Input:** `ncr_rowlevel_revenue_model.csv`
**Train rows:** 119,013 | **Test rows:** 29,754
**Features (no leakage):** 23

**Results (20% holdout, original currency)**
- RF(logY): MAE=180.86, RMSE=303.37
- XGB(logY): MAE=181.22, RMSE=303.57

**Interpretability**
- RF impurity + permutation importances
- XGB gain importances (if trained)
- PDPs: Ride Distance, Hour
- SHAP: RF always; XGB if competitive

**Segmentation**
- Vehicle & Hour means; KMeans on scaled [Ride Distance, Avg VTAT, Avg CTAT, Hour]

**Notes**
- **price_per_km** computed for analysis but **excluded from training** to avoid target leakage.
- XGB search adds `min_child_weight` and `gamma` for stability; early stopping in constructor.