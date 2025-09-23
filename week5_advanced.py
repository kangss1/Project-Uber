#%% [markdown]
# ============================================
#  Week 5 — Advanced Demand Models (Daily Rides)
# ============================================
# Steps:
# 1) Load engineered daily dataset (lags/rollings/calendar)
# 2) Prepare features & time-aware train/test split
# 3) Random Forest with TimeSeriesSplit + GridSearchCV
# 4) XGBoost with TimeSeriesSplit + GridSearchCV (optional)
# 5) (Optional) SHAP explainability + summary report/plots/models
#
# Inputs (from Week 3/4):
#   data/processed/ncr_daily_demand_hard.csv
#
# Outputs:
#   reports/week5_demand_rf_tuned.png
#   reports/week5_demand_xgb_tuned.png              (if XGBoost available)
#   reports/week5_demand_xgb_importance.png         (if XGBoost available)
#   reports/week5_demand_xgb_shap_summary.png       (if SHAP & XGBoost available)
#   reports/week5_demand_summary.md
#   models/week5_best_rf.pkl
#   models/week5_best_xgb.json, week5_best_xgb_params.json  (if XGBoost available)

#%% 
# 0) Imports & Paths
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from textwrap import dedent
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Optional deps (handled gracefully)
try:
    from xgboost import XGBRegressor, plot_importance
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Project dirs
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

DATA_PROC = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
MODELS    = ROOT / "models"
for d in (DATA_PROC, REPORTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

#%% 
# 1) Load engineered daily dataset (Week 3/4 output)
daily_path = DATA_PROC / "ncr_daily_demand_hard.csv"
assert daily_path.exists(), f"Cannot find {daily_path} — run Week 3/4 first."
daily = pd.read_csv(daily_path, parse_dates=["Date_parsed"])
print("Loaded daily demand:", daily.shape)

# Features prepared in Week 3
lag_cols  = [c for c in ["lag_1","lag_7","lag_14"] if c in daily.columns]
roll_cols = [c for c in ["roll_7","roll_14","roll_28"] if c in daily.columns]
cal_cols  = [c for c in ["Weekday_num","Month"] if c in daily.columns]
feature_cols = lag_cols + roll_cols + cal_cols

missing_expected = [c for c in ["lag_1","lag_7","lag_14","roll_7","roll_14","roll_28","Weekday_num","Month"]
                    if c not in daily.columns]
if len(feature_cols) == 0:
    raise ValueError(f"No usable features found. Missing examples: {missing_expected}")

X = daily[feature_cols].copy()
y = daily["rides"].copy()

#%% 
# 2) Time-aware split (last N_TEST days)
N_TEST = 30
if len(daily) <= N_TEST + 10:
    raise ValueError(f"Too few rows ({len(daily)}) for a {N_TEST}-day test split.")

X_train, X_test = X.iloc[:-N_TEST], X.iloc[-N_TEST:]
y_train, y_test = y.iloc[:-N_TEST], y.iloc[-N_TEST:]
dates_train = daily["Date_parsed"].iloc[:-N_TEST]
dates_test  = daily["Date_parsed"].iloc[-N_TEST:]

print(f"Train days: {len(X_train)} | Test days: {len(X_test)}")
print("Features used:", feature_cols)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#%% 
# 3) Random Forest — Tuning with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [6, 12, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
yhat_rf  = best_rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, yhat_rf)
rmse_rf = RMSE(y_test, yhat_rf)
print("\nBest RF Params:", grid_rf.best_params_)
print(f"RF Tuned — MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}")

# Plot RF forecast
plt.figure()
plt.plot(dates_train, y_train, label="Train")
plt.plot(dates_test,  y_test,  label="Test")
plt.plot(dates_test,  yhat_rf, label="RF Tuned Forecast")
plt.title("Daily Rides — Random Forest (Tuned)")
plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
plt.savefig(REPORTS / "week5_demand_rf_tuned.png", dpi=150)
plt.close()

# Persist RF
with open(MODELS / "week5_best_rf.pkl", "wb") as f:
    pickle.dump(best_rf, f)

results = [("RF_tuned", mae_rf, rmse_rf)]

#%% 
# 4) XGBoost — Tuning with TimeSeriesSplit (optional)
xgb_summary_line = ""

if HAS_XGB:
    xgb_param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    grid_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=xgb_param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid_xgb.fit(X_train, y_train)

    best_xgb = grid_xgb.best_estimator_
    yhat_xgb = best_xgb.predict(X_test)

    mae_xgb  = mean_absolute_error(y_test, yhat_xgb)
    rmse_xgb = RMSE(y_test, yhat_xgb)
    print("\nBest XGB Params:", grid_xgb.best_params_)
    print(f"XGB Tuned — MAE: {mae_xgb:.2f} | RMSE: {rmse_xgb:.2f}")

    results.append(("XGB_tuned", mae_xgb, rmse_xgb))

    # Plot XGB forecast
    plt.figure()
    plt.plot(dates_train, y_train, label="Train")
    plt.plot(dates_test,  y_test,  label="Test")
    plt.plot(dates_test,  yhat_xgb, label="XGB Tuned Forecast")
    plt.title("Daily Rides — XGBoost (Tuned)")
    plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "week5_demand_xgb_tuned.png", dpi=150)
    plt.close()

    # XGB feature importance (gain)
    try:
        plot_importance(best_xgb, max_num_features=12, importance_type="gain")
        plt.title("XGB Feature Importance — Demand")
        plt.tight_layout()
        plt.savefig(REPORTS / "week5_demand_xgb_importance.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Persist XGB
    best_xgb.save_model(str(MODELS / "week5_best_xgb.json"))
    with open(MODELS / "week5_best_xgb_params.json", "w") as f:
        json.dump(grid_xgb.best_params_, f, indent=2)
else:
    xgb_summary_line = "\n- XGBoost not available — skipped."

#%% 
# 5) (Optional) SHAP explanation (works best with XGBoost)
if HAS_XGB and HAS_SHAP:
    try:
        background = X_train.sample(min(200, len(X_train)), random_state=42)
        explainer = shap.Explainer(best_xgb, background)
        shap_values = explainer(X_test)

        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS / "week5_demand_xgb_shap_summary.png", dpi=150)
        plt.close()
    except Exception as e:
        print("SHAP step skipped:", e)

#%% 
# 5b) Save Markdown summary
summary = dedent(f"""
# Week 5 — Advanced Demand Models

**Data**
- Source: `{daily_path.name}`
- Train size: {len(X_train)} days
- Test size: {len(X_test)} days
- Features: {feature_cols}

**Results (30-day holdout)**
{chr(10).join([f"- {n}: MAE={mae:.2f}, RMSE={rmse:.2f}" for n, mae, rmse in results])}{xgb_summary_line}

**Artifacts**
- `week5_demand_rf_tuned.png`
- `week5_demand_xgb_tuned.png` (if available)
- `week5_demand_xgb_importance.png` (if available)
- `week5_demand_xgb_shap_summary.png` (if available)

**Models**
- `models/week5_best_rf.pkl`
- `models/week5_best_xgb.json` (if available)
- `models/week5_best_xgb_params.json` (if available)
""").strip()

with open(REPORTS / "week5_demand_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote:", REPORTS / "week5_demand_summary.md")
print("Week 5 demand modeling complete.")
# %%
