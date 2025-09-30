#%%
# ============================================
# Week 4 — Baseline Models (Demand & Revenue)
# File: week4_baselines.py
# Depends on: Week 3 outputs in data/processed/
# ============================================
#
# Purpose
# Build strong but simple baselines to benchmark later tuning (Week 5+) and
# revenue modeling (Week 6). This version adds:
#   - Demand: Naive (last & seasonal-7), small ARIMA/SARIMA grid, and RF/XGB
#     on lag/rolling features with a time-aware split.
#   - Revenue: Log-target linear baselines WITH SCALING (Linear/Ridge/Lasso),
#     plus RF/XGB; residual plots & importances.
#   - Reporting: Saves CSV leaderboards and PNG plots to /reports.
#
# Outputs (saved under /reports unless noted)
#   demand:
#     - week4_demand_arima.png, week4_demand_sarima.png
#     - week4_demand_rf_lags.png, week4_demand_xgb_lags.png
#     - week4_demand_baselines.csv (MAE/RMSE table)
#   revenue:
#     - week4_revenue_rf_residuals_log.png
#     - week4_revenue_rf_importances.csv, week4_revenue_rf_importances_plot.png
#     - week4_revenue_baselines.csv (MAE/RMSE table; units = original currency)
#
# Notes
#   - We hold out the last N_TEST days for demand (chronological split).
#   - For revenue, we sort by Date_parsed and use the last 20% as test.
#   - XGBoost/Statsmodels are optional; code guards will skip if unavailable.

#%% 0) Imports & paths
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Optional libs (safe fallbacks if missing)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Dirs
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()
DATA_PROC = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
for d in (DATA_PROC, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

#%% 1) Load engineered datasets (Week 3 outputs)
def _latest(patterns):
    cands = []
    for p in patterns:
        cands += list(DATA_PROC.glob(p))
    if not cands:
        raise FileNotFoundError(f"No files for: {patterns}")
    return max(cands, key=lambda p: p.stat().st_mtime)

# Daily demand (hard = no NA lags)
daily_path = _latest(["ncr_daily_demand_hard.parquet", "ncr_daily_demand_hard.csv"])
daily = (pd.read_parquet(daily_path) if daily_path.suffix==".parquet"
         else pd.read_csv(daily_path, parse_dates=["Date_parsed"]))

# Revenue (row-level modeling frame from Week 3)
rev_path = _latest(["ncr_rowlevel_revenue_model.parquet", "ncr_rowlevel_revenue_model.csv"])
rev_model = (pd.read_parquet(rev_path) if rev_path.suffix==".parquet"
             else pd.read_csv(rev_path))
if "Date_parsed" in rev_model and not np.issubdtype(rev_model["Date_parsed"].dtype, np.datetime64):
    rev_model["Date_parsed"] = pd.to_datetime(rev_model["Date_parsed"], errors="coerce")

print(f"Loaded daily:   {daily_path.name}, shape={daily.shape}")
print(f"Loaded revenue: {rev_path.name}, shape={rev_model.shape}")

#%% 2) DEMAND — Baselines (last N_TEST days holdout)
N_TEST = 30
results_demand = []

# --- Prepare series & time split
s = (daily.set_index("Date_parsed")["rides"]
           .asfreq("D")
           .ffill())  # fill occasional gaps
train = s.iloc[:-N_TEST]
test  = s.iloc[-N_TEST:]
dates_train, dates_test = train.index, test.index

def _add_result(name, y_true, y_pred):
    results_demand.append({
        "Model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred))
    })

# --- Naive baselines
# last-value
_nv = np.repeat(train.iloc[-1], len(test))
_add_result("Naive-last", test.values, _nv)
# seasonal-7
_nv7 = s.shift(7).iloc[-N_TEST:].values
_add_result("Naive-seasonal-7", test.values, _nv7)

# --- ARIMA / SARIMA (small grid)
if HAS_ARIMA:
    # ARIMA grid
    arima_grid = [(1,1,0), (2,1,1), (7,1,1)]
    best = None
    for order in arima_grid:
        try:
            m = ARIMA(train, order=order).fit()
            fc = m.forecast(steps=N_TEST)
            mae = mean_absolute_error(test.values, fc.values)
            rmse = root_mean_squared_error(test.values, fc.values)
            _add_result(f"ARIMA{order}", test.values, fc.values)
            if best is None or mae < best[0]:
                best = (mae, rmse, order, fc)
        except Exception:
            continue
    if best is not None:
        _, _, best_order, fc = best
        plt.figure()
        plt.plot(dates_train, train.values, label="Train")
        plt.plot(dates_test,  test.values,  label="Test")
        plt.plot(dates_test,  fc.values,    label=f"ARIMA{best_order}")
        plt.title("Daily Rides — ARIMA baseline")
        plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / "week4_demand_arima.png", dpi=150)
        plt.close()

    # SARIMA weekly seasonal
    try:
        sar = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,7),
                      enforce_stationarity=False, enforce_invertibility=False)
        sar_fit = sar.fit(disp=False)
        sar_fc  = sar_fit.forecast(steps=N_TEST)
        _add_result("SARIMA(1,1,1)(1,0,1,7)", test.values, sar_fc.values)

        plt.figure()
        plt.plot(dates_train, train.values, label="Train")
        plt.plot(dates_test,  test.values,  label="Test")
        plt.plot(dates_test,  sar_fc.values, label="SARIMA")
        plt.title("Daily Rides — SARIMA(1,1,1)(1,0,1,7)")
        plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / "week4_demand_sarima.png", dpi=150)
        plt.close()
    except Exception:
        pass
else:
    print("statsmodels not available — skipping ARIMA/SARIMA.")

# --- Feature models: RF/XGB on Week-3 features
lag_cols  = [c for c in ["lag_1","lag_7","lag_14"] if c in daily.columns]
roll_cols = [c for c in ["roll_7","roll_14","roll_28"] if c in daily.columns]
cal_cols  = [c for c in ["Weekday_num","Month"] if c in daily.columns]

X = daily[lag_cols + roll_cols + cal_cols].copy()
y = daily["rides"].copy()
X_train, X_test = X.iloc[:-N_TEST], X.iloc[-N_TEST:]
y_train, y_test = y.iloc[:-N_TEST], y.iloc[-N_TEST:]

# Random Forest
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yhat_rf = rf.predict(X_test)
_add_result("RF_lags", y_test.values, yhat_rf)

plt.figure()
plt.plot(dates_train, y_train, label="Train")
plt.plot(dates_test,  y_test,  label="Test")
plt.plot(dates_test,  yhat_rf, label="RF Forecast")
plt.title("Daily Rides — Random Forest (lags/rollings)")
plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
plt.savefig(REPORTS / "week4_demand_rf_lags.png", dpi=150)
plt.close()

# XGBoost (optional)
if HAS_XGB:
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    xgb.fit(X_train, y_train)
    yhat_xgb = xgb.predict(X_test)
    _add_result("XGB_lags", y_test.values, yhat_xgb)

    plt.figure()
    plt.plot(dates_train, y_train, label="Train")
    plt.plot(dates_test,  y_test,  label="Test")
    plt.plot(dates_test,  yhat_xgb, label="XGB Forecast")
    plt.title("Daily Rides — XGBoost (lags/rollings)")
    plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "week4_demand_xgb_lags.png", dpi=150)
    plt.close()
else:
    print("XGBoost not available — skipping demand XGB.")

# Save demand results
df_dem = pd.DataFrame(results_demand).sort_values(["MAE","RMSE"])
df_dem.to_csv(REPORTS / "week4_demand_baselines.csv", index=False)
print("\n== Demand baselines ==")
print(df_dem.to_string(index=False))

#%% 3) REVENUE — Baselines (log target, time-aware split)
# Prepare data
rev = rev_model.dropna(subset=["Booking Value"]).copy()
if "Date_parsed" in rev.columns:
    rev = rev.sort_values("Date_parsed")
split_idx = int(len(rev)*0.8)
train_rev, test_rev = rev.iloc[:split_idx].copy(), rev.iloc[split_idx:].copy()

# Features = everything except targets/dates
drop_cols = {"Booking Value","Date_parsed"}
X_cols = [c for c in rev.columns if c not in drop_cols]
# log target
train_rev["logY"] = np.log1p(train_rev["Booking Value"])
test_rev["logY"]  = np.log1p(test_rev["Booking Value"])

Xtr, Xte = train_rev[X_cols].values, test_rev[X_cols].values
ytr_log, yte_log = train_rev["logY"].values, test_rev["logY"].values

def _report_rev(name, y_true_log, y_pred_log, bag):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    bag.append({"Model": name,
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(root_mean_squared_error(y_true, y_pred))})

rev_rows = []

# Linear/Ridge/Lasso with scaling
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

lin = LinearRegression().fit(Xtr_s, ytr_log)
_report_rev("Linear(logY)+scale", yte_log, lin.predict(Xte_s), rev_rows)

ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr_log)
_report_rev("Ridge(logY)+scale", yte_log, ridge.predict(Xte_s), rev_rows)

lasso = Lasso(alpha=0.001, max_iter=10000).fit(Xtr_s, ytr_log)
_report_rev("Lasso(logY)+scale", yte_log, lasso.predict(Xte_s), rev_rows)

# Random Forest on log target (no scaling)
rf_rev = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_rev.fit(Xtr, ytr_log)
yhat_rf_log = rf_rev.predict(Xte)
_report_rev("RF(logY)", yte_log, yhat_rf_log, rev_rows)

# XGBoost on log target (optional)
if HAS_XGB:
    xgb_rev = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, objective="reg:squarederror"
    ).fit(Xtr, ytr_log)
    _report_rev("XGB(logY)", yte_log, xgb_rev.predict(Xte), rev_rows)
else:
    print("XGBoost not available — skipping revenue XGB.")

# Results table
df_rev = pd.DataFrame(rev_rows).sort_values(["MAE","RMSE"])
df_rev.to_csv(REPORTS / "week4_revenue_baselines.csv", index=False)
print("\n== Revenue baselines (original currency) ==")
print(df_rev.to_string(index=False))

# Residuals (RF in log space)
plt.figure()
plt.scatter(range(len(yte_log)), yte_log - yhat_rf_log, s=8)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Revenue — RF Residuals (log space)")
plt.xlabel("Test rows (chronological)"); plt.ylabel("Residual (log1p units)")
plt.tight_layout()
plt.savefig(REPORTS / "week4_revenue_rf_residuals_log.png", dpi=150)
plt.close()

# RF feature importances
imp = pd.Series(rf_rev.feature_importances_, index=X_cols).sort_values(ascending=False)
imp.to_csv(REPORTS / "week4_revenue_rf_importances.csv")
imp_plot = imp.head(15)[::-1]
plt.figure(figsize=(7,5))
plt.barh(imp_plot.index, imp_plot.values)
plt.title("Revenue — RF Feature Importances (Top 15)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(REPORTS / "week4_revenue_rf_importances_plot.png", dpi=150)
plt.close()

#%% 4) Summary markdown
summary = dedent(f"""
# Week 4 — Baseline Results

## Demand (Daily rides) — last {N_TEST} days holdout
{df_dem.to_string(index=False)}

Artifacts:
- week4_demand_baselines.csv
- week4_demand_arima.png (best ARIMA)
- week4_demand_sarima.png
- week4_demand_rf_lags.png
- week4_demand_xgb_lags.png

## Revenue (Booking Value; metrics in original units)
{df_rev.to_string(index=False)}

Artifacts:
- week4_revenue_baselines.csv
- week4_revenue_rf_residuals_log.png
- week4_revenue_rf_importances.csv
- week4_revenue_rf_importances_plot.png
""").strip()

with open(REPORTS / "week4_baselines_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote:", REPORTS / "week4_baselines_summary.md")
print("Week 4 baselines complete.")
#%%