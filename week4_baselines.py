#%% [markdown]
# ============================================
# Week 4 — Baseline Models (Demand & Revenue)
# ============================================
# - Demand (daily):
#     * ARIMA baseline on 'rides'
#     * RandomForest / XGBoost on lag/rolling features
# - Revenue (row-level):
#     * Linear Regression on log1p(Booking Value)
#     * RandomForest / XGBoost on distance/time/vehicle/VTAT/CTAT
# - Time-aware splits, MAE/RMSE, plots → reports/

#%% 0) Imports & Paths
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import dedent

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional libs (safe fallbacks if missing)
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Project dirs
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

DATA_PROC = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
for d in (DATA_PROC, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

#%% 1) Load engineered datasets (Week 3 outputs)
def _find_latest(patterns):
    candidates = []
    for pat in patterns:
        candidates += list(DATA_PROC.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No files found for patterns: {patterns}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

# Daily demand (hard = no NA lags)
daily_path = _find_latest(["ncr_daily_demand_hard.parquet", "ncr_daily_demand_hard.csv"])
if daily_path.suffix == ".parquet":
    daily = pd.read_parquet(daily_path)
else:
    daily = pd.read_csv(daily_path, parse_dates=["Date_parsed"])

# Revenue modeling (row-level)
rev_path = _find_latest(["ncr_rowlevel_revenue_model.parquet", "ncr_rowlevel_revenue_model.csv"])
if rev_path.suffix == ".parquet":
    rev_model = pd.read_parquet(rev_path)
else:
    # Let pandas guess types; Date_parsed might be string → coerce below
    rev_model = pd.read_csv(rev_path)

print(f"Loaded daily:   {daily_path.name}, shape={daily.shape}")
print(f"Loaded revenue: {rev_path.name}, shape={rev_model.shape}")

# Coerce dates if needed
if "Date_parsed" in daily.columns and not np.issubdtype(daily["Date_parsed"].dtype, np.datetime64):
    daily["Date_parsed"] = pd.to_datetime(daily["Date_parsed"], errors="coerce")
if "Date_parsed" in rev_model.columns and not np.issubdtype(rev_model["Date_parsed"].dtype, np.datetime64):
    rev_model["Date_parsed"] = pd.to_datetime(rev_model["Date_parsed"], errors="coerce")

#%% 2) DEMAND — ARIMA baseline (univariate rides)
N_TEST = 30  # last 30 days as test

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

results_demand = []

if HAS_ARIMA:
    series = daily.set_index("Date_parsed")["rides"].asfreq("D")
    # forward-fill gaps if any (rare but safe)
    series = series.fillna(method="ffill")

    train = series.iloc[:-N_TEST]
    test  = series.iloc[-N_TEST:]

    # Simple baseline (tuned lightly earlier): order=(7,1,1)
    try:
        arima = ARIMA(train, order=(7,1,1))
        arima_fit = arima.fit()
        fcst = arima_fit.forecast(steps=N_TEST)
        mae = mean_absolute_error(test.values, fcst.values)
        r  = rmse(test.values, fcst.values)
        results_demand.append(("ARIMA(7,1,1)", mae, r))

        # Plot
        plt.figure()
        plt.plot(train.index, train.values, label="Train")
        plt.plot(test.index,  test.values,  label="Test")
        plt.plot(test.index,  fcst.values,  label="ARIMA Forecast")
        plt.title("Daily Rides — ARIMA baseline")
        plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / "week4_demand_arima.png", dpi=150)
        plt.close()
    except Exception as e:
        print("ARIMA failed:", e)
else:
    print("statsmodels (ARIMA) not available — skipping ARIMA baseline.")

#%% 3) DEMAND — Tree/Boosted models on lags/rollings
# Features from Week 3:
lag_cols  = [c for c in ["lag_1","lag_7","lag_14"] if c in daily.columns]
roll_cols = [c for c in ["roll_7","roll_14","roll_28"] if c in daily.columns]
cal_cols  = [c for c in ["Weekday_num","Month"] if c in daily.columns]

X_all = daily[lag_cols + roll_cols + cal_cols].copy()
y_all = daily["rides"].copy()

# Time-aware split (last N_TEST days)
X_train, X_test = X_all.iloc[:-N_TEST], X_all.iloc[-N_TEST:]
y_train, y_test = y_all.iloc[:-N_TEST], y_all.iloc[-N_TEST:]

# RandomForest
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yhat_rf = rf.predict(X_test)
results_demand.append(("RF_lags", mean_absolute_error(y_test, yhat_rf), rmse(y_test, yhat_rf)))

plt.figure()
plt.plot(y_all.index[:-N_TEST], y_all.iloc[:-N_TEST], label="Train")
plt.plot(y_all.index[-N_TEST:], y_all.iloc[-N_TEST:], label="Test")
plt.plot(y_all.index[-N_TEST:], yhat_rf, label="RF Forecast")
plt.title("Daily Rides — Random Forest (lags/rollings)")
plt.xlabel("Row (chronological)"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
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
    results_demand.append(("XGB_lags", mean_absolute_error(y_test, yhat_xgb), rmse(y_test, yhat_xgb)))

    plt.figure()
    plt.plot(y_all.index[:-N_TEST], y_all.iloc[:-N_TEST], label="Train")
    plt.plot(y_all.index[-N_TEST:], y_all.iloc[-N_TEST:], label="Test")
    plt.plot(y_all.index[-N_TEST:], yhat_xgb, label="XGB Forecast")
    plt.title("Daily Rides — XGBoost (lags/rollings)")
    plt.xlabel("Row (chronological)"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "week4_demand_xgb_lags.png", dpi=150)
    plt.close()
else:
    print("XGBoost not available — skipping XGB demand model.")

# Print demand results
print("\n== Demand Baselines (MAE / RMSE) ==")
for name, mae, r in results_demand:
    print(f"{name:12s} — MAE: {mae:.2f} | RMSE: {r:.2f}")

#%% 4) REVENUE — Baselines (log target)
# Target transform to handle skew
rev = rev_model.dropna(subset=["Booking Value"]).copy()
rev["log_booking_value"] = np.log1p(rev["Booking Value"])

# Time-aware split by Date_parsed (last 20% as test)
if "Date_parsed" in rev.columns:
    rev = rev.sort_values("Date_parsed")
else:
    print("Warning: Date_parsed missing in rev_model; falling back to simple index order.")

split_idx = int(len(rev) * 0.8)
train_rev = rev.iloc[:split_idx].copy()
test_rev  = rev.iloc[split_idx:].copy()

# Feature columns = everything except targets/dates
drop_cols = {"Booking Value", "log_booking_value", "Date_parsed"}
X_cols = [c for c in rev.columns if c not in drop_cols]
Xtr, Xte = train_rev[X_cols], test_rev[X_cols]
ytr_log  = train_rev["log_booking_value"].values
yte_log  = test_rev["log_booking_value"].values

def _report_rev(name, y_true_log, y_pred_log):
    # Convert back to currency units
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    r   = rmse(y_true, y_pred)
    print(f"{name:12s} — MAE: {mae:.2f} | RMSE: {r:.2f}")
    return mae, r

rev_results = []

# Linear Regression
lin = LinearRegression()
lin.fit(Xtr, ytr_log)
yhat_lin_log = lin.predict(Xte)
rev_results.append(("Linear(logY)",) + _report_rev("Linear(logY)", yte_log, yhat_lin_log))

# RandomForest
rf_rev = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_rev.fit(Xtr, ytr_log)
yhat_rf_log = rf_rev.predict(Xte)
rev_results.append(("RF(logY)",) + _report_rev("RF(logY)", yte_log, yhat_rf_log))

# XGBoost (optional)
if HAS_XGB:
    xgb_rev = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    xgb_rev.fit(Xtr, ytr_log)
    yhat_xgb_log = xgb_rev.predict(Xte)
    rev_results.append(("XGB(logY)",) + _report_rev("XGB(logY)", yte_log, yhat_xgb_log))
else:
    print("XGBoost not available — skipping XGB revenue model.")

# Residual sanity plot (RF example) — in log space
plt.figure()
plt.scatter(range(len(yte_log)), yte_log - yhat_rf_log, s=8)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Revenue — RF Residuals (log space)")
plt.xlabel("Test rows (chronological)"); plt.ylabel("Residual (log1p units)")
plt.tight_layout(); plt.savefig(REPORTS / "week4_revenue_rf_residuals_log.png", dpi=150); plt.close()

# Feature importances (RF)
imp = pd.Series(rf_rev.feature_importances_, index=X_cols).sort_values(ascending=False)
imp.to_csv(REPORTS / "week4_revenue_rf_importances.csv")
print("\nTop RF Revenue Importances:")
print(imp.head(12))

#%% 5) Write a short summary to reports/
summary = dedent(f"""
# Week 4 — Baseline Results

## Demand (Daily Rides)
Test window: last {N_TEST} days
Models evaluated:
{chr(10).join([f"- {n}: MAE={mae:.2f}, RMSE={r:.2f}" for n, mae, r in results_demand])}

Plots:
- `week4_demand_arima.png` (if ARIMA available)
- `week4_demand_rf_lags.png`
- `week4_demand_xgb_lags.png` (if XGBoost available)

## Revenue (Row-level Booking Value; evaluated in original currency units)
Models evaluated:
{chr(10).join([f"- {n}: MAE={mae:.2f}, RMSE={r:.2f}" for n, mae, r in rev_results])}

Artifacts:
- `week4_revenue_rf_residuals_log.png`
- `week4_revenue_rf_importances.csv`
""").strip()

with open(REPORTS / "week4_baselines_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote:", REPORTS / "week4_baselines_summary.md")
print("✅ Week 4 baselines complete.")