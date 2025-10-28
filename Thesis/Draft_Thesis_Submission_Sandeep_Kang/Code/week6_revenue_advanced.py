#%% [markdown]
# ============================================
# Week 6 — Advanced Revenue Models (Clean, SHAP-ready)
# File: week6_revenue_advanced.py
# Depends on: Week 3 -> data/processed/ncr_rowlevel_revenue_model.(csv|parquet)
# ============================================
#
# Purpose
#   Train revenue models (RF, XGB) on **non-leaky features** and produce reliable
#   interpretability (Permutation/XGB importances, PDPs, SHAP). Segment revenue
#   by vehicle/time and add clustering (scaled).
#
# Key Fix
#   Compute `price_per_km` for *descriptives*, but **exclude from training** to
#   prevent target leakage (it divides Booking Value by Distance).
#
# Outputs
#   reports/week6_revenue_rf_importances.png
#   reports/week6_revenue_xgb_importances.png
#   reports/week6_revenue_rf_residuals_log.png
#   reports/week6_revenue_pdp_distance.png
#   reports/week6_revenue_pdp_hour.png
#   reports/week6_revenue_shap_bar_rf.png
#   reports/week6_revenue_shap_beeswarm_rf.png
#   (if XGB competitive) week6_revenue_shap_bar_xgb.png / _beeswarm_xgb.png
#   reports/week6_revenue_segment_vehicle.png
#   reports/week6_revenue_segment_hour.png
#   reports/week6_revenue_clusters_scatter.png
#   reports/week6_revenue_summary.md
#   data/processed/week6_revenue_metrics.csv
#   models/week6_rf_revenue.pkl
#   models/week6_xgb_revenue.json, models/week6_xgb_revenue_params.json
# ============================================

#%% 0) Imports & Paths
from pathlib import Path
import json, pickle
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Optional deps
try:
    from xgboost import XGBRegressor
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

RANDOM_STATE   = 42
TS_FOLDS       = 3
TUNE_ROWS      = 20000   # small tuning slice keeps things cool

#%% 1) Load engineered revenue dataset
def _find_latest(patterns):
    cands = []
    for pat in patterns:
        cands += list(DATA_PROC.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No files found for {patterns} in {DATA_PROC}")
    return max(cands, key=lambda p: p.stat().st_mtime)

rev_path = _find_latest(["ncr_rowlevel_revenue_model.parquet",
                         "ncr_rowlevel_revenue_model.csv"])
rev = pd.read_parquet(rev_path) if rev_path.suffix.lower()==".parquet" else pd.read_csv(rev_path)

if "Date_parsed" in rev.columns and not np.issubdtype(rev["Date_parsed"].dtype, np.datetime64):
    rev["Date_parsed"] = pd.to_datetime(rev["Date_parsed"], errors="coerce")

rev = rev.dropna(subset=["Booking Value"]).copy()
print(f"Loaded revenue modeling frame: {rev_path.name}, shape={rev.shape}")

#%% 2) Light extra features (safe) + OHE
def _safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return num / den

# Compute but DO NOT TRAIN WITH price_per_km (leaky)
if {"Booking Value","Ride Distance"} <= set(rev.columns):
    rev["price_per_km"] = _safe_div(rev["Booking Value"], rev["Ride Distance"])

if "Avg CTAT" in rev.columns:
    rev["speed_proxy"] = _safe_div(rev["Ride Distance"], pd.to_numeric(rev["Avg CTAT"], errors="coerce")/60.0)

if "Hour" in rev.columns:
    rev["rush_am"] = rev["Hour"].between(7,10).astype(int)
    rev["rush_pm"] = rev["Hour"].between(16,19).astype(int)
if "Weekday" in rev.columns:
    rev["is_weekend"] = rev["Weekday"].isin(["Saturday","Sunday"]).astype(int)
elif "Weekday_num" in rev.columns:
    rev["is_weekend"] = rev["Weekday_num"].isin([5,6]).astype(int)

if any(c in rev.columns for c in ["Vehicle Type","Weekday"]):
    rev = pd.get_dummies(
        rev,
        columns=[c for c in ["Vehicle Type","Weekday"] if c in rev.columns],
        prefix=[c for c in ["Vehicle Type","Weekday"] if c in rev.columns],
        dummy_na=False
    )

#%% 3) Target + Feature matrix (EXCLUDE leaky cols)
rev["log_booking_value"] = np.log1p(pd.to_numeric(rev["Booking Value"], errors="coerce"))
rev = rev.dropna(subset=["log_booking_value"])

drop_cols = {"Booking Value","log_booking_value","Date_parsed",
             "price_per_km"}  # <-- leakage removed
X_cols = [c for c in rev.columns if c not in drop_cols]

X = rev[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64)
y_log = rev["log_booking_value"].to_numpy(dtype=np.float64)

# Chronological 80/20 split
if "Date_parsed" in rev.columns and rev["Date_parsed"].notna().any():
    order = np.argsort(rev["Date_parsed"].values)
    X, y_log = X.iloc[order].reset_index(drop=True), y_log[order]
else:
    print("Warning: Date_parsed missing; using index order.")

split = int(len(X)*0.80)
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr_log, yte_log = y_log[:split], y_log[split:]

print(f"Train rows: {len(Xtr):,} | Test rows: {len(Xte):,}")
print(f"Features used (no leakage): {len(X.columns)}")

def report_real(name, y_true_log, y_pred_log):
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    print(f"{name:18s} MAE={mae:,.2f} | RMSE={rmse:,.2f}")
    return mae, rmse

tscv = TimeSeriesSplit(n_splits=TS_FOLDS)

#%% 4) Random Forest — RandomizedSearchCV on a slice, refit full
Xtr_tune = Xtr.iloc[:min(TUNE_ROWS, len(Xtr))]
ytr_tune = ytr_log[:len(Xtr_tune)]

rf_base = RandomForestRegressor(
    n_estimators=80, bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE
)
rf_param_dist = {
    "max_depth": [None, 10, 16],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3]
}
rf_search = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=8, cv=tscv,
    scoring="neg_mean_absolute_error", n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
rf_search.fit(Xtr_tune, ytr_tune)
rf_best = rf_search.best_estimator_
rf_best.set_params(n_estimators=250)         # stronger final fit
rf_best.fit(Xtr, ytr_log)

yhat_rf_log = rf_best.predict(Xte)
rf_mae, rf_rmse = report_real("RF(log target)", yte_log, yhat_rf_log)

# Save + plots
with open(MODELS / "week6_rf_revenue.pkl", "wb") as f:
    pickle.dump(rf_best, f)

imp_rf = pd.Series(rf_best.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)[::-1]
plt.figure(figsize=(7,6)); plt.barh(imp_rf.index, imp_rf.values)
plt.title("Revenue — RF Feature Importances (Top 20)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(REPORTS / "week6_revenue_rf_importances.png", dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.scatter(range(len(yte_log)), yte_log - yhat_rf_log, s=6)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Revenue — RF Residuals (log space)")
plt.xlabel("Test rows (chronological)"); plt.ylabel("Residual (log1p units)")
plt.tight_layout(); plt.savefig(REPORTS / "week6_revenue_rf_residuals_log.png", dpi=150); plt.close()

# Permutation importance (top 12)
perm = permutation_importance(rf_best, Xte, yte_log, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
idx = np.argsort(perm.importances_mean)[-12:]
plt.figure(figsize=(7,6)); plt.barh(np.array(X.columns)[idx], perm.importances_mean[idx])
plt.title("RF Permutation Importance (log target)"); plt.xlabel("Mean importance (Δscore)")
plt.tight_layout(); plt.savefig(REPORTS / "week6_revenue_perm_importance_rf.png", dpi=150); plt.close()

results = [("RF(logY)", rf_mae, rf_rmse)]
best_model, best_name = rf_best, "RF"

#%% 5) XGBoost — RandomizedSearchCV (regularized, early stopping)
if HAS_XGB:
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=50,  # set in constructor to avoid warnings
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_param_dist = {
        "n_estimators": [400, 700, 1000],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.06, 0.10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [3.0, 5.0, 8.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.5, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_param_dist, n_iter=24, cv=tscv,
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    cut = int(len(Xtr)*0.90)
    xgb_search.fit(Xtr, ytr_log, eval_set=[(Xtr.iloc[cut:], ytr_log[cut:])], verbose=False)

    xgb_best = xgb_search.best_estimator_
    cut2 = int(len(Xtr)*0.95)
    xgb_best.fit(Xtr.iloc[:cut2], ytr_log[:cut2],
                 eval_set=[(Xtr.iloc[cut2:], ytr_log[cut2:])],
                 verbose=False)

    yhat_xgb_log = xgb_best.predict(Xte)
    xgb_mae, xgb_rmse = report_real("XGB(log target)", yte_log, yhat_xgb_log)
    results.append(("XGB(logY)", xgb_mae, xgb_rmse))

    xgb_best.save_model(str(MODELS / "week6_xgb_revenue.json"))
    with open(MODELS / "week6_xgb_revenue_params.json", "w") as f:
        json.dump(xgb_search.best_params_, f, indent=2)

    # XGB gain importances (top 20)
    try:
        booster = xgb_best.get_booster()
        gain = pd.Series(booster.get_score(importance_type="gain"))
        if all(k.startswith("f") for k in gain.index):
            mapper = {f"f{i}": col for i, col in enumerate(X.columns)}
            gain.index = [mapper.get(k, k) for k in gain.index]
        gain = gain.sort_values().tail(20)
        plt.figure(figsize=(7,6)); plt.barh(gain.index, gain.values)
        plt.title("Revenue — XGB Feature Importances (Top 20, gain)"); plt.xlabel("Gain")
        plt.tight_layout(); plt.savefig(REPORTS / "week6_revenue_xgb_importances.png", dpi=150); plt.close()
    except Exception:
        pass

    if xgb_mae < rf_mae:
        best_model, best_name = xgb_best, "XGB"

#%% 6) PDPs (Ride Distance, Hour)
for feat, outname in [("Ride Distance", "week6_revenue_pdp_distance.png"),
                      ("Hour",          "week6_revenue_pdp_hour.png")]:
    if feat in X.columns:
        fig, ax = plt.subplots(figsize=(7,4))
        PartialDependenceDisplay.from_estimator(best_model, Xtr, [feat], ax=ax, grid_resolution=24)
        ax.set_title(f"Partial Dependence — {feat} ({best_name})")
        plt.tight_layout(); plt.savefig(REPORTS / outname, dpi=150); plt.close()

#%% 7) SHAP (RF always; XGB only if better)
if HAS_SHAP:
    try:
        def _shap_for_tree(model, X_train_full, X_eval, tag):
            rng = np.random.default_rng(42)
            bg = X_train_full.iloc[rng.choice(len(X_train_full), size=min(2000, len(X_train_full)), replace=False)]
            xp = X_eval.iloc[rng.choice(len(X_eval), size=min(5000, len(X_eval)), replace=False)]
            explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
            sv = explainer(xp)
            shap.plots.bar(sv, max_display=15, show=False); plt.tight_layout()
            plt.savefig(REPORTS / f"week6_revenue_shap_bar_{tag}.png", dpi=150); plt.close()
            shap.plots.beeswarm(sv, max_display=15, show=False); plt.tight_layout()
            plt.savefig(REPORTS / f"week6_revenue_shap_beeswarm_{tag}.png", dpi=150); plt.close()

        _shap_for_tree(rf_best, Xtr, Xte, tag="rf")
        if 'xgb_best' in globals() and results[-1][0].startswith("XGB") and results[-1][1] <= rf_mae:
            _shap_for_tree(xgb_best, Xtr, Xte, tag="xgb")
    except Exception as e:
        print("SHAP step skipped:", e)

#%% 8) Segmentation & Clustering (scaled)
# Vehicle type averages (if OHE present)
veh_cols = [c for c in X.columns if c.startswith("Vehicle Type_")]
if veh_cols:
    def _veh(c): return c.replace("Vehicle Type_","")
    seg = {}
    for c in veh_cols:
        mask = (rev.index[:len(X)] >= 0) & (X[c] == 1)
        seg[_veh(c)] = rev.loc[mask, "Booking Value"].mean()
    ser = pd.Series(seg).sort_values(ascending=False)
    plt.figure(figsize=(9,5)); plt.bar(ser.index, ser.values)
    plt.title("Average Booking Value by Vehicle Type"); plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Booking Value"); plt.tight_layout()
    plt.savefig(REPORTS / "week6_revenue_segment_vehicle.png", dpi=150); plt.close()

# Hour averages
if "Hour" in rev.columns:
    hr = rev.groupby("Hour")["Booking Value"].mean().sort_index()
    plt.figure(); plt.plot(hr.index, hr.values, marker="o")
    plt.title("Average Booking Value by Hour"); plt.xlabel("Hour"); plt.ylabel("Mean Booking Value")
    plt.tight_layout(); plt.savefig(REPORTS / "week6_revenue_segment_hour.png", dpi=150); plt.close()

# KMeans on scaled core features
core_for_k = [c for c in ["Ride Distance","Avg VTAT","Avg CTAT","Hour"] if c in X.columns]
if len(core_for_k) >= 2:
    Xk = X[core_for_k].copy()
    scaler = StandardScaler()
    Xk_scaled = scaler.fit_transform(Xk)
    km = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(Xk_scaled)
    f1, f2 = core_for_k[:2]
    plt.figure(figsize=(7,5))
    plt.scatter(Xk[f1], Xk[f2], c=labels, s=8)
    plt.title(f"KMeans clusters (k=4) — {f1} vs {f2}")
    plt.xlabel(f1); plt.ylabel(f2); plt.tight_layout()
    plt.savefig(REPORTS / "week6_revenue_clusters_scatter.png", dpi=150); plt.close()

#%% 9) Save metrics + summary
metrics_df = pd.DataFrame([{"Model": n, "MAE": mae, "RMSE": rmse} for n, mae, rmse in results]).sort_values("MAE")
metrics_csv = DATA_PROC / "week6_revenue_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False)

summary = dedent(f"""
# Week 6 — Advanced Revenue Models (Summary, no leakage)

**Input:** `{rev_path.name}`
**Train rows:** {len(Xtr):,} | **Test rows:** {len(Xte):,}
**Features (no leakage):** {len(X.columns):,}

**Results (20% holdout, original currency)**
{chr(10).join([f"- {r['Model']}: MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}" for _, r in metrics_df.iterrows()])}

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
""").strip()

with open(REPORTS / "week6_revenue_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote:", REPORTS / "week6_revenue_summary.md")
print("Wrote metrics:", metrics_csv)
print("Week 6 revenue modeling complete.")