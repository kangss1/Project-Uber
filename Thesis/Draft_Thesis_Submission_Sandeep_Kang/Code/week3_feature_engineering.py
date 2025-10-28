#%% [markdown]
# ============================================
# Week 3 — Feature Engineering (updated after Week 2)
# File: week3_feature_engineering.py
# Depends on: week2_cleaning_eda.py -> data/processed/ncr_ride_bookings_cleaned.*
# ============================================
#
# What changed (why)
# - Aligns with Week 2 results (clean, deduped, stable time axis).
# - REVENUE (row-level): keeps baseline features and adds:
#     * price_per_km  = Booking Value / Ride Distance  (safe divide)
#     * speed_proxy   = Ride Distance / Avg CTAT       (safe divide; "km per time-unit")
#     * is_weekend, rush-hour flags (AM 7–10, PM 17–20), hour bins
#     * one-hot Vehicle Type + Weekday
# - DEMAND (daily): preserves lag_1/7/14 and roll_7/14/28 and adds:
#     * roll_std_7, roll_std_14  (volatility)
#     * weekly seasonality (sin/cos with period=7)
#     * contiguous daily index (asfreq('D')) to avoid holes
# - Strong guards: safe parsing, dtype coercions, and schema checks.
#
# Outputs
# - data/processed/ncr_rowlevel_revenue_full.(parquet|csv)
# - data/processed/ncr_rowlevel_revenue_model.(parquet|csv)
# - data/processed/ncr_daily_demand_soft.(parquet|csv)
# - data/processed/ncr_daily_demand_hard.(parquet|csv)
# - reports/week3_feature_engineering.md

#%% 1) Imports & paths
from pathlib import Path
import numpy as np
import pandas as pd
from textwrap import dedent

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

DATA_PROC = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
for d in (DATA_PROC, DATA_INTERIM, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

def _latest_cleaned() -> Path:
    cands = []
    for base in (DATA_PROC, DATA_INTERIM):
        if base.exists():
            cands += list(base.glob("ncr_ride_bookings_cleaned*.parquet"))
            cands += list(base.glob("ncr_ride_bookings_cleaned*.csv"))
    if not cands:
        raise FileNotFoundError("No cleaned dataset found in data/processed or data/interim.")
    return max(cands, key=lambda p: p.stat().st_mtime)

IN_CLEAN = _latest_cleaned()
print(f"Using cleaned file → {IN_CLEAN}")

#%% 2) Load + time parsing + minimal sanity
if IN_CLEAN.suffix.lower() == ".parquet":
    df = pd.read_parquet(IN_CLEAN)
else:
    df = pd.read_csv(IN_CLEAN)

print("Loaded shape:", df.shape)

# Safe time fields
df["Date_parsed"] = pd.to_datetime(df.get("Date_parsed", df.get("Date")), errors="coerce")
if "Hour" not in df.columns and "Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour

bad_dates = int(df["Date_parsed"].isna().sum())
if bad_dates:
    print(f"Warning: dropping {bad_dates} rows with invalid Date_parsed")
    df = df[df["Date_parsed"].notna()].copy()

# Lightweight re-dedup by Booking ID (defensive; Week 2 already did this)
if "Booking ID" in df.columns:
    before = len(df)
    df = df.sort_values(["Date_parsed"], kind="stable").drop_duplicates(subset=["Booking ID"], keep="first")
    print(f"Re-deduped by Booking ID: removed {before - len(df):,} rows; shape={df.shape}")

# Calendar helpers
df["Weekday"] = df["Date_parsed"].dt.day_name()
df["Weekday_num"] = df["Date_parsed"].dt.weekday  # Mon=0
df["Month"] = df["Date_parsed"].dt.month
df["is_weekend"] = df["Weekday_num"].isin([5, 6]).astype(int)

# Rush-hour flags (Delhi NCR typical peaks)
df["rush_am"] = df["Hour"].between(7, 10, inclusive="both").astype(int)
df["rush_pm"] = df["Hour"].between(17, 20, inclusive="both").astype(int)

#%% 3) REVENUE (row-level) frame
target_col = "Booking Value"

# Safe numeric coercions
for c in ["Ride Distance", "Avg VTAT", "Avg CTAT", target_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Engineered numerics (safe divide)
def _safe_div(num, den):
    return np.where((pd.notna(num)) & (pd.notna(den)) & (den != 0), num / den, np.nan)

if {"Ride Distance", target_col}.issubset(df.columns):
    df["price_per_km"] = _safe_div(df[target_col], df["Ride Distance"])

if {"Ride Distance", "Avg CTAT"}.issubset(df.columns):
    # If CTAT is in minutes, this is "km per minute"; unit is a proxy only
    df["speed_proxy"] = _safe_div(df["Ride Distance"], df["Avg CTAT"])

num_cols = [c for c in ["Ride Distance", "Avg VTAT", "Avg CTAT", "Hour",
                        "price_per_km", "speed_proxy"] if c in df.columns]
cat_cols = [c for c in ["Vehicle Type", "Weekday"] if c in df.columns]

rev_full = df.copy()
if cat_cols:
    rev_full = pd.get_dummies(rev_full, columns=cat_cols, prefix=cat_cols, dummy_na=False)

ohe_cols = [c for c in rev_full.columns if c.startswith("Vehicle Type_") or c.startswith("Weekday_")]
feature_cols = num_cols + ["is_weekend", "rush_am", "rush_pm"] + ohe_cols

keep_cols = [c for c in (feature_cols + [target_col, "Date_parsed"]) if c in rev_full.columns]
rev_model = rev_full.dropna(subset=[target_col]).loc[:, keep_cols].copy()

if rev_model.empty:
    print("rev_model is empty — check Booking Value and feature columns.")

#%% 4) DEMAND (daily) frame
daily = (
    df.groupby("Date_parsed")
      .agg(rides=("Booking ID", "count"),
           revenue=(target_col, "sum"))
      .sort_index()
)

# ensure contiguous daily index (fills missing calendar days with 0 rides/revenue)
daily = (
    daily.asfreq("D")
         .fillna({"rides": 0, "revenue": 0})
         .reset_index()
)

# Calendar on daily
daily["Weekday_num"] = daily["Date_parsed"].dt.weekday
daily["Month"] = daily["Date_parsed"].dt.month

# Lags & rolling means
for lag in (1, 7, 14):
    daily[f"lag_{lag}"] = daily["rides"].shift(lag)
for w in (7, 14, 28):
    daily[f"roll_{w}"] = daily["rides"].rolling(window=w, min_periods=1).mean()

# Rolling volatility (useful given near-stationary mean but noisy day-to-day)
for w in (7, 14):
    daily[f"roll_std_{w}"] = daily["rides"].rolling(window=w, min_periods=2).std()

# Light weekly seasonality with Fourier terms (period = 7 days)
two_pi = 2 * np.pi
t = np.arange(len(daily))
daily["sin_7"] = np.sin(two_pi * t / 7.0)
daily["cos_7"] = np.cos(two_pi * t / 7.0)

# Optional: revenue lags/rolls (helpful for revenue forecasting later)
for lag in (1, 7, 14):
    daily[f"rev_lag_{lag}"] = daily["revenue"].shift(lag)
for w in (7, 14, 28):
    daily[f"rev_roll_{w}"] = daily["revenue"].rolling(window=w, min_periods=1).mean()

daily_soft = daily.copy()
daily_hard = daily.dropna().copy()
if daily_hard.empty:
    print("daily_hard is empty (early NA lags). Use daily_soft for models that tolerate NAs or trim manually.")

#%% 5) Save datasets (Parquet→CSV fallback) + schema guards
def _safe_save(df_: pd.DataFrame, pq: Path, csv: Path) -> Path:
    try:
        df_.to_parquet(pq, index=False)
        return pq
    except Exception:
        df_.to_csv(csv, index=False)
        return csv

rev_full_path  = _safe_save(rev_full,  DATA_PROC / "ncr_rowlevel_revenue_full.parquet",  DATA_PROC / "ncr_rowlevel_revenue_full.csv")
rev_model_path = _safe_save(rev_model, DATA_PROC / "ncr_rowlevel_revenue_model.parquet", DATA_PROC / "ncr_rowlevel_revenue_model.csv")
daily_soft_path = _safe_save(daily_soft, DATA_PROC / "ncr_daily_demand_soft.parquet", DATA_PROC / "ncr_daily_demand_soft.csv")
daily_hard_path = _safe_save(daily_hard, DATA_PROC / "ncr_daily_demand_hard.parquet", DATA_PROC / "ncr_daily_demand_hard.csv")

print("Saved engineered datasets:")
print(" - Revenue (full): ",  rev_full_path)
print(" - Revenue (model):",  rev_model_path)
print(" - Daily (soft):   ",  daily_soft_path)
print(" - Daily (hard):   ",  daily_hard_path)

# Required columns (early failure if something drifted)
need_rev = set(["Date_parsed", target_col, "is_weekend", "rush_am", "rush_pm"] + num_cols)
missing_rev = need_rev - set(rev_model.columns)
assert not missing_rev, f"rev_model missing: {missing_rev}"

need_daily = {"Date_parsed","rides","revenue","lag_1","lag_7","lag_14",
              "roll_7","roll_14","roll_28","roll_std_7","roll_std_14","sin_7","cos_7"}
missing_daily = need_daily - set(daily_soft.columns)
assert not missing_daily, f"daily_* missing: {missing_daily}"

#%% 6) Summary report (Markdown)
def _md(df_: pd.DataFrame, k=8):
    p = df_.head(k).copy()
    try:
        return p.to_markdown(index=False)
    except Exception:
        return "```\n" + p.to_string(index=False) + "\n```"

summary = dedent(f"""
# Week 3 — Feature Engineering (Updated)

**Input**: `{IN_CLEAN.name}`  
**Rows (after time guards & re-dedup)**: {len(df):,}

## Revenue (row-level)
- Saved: `{rev_full_path.name}`, `{rev_model_path.name}` (rows: {len(rev_model):,})
- Numeric features: {num_cols}
- Added flags: `is_weekend`, `rush_am`, `rush_pm`
- One-hot: Vehicle Type / Weekday → {len([c for c in rev_model.columns if c.startswith('Vehicle Type_') or c.startswith('Weekday_')])} columns
- Engineered: `price_per_km`, `speed_proxy` (safe divide)

**Preview (`rev_model`)**
{_md(rev_model[[c for c in ["Ride Distance","Avg VTAT","Avg CTAT","Hour","price_per_km","speed_proxy",target_col,"Date_parsed"] if c in rev_model.columns]])}

## Demand (daily)
- Saved: `{daily_soft_path.name}` (soft), `{daily_hard_path.name}` (hard)
- Added: lags (1/7/14), rolls (7/14/28), **roll_std_7/14**, **sin_7/cos_7** weekly seasonality
- Contiguous daily index via `asfreq('D')`

**Preview (`daily_hard`)**
{_md(daily_hard)}

## Notes
- Week 4 can now use:
  - Demand: RF/XGB with lags/rolls **and** volatility + seasonal terms.
  - Revenue: Linear (log target) vs RF/XGB; importances expected to emphasize distance, VTAT/CTAT, hour/peaks.
""").strip()

with open(REPORTS / "week3_feature_engineering.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote report →", REPORTS / "week3_feature_engineering.md")
print("Week 3 complete (enhanced features ready).")