#%% [markdown]
# ============================================
# Week 3 — Feature Engineering (clean copy)
#   - Find latest cleaned dataset
#   - Parse times & deduplicate by Booking ID
#   - Build revenue row-level & daily demand frames
#   - Save engineered datasets + summary report
# ============================================

#%% 1) Paths, imports, and cleaned-file discovery
from pathlib import Path
import pandas as pd
import numpy as np
from textwrap import dedent

# Resolve project root (works whether run as script or notebook)
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

OUT_DIR_PROC = ROOT / "data" / "processed"
OUT_DIR_REP  = ROOT / "reports"
for d in (OUT_DIR_PROC, OUT_DIR_REP):
    d.mkdir(parents=True, exist_ok=True)

def find_cleaned_file() -> Path:
    """Find the newest 'cleaned' dataset in processed/ or interim/."""
    candidates = []
    for d in [OUT_DIR_PROC, ROOT / "data" / "interim"]:
        if d.exists():
            candidates += list(d.glob("ncr_ride_bookings_cleaned*.parquet"))
            candidates += list(d.glob("ncr_ride_bookings_cleaned*.csv"))
    if not candidates:
        raise FileNotFoundError("No cleaned dataset found in data/processed or data/interim.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

IN_CLEAN = find_cleaned_file()
print(f"Using cleaned file → {IN_CLEAN}")

# Load (CSV or Parquet)
if IN_CLEAN.suffix.lower() == ".parquet":
    df = pd.read_parquet(IN_CLEAN)
else:
    df = pd.read_csv(IN_CLEAN)

print("Loaded shape:", df.shape)

#%% 2) Parse times, add calendar cols, and deduplicate by Booking ID
# Ensure essential time fields exist
df["Date_parsed"] = pd.to_datetime(df.get("Date_parsed", df.get("Date")), errors="coerce")
if "Hour" not in df.columns and "Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour

df["Weekday"] = df["Date_parsed"].dt.day_name()
df["Weekday_num"] = df["Date_parsed"].dt.weekday  # Monday=0
df["Month"] = df["Date_parsed"].dt.month

# Drop rows with invalid dates
bad_dates = df["Date_parsed"].isna().sum()
if bad_dates:
    print(f"Warning: dropping {bad_dates} rows with invalid Date_parsed")
    df = df[df["Date_parsed"].notna()].copy()

# De-duplicate by Booking ID (keep earliest by date)
if "Booking ID" in df.columns:
    before = len(df)
    df = df.sort_values("Date_parsed", kind="stable").drop_duplicates(subset=["Booking ID"], keep="first")
    after = len(df)
    print(f"De-duplicated by Booking ID: removed {before - after:,} rows; new shape {df.shape}")

print("Post time/dup shape:", df.shape)

#%% 3) Build modeling frames (revenue row-level & daily demand)
# ---- Revenue (row-level) ----
target_col = "Booking Value"
num_cols = [c for c in ["Ride Distance", "Avg VTAT", "Avg CTAT", "Hour"] if c in df.columns]
cat_cols = [c for c in ["Vehicle Type", "Weekday"] if c in df.columns]

# Keep rows with target
rev = df.dropna(subset=[target_col]).copy()

# One-hot encode categoricals (pandas)
if cat_cols:
    rev = pd.get_dummies(rev, columns=cat_cols, prefix=cat_cols, dummy_na=False)

# Feature columns = numeric + one-hot Vehicle Type/Weekday
ohe_cols = [c for c in rev.columns if c.startswith("Vehicle Type_") or c.startswith("Weekday_")]
feature_cols = num_cols + ohe_cols

# Modeling subset (features + target + date for plotting/temporal analyses)
keep_cols = [c for c in (feature_cols + [target_col, "Date_parsed"]) if c in rev.columns]
rev_model = rev[keep_cols].copy()

# ---- Demand (daily aggregate) ----
daily = (
    df.groupby("Date_parsed")
      .agg(rides=("Booking ID", "count"),
           revenue=("Booking Value", "sum"))
      .sort_index()
      .reset_index()
)

# Calendar features on daily
daily["Weekday_num"] = daily["Date_parsed"].dt.weekday
daily["Month"] = daily["Date_parsed"].dt.month

# Rides lags & rolling means
for lag in (1, 7, 14):
    daily[f"lag_{lag}"] = daily["rides"].shift(lag)
for w in (7, 14, 28):
    daily[f"roll_{w}"] = daily["rides"].rolling(window=w, min_periods=1).mean()

# (Optional) revenue lags/rollings (if forecasting revenue too)
for lag in (1, 7, 14):
    daily[f"rev_lag_{lag}"] = daily["revenue"].shift(lag)
for w in (7, 14, 28):
    daily[f"rev_roll_{w}"] = daily["revenue"].rolling(window=w, min_periods=1).mean()

# Two variants: keep early NA (soft) vs drop NA (hard)
daily_soft = daily.copy()
daily_hard = daily.dropna().copy()

# Quick guards so save/summary won’t crash
if rev_model.empty:
    print("⚠️ rev_model is empty — check that Booking Value exists and keep_cols are valid.")
if daily_hard.empty:
    print("⚠️ daily_hard is empty — consider using daily_soft for early periods.")

#%% 4) Save engineered datasets (Parquet→CSV fallback) + summary
def _safe_save_parquet_or_csv(df_: pd.DataFrame, pq_path: Path, csv_path: Path) -> Path:
    """Try Parquet (smaller/faster). Fallback to CSV if parquet libs missing."""
    try:
        df_.to_parquet(pq_path, index=False)  # needs pyarrow/fastparquet
        return pq_path
    except Exception:
        df_.to_csv(csv_path, index=False)
        return csv_path

# --- Save datasets ---
rev_full_path  = _safe_save_parquet_or_csv(
    rev,
    OUT_DIR_PROC / "ncr_rowlevel_revenue_full.parquet",
    OUT_DIR_PROC / "ncr_rowlevel_revenue_full.csv"
)
rev_model_path = _safe_save_parquet_or_csv(
    rev_model,
    OUT_DIR_PROC / "ncr_rowlevel_revenue_model.parquet",
    OUT_DIR_PROC / "ncr_rowlevel_revenue_model.csv"
)
daily_soft_path = _safe_save_parquet_or_csv(
    daily_soft,
    OUT_DIR_PROC / "ncr_daily_demand_soft.parquet",
    OUT_DIR_PROC / "ncr_daily_demand_soft.csv"
)
daily_hard_path = _safe_save_parquet_or_csv(
    daily_hard,
    OUT_DIR_PROC / "ncr_daily_demand_hard.parquet",
    OUT_DIR_PROC / "ncr_daily_demand_hard.csv"
)

print("✅ Saved engineered datasets:")
print(" - Revenue (full): ",  rev_full_path)
print(" - Revenue (model):",  rev_model_path)
print(" - Daily (soft):   ",  daily_soft_path)
print(" - Daily (hard):   ",  daily_hard_path)

# --- Assertions to catch schema drift ---
required_rev = set(num_cols + ["Date_parsed", target_col])
missing_rev = required_rev - set(rev_model.columns)
assert not missing_rev, f"rev_model missing columns: {missing_rev}"

required_daily = {"Date_parsed","rides","revenue","lag_1","lag_7","lag_14","roll_7","roll_14","roll_28"}
missing_daily = required_daily - set(daily_soft.columns)
assert not missing_daily, f"daily_* missing columns: {missing_daily}"

# --- Summary report (+ previews) ---
def _to_markdown_safe(df_: pd.DataFrame, max_rows: int = 8) -> str:
    preview = df_.head(max_rows).copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return "```\n" + preview.to_string(index=False) + "\n```"

summary = dedent(f"""
# Week 3 — Feature Engineering Summary

**Inputs**
- Cleaned file: `{IN_CLEAN.name}`

**Row-level revenue datasets**
- Full (all columns + one-hot): `{rev_full_path.name}`  (rows: {len(rev):,})
- Modeling subset (features + target): `{rev_model_path.name}`  (rows: {len(rev_model):,})
- Numeric features: {num_cols}
- One-hot columns (Vehicle Type / Weekday): {len([c for c in feature_cols if c not in num_cols])}

**Daily demand datasets**
- Soft (keeps early NA lags): `{daily_soft_path.name}` (rows: {len(daily_soft):,})
- Hard (drops NA lags/rolls): `{daily_hard_path.name}` (rows: {len(daily_hard):,})
- Added features: lag_1, lag_7, lag_14, roll_7, roll_14, roll_28 (+ revenue lags/rolls)

**Notes**
- Parsed Date/Time and added Hour/Weekday/Month.
- De-duplicated by `Booking ID` (kept earliest by date).
- Next (Week 4): Baselines
  - Demand: ARIMA on `rides` + RF/XGB using lags/rolls (compare MAE/RMSE).
  - Revenue: Linear Regression (log(Booking Value) recommended) vs RF/XGB + importances.

---

## Preview — Revenue Modeling Frame (`rev_model`)
{_to_markdown_safe(rev_model[[*num_cols, target_col, "Date_parsed"]] if set([*num_cols, target_col, "Date_parsed"]).issubset(rev_model.columns) else rev_model)}

## Preview — Daily Demand Frame (`daily_hard`)
{_to_markdown_safe(daily_hard)}
""").strip()

rep_path = OUT_DIR_REP / "week3_feature_engineering.md"
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(summary)

print("\nWrote report →", rep_path)
print("✅ Week 3 complete: engineered datasets are ready for modeling.")
#%%