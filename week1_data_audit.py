#%% [markdown]
# ============================================
# Week 1 — Data Intake & Quality Audit
# File: week1_data_audit.py  (run in VS Code/Jupyter)
# ============================================

#%% 0) Imports & Paths
import os
from pathlib import Path
import sys
import json
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent
from datetime import datetime

# Project paths
ROOT = Path(".").resolve()
DATA_PATH = ROOT / "ncr_ride_bookings.csv"           # <-- change if needed
OUT_DIR_REPORTS = ROOT / "reports"
OUT_DIR_DATA = ROOT / "data" / "interim"
OUT_DIRS = [OUT_DIR_REPORTS, OUT_DIR_DATA]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

#%% 1) (Optional) Environment lock helpers — run once, keep for reproducibility
print("Python:", sys.version)
print("Platform:", platform.platform())

# To lock dependencies (choose ONE):
# !pip freeze > requirements.txt
# or, with conda:
# !conda env export --no-builds > environment.yml

#%% 2) Load CSV & basic shape checks
assert DATA_PATH.exists(), f"CSV not found at: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Expected columns (based on your dataset profile)
expected_cols = [
    "Date","Time","Booking ID","Booking Status","Customer ID",
    "Vehicle Type","Pickup Location","Drop Location",
    "Avg VTAT","Avg CTAT",
    "Cancelled Rides by Customer","Reason for cancelling by Customer",
    "Cancelled Rides by Driver","Driver Cancellation Reason",
    "Incomplete Rides","Incomplete Rides Reason",
    "Booking Value","Ride Distance","Driver Ratings","Customer Rating",
    "Payment Method"
]
missing_expected = [c for c in expected_cols if c not in df.columns]
extra_cols = [c for c in df.columns if c not in expected_cols]
print("Missing expected cols:", missing_expected)
print("Unexpected extra cols:", extra_cols)

#%% 3) Coerce obvious types gently (no destructive changes)
# Keep IDs as strings
for col in ["Booking ID", "Customer ID"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False)

# Preview raw types
print("\nDtypes BEFORE parsing:")
print(df.dtypes)

#%% 4) Parse Date/Time and engineer Hour/Weekday/Month (non-destructive)
# Safe datetime parsing
if "Date" in df.columns:
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    df["Date_parsed"] = pd.NaT

if "Time" in df.columns:
    # If Time is HH:MM:SS string; create Hour from it
    hour_series = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
else:
    hour_series = pd.Series([np.nan]*len(df))

df["Hour"] = hour_series
df["Weekday"] = df["Date_parsed"].dt.day_name()
df["Month"] = df["Date_parsed"].dt.to_period("M").astype(str)

# Check parsing quality
n_date_nat = df["Date_parsed"].isna().sum()
n_hour_nan = df["Hour"].isna().sum()
print(f"\nDatetime parsing — NaT in Date_parsed: {n_date_nat:,} | NaN in Hour: {n_hour_nan:,}")

#%% 5) Key integrity checks
issues = []

# 5a) Booking ID uniqueness
if "Booking ID" in df.columns:
    n_unique_bookings = df["Booking ID"].nunique(dropna=True)
    dup_bookings = df.shape[0] - n_unique_bookings
    print(f"Unique Booking IDs: {n_unique_bookings:,} | Possible dup rows by Booking ID: {dup_bookings:,}")
    if dup_bookings > 0:
        issues.append(f"- {dup_bookings:,} possible duplicate booking rows by 'Booking ID'.")

# 5b) Full-row duplicates
n_dupe_rows = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dupe_rows:,}")
if n_dupe_rows > 0:
    issues.append(f"- {n_dupe_rows:,} exact duplicate rows detected.")

#%% 6) Missing values audit
missing_summary = df[expected_cols].isna().sum() if not missing_expected else df.isna().sum()
print("\nMissing Values Summary:")
print(missing_summary.sort_values(ascending=False))

#%% 7) Numeric columns profile (ranges & quantiles)
numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
# Include known numeric features if typed as object but numeric values
for c in ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating",
          "Cancelled Rides by Customer","Cancelled Rides by Driver","Incomplete Rides"]:
    if c in df.columns and c not in numeric_cols:
        # attempt to coerce a copy to numeric for profiling (don't overwrite main yet)
        pass

def numeric_profile(frame, columns):
    prof = {}
    for c in columns:
        s = pd.to_numeric(frame[c], errors="coerce")
        prof[c] = {
            "non_null": int(s.notna().sum()),
            "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
            "p25": float(np.nanpercentile(s, 25)) if s.notna().any() else np.nan,
            "median": float(np.nanmedian(s)) if s.notna().any() else np.nan,
            "p75": float(np.nanpercentile(s, 75)) if s.notna().any() else np.nan,
            "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
            "n_missing": int(s.isna().sum())
        }
    return pd.DataFrame(prof).T

num_prof = numeric_profile(df, [c for c in expected_cols if c in df.columns])
print("\nNumeric Profile (selected):")
print(num_prof)

# Flag impossible values (non-negative checks)
flag_msgs = []
for col in ["Booking Value","Ride Distance","Avg VTAT","Avg CTAT","Driver Ratings","Customer Rating"]:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n_neg = (s < 0).sum()
        if n_neg > 0:
            flag_msgs.append(f"- {col}: {n_neg} negative values found (should be non-negative).")
        if col in ["Driver Ratings","Customer Rating"]:
            n_out = ((s < 0) | (s > 5)).sum()
            if n_out > 0:
                flag_msgs.append(f"- {col}: {n_out} values outside 0–5 range.")
        if col == "Ride Distance":
            n_out = (s > 200).sum()
            if n_out > 0:
                flag_msgs.append(f"- {col}: {n_out} values > 200 km (verify outliers).")
        if col == "Booking Value":
            n_out = (s > 10000).sum()
            if n_out > 0:
                flag_msgs.append(f"- {col}: {n_out} values > 10,000 (verify outliers).")

#%% 8) High-cardinality categorical checks
cat_cols = [c for c in expected_cols if c in df.columns and df[c].dtype == "object"]
cat_card = pd.Series({c: df[c].nunique(dropna=True) for c in cat_cols}).sort_values(ascending=False)
print("\nCategorical Cardinality (unique non-null values):")
print(cat_card)

#%% 9) Save augmented data (with time columns only; no cleaning yet)
out_csv = OUT_DIR_DATA / "ncr_ride_bookings_week1_with_timecols.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved augmented CSV (with Date_parsed/Hour/Weekday/Month) → {out_csv}")

#%% 10) Compose the Data Audit Report (Markdown) + Checklist
report_lines = []

report_lines.append("# Week 1 — Data Intake & Quality Audit\n")
report_lines.append(f"- File: `{DATA_PATH.name}`  \n- Rows × Cols: **{df.shape[0]:,} × {df.shape[1]}**\n")
if missing_expected:
    report_lines.append(f"- ⚠️ Missing expected columns: {missing_expected}\n")
if extra_cols:
    report_lines.append(f"- ℹ️ Extra columns present: {extra_cols}\n")

report_lines.append("## Schema (dtypes before cleaning)\n")
report_lines.append("```\n" + str(df.dtypes) + "\n```\n")

report_lines.append("## Missing Values Summary\n")
report_lines.append("```\n" + str(missing_summary.sort_values(ascending=False)) + "\n```\n")

report_lines.append("## Numeric Profile (selected)\n")
report_lines.append(num_prof.to_markdown())

report_lines.append("\n## Categorical Cardinality\n")
report_lines.append(cat_card.to_frame("n_unique").to_markdown())

report_lines.append("\n## Key Integrity Checks\n")
report_lines.append(f"- Exact duplicate rows: **{n_dupe_rows:,}**")
if "Booking ID" in df.columns:
    report_lines.append(f"- Unique Booking IDs: **{n_unique_bookings:,}**  \n- Possible duplicates by ID: **{dup_bookings:,}**")
report_lines.append(f"- Date parsing NaT count: **{n_date_nat:,}**; Hour NaN count: **{n_hour_nan:,}**")

if flag_msgs:
    report_lines.append("\n## Potential Outlier/Range Flags\n" + "\n".join(flag_msgs))

if issues:
    report_lines.append("\n## Other Issues\n" + "\n".join(issues))

audit_md_path = OUT_DIR_REPORTS / "week1_data_audit.md"
with open(audit_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nWrote data audit report → {audit_md_path}")

# Checklist (explicit actions for Week 2)
checklist = dedent("""
# Week 1 → Week 2 Checklist
- [ ] Impute event columns (0 / "Not Applicable"); finalize numeric imputation plan (KNN vs median).
- [ ] Decide treatment for extreme values in Ride Distance and Booking Value (cap vs remove).
- [ ] Confirm handling of rating bounds (clip to [0,5]).
- [ ] Verify Booking ID duplicates; deduplicate if needed.
- [ ] Confirm Date/Time parsing choices (timezone, if applicable).
- [ ] Lock environment (requirements.txt or environment.yml) and commit to repo.
""").strip()

checklist_path = OUT_DIR_REPORTS / "week1_checklist.md"
with open(checklist_path, "w", encoding="utf-8") as f:
    f.write(checklist)
print(f"Wrote checklist → {checklist_path}")
#%%