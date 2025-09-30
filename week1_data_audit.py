#%% [markdown]
# ============================================
# Week 1 — Data Intake & Quality Audit
# File: week1_data_audit.py  (run in VS Code/Jupyter)
# ============================================
#
#  Purpose
# Below is a clear plan (what to do + why), followed by a ready-to-run,
# cell-formatted Python script. It fully covers your Week-1 milestone:
# environment reproducibility notes, schema validation, missing/duplicate/
# outlier audits, Date/Time parsing + enrichment, and export of both a clean
# Data Audit Report (Markdown) and a to-fix checklist.
#
# ─────────────────────────────────────────────────────────────────────────────
# What we'll do (direction)
# 1) Environment & reproducibility
#    - Create a project environment; pin library versions so results are repeatable.
#    - Save requirements.txt (or environment.yml if using conda).
# 2) Load & validate schema
#    - Confirm column presence, dtypes, and row count against expectations.
#    - Ensure ID fields are string-typed and check Booking ID uniqueness.
# 3) Audit data quality
#    - Summarize missing values by column (counts + %).
#    - Detect duplicates: full-row and key-based (Booking ID).
#    - Profile numeric ranges; flag impossible values (e.g., negative distance/value).
#    - Review categorical cardinality (raw vs normalized) to spot encoding/leakage risks.
# 4) Datetime parsing & enrichment
#    - Parse Date and Time; derive Hour, Weekday, Month; build a unified timestamp.
#    - Verify NaT/NaN counts to confirm parsing success.
# 5) Light outlier screening
#    - Show numeric quantiles + IQR fences.
#    - Flag domain-cap candidates (e.g., Ride Distance > 200 km, Booking Value > 10,000)
#      without dropping anything yet—just log them.
# 6) Deliverables
#    - /reports/week1_data_audit.md — human-readable audit summary.
#    - /reports/week1_checklist.md — explicit fix actions for Week-2.
#    - /reports/week1_data_audit_summary.json — light machine-readable summary.
#    - /data/interim/ncr_ride_bookings_week1_with_timecols.csv — augmented dataset
#      with time features added (no cleaning yet).
# ─────────────────────────────────────────────────────────────────────────────

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

# ---- Config (kept simple & in-file) ----
CONFIG = {
    "expected_cols": [
        "Date","Time","Booking ID","Booking Status","Customer ID",
        "Vehicle Type","Pickup Location","Drop Location",
        "Avg VTAT","Avg CTAT",
        "Cancelled Rides by Customer","Reason for cancelling by Customer",
        "Cancelled Rides by Driver","Driver Cancellation Reason",
        "Incomplete Rides","Incomplete Rides Reason",
        "Booking Value","Ride Distance","Driver Ratings","Customer Rating",
        "Payment Method"
    ],
    "id_cols_str": ["Booking ID", "Customer ID"],
    "rating_bounds": {"Driver Ratings": (0,5), "Customer Rating": (0,5)},
    "hard_caps": {"Ride Distance": 200.0, "Booking Value": 10000.0},
    "na_tokens": ["NA","N/A","-","","null","None"]
}

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
df = pd.read_csv(DATA_PATH, na_values=CONFIG["na_tokens"])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Expected columns (from config)
expected_cols = CONFIG["expected_cols"]
missing_expected = [c for c in expected_cols if c not in df.columns]
extra_cols = [c for c in df.columns if c not in expected_cols]
print("Missing expected cols:", missing_expected)
print("Unexpected extra cols:", extra_cols)

#%% 3) Coerce obvious types gently (no destructive changes)
# Keep IDs as strings
for col in CONFIG["id_cols_str"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False)

# Preview raw types
print("\nDtypes BEFORE parsing:")
print(df.dtypes)

#%% 4) Parse Date/Time and engineer Hour/Weekday/Month + unified timestamp (non-destructive)
# Safe datetime parsing
if "Date" in df.columns:
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    df["Date_parsed"] = pd.NaT

if "Time" in df.columns:
    hour_series = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
else:
    hour_series = pd.Series([np.nan]*len(df), index=df.index)

df["Hour"] = hour_series
df["Weekday"] = df["Date_parsed"].dt.day_name()
df["Month"] = df["Date_parsed"].dt.to_period("M").astype(str)

# Unified timestamp (helps Week-2 time-series resampling)
if "Date" in df.columns and "Time" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )
else:
    df["timestamp"] = pd.NaT

# Check parsing quality
n_date_nat = df["Date_parsed"].isna().sum()
n_hour_nan = df["Hour"].isna().sum()
n_ts_nat = df["timestamp"].isna().sum()
print(f"\nDatetime parsing — NaT in Date_parsed: {n_date_nat:,} | NaN in Hour: {n_hour_nan:,} | NaT in timestamp: {n_ts_nat:,}")

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

#%% 6) Missing values audit (counts + %)
if not missing_expected:
    miss_counts = df[expected_cols].isna().sum()
else:
    miss_counts = df.isna().sum()

miss_pct = (miss_counts / len(df) * 100).round(2)
missing_table = (
    pd.DataFrame({"n_missing": miss_counts, "pct_missing": miss_pct})
      .sort_values(["pct_missing","n_missing"], ascending=False)
)

print("\nMissing Values Summary (top 25):")
print(missing_table.head(25))

#%% 7) Numeric columns profile (ranges, quantiles, IQR outliers)
# Detect numeric columns by dtype; also allow selected object columns to be profiled numerically
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
probe_numeric_objects = [
    "Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating",
    "Cancelled Rides by Customer","Cancelled Rides by Driver","Incomplete Rides"
]
for c in probe_numeric_objects:
    if c in df.columns and c not in numeric_cols:
        # include in profiling, but do not overwrite df[c] dtype
        numeric_cols.append(c)

def numeric_profile(frame, columns):
    prof = {}
    for c in columns:
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().any():
            q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            n_out_iqr = int(((s < lower) | (s > upper)).sum())
            prof[c] = {
                "non_null": int(s.notna().sum()),
                "min": float(np.nanmin(s)),
                "p25": float(q1),
                "median": float(np.nanmedian(s)),
                "p75": float(q3),
                "max": float(np.nanmax(s)),
                "n_missing": int(s.isna().sum()),
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "n_outliers_iqr": n_out_iqr
            }
        else:
            prof[c] = {
                "non_null": 0, "min": np.nan, "p25": np.nan, "median": np.nan,
                "p75": np.nan, "max": np.nan, "n_missing": int(len(s)),
                "lower_fence": np.nan, "upper_fence": np.nan, "n_outliers_iqr": 0
            }
    return pd.DataFrame(prof).T

num_prof = numeric_profile(df, [c for c in expected_cols if c in df.columns] + ["timestamp","Hour"])
print("\nNumeric Profile (selected):")
print(num_prof.head(25))

# Configurable rule flags (non-negative, rating bounds, hard caps)
flag_msgs = []

# Non-negative checks for common metrics
for col in ["Booking Value","Ride Distance","Avg VTAT","Avg CTAT","Driver Ratings","Customer Rating"]:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n_neg = int((s < 0).sum())
        if n_neg > 0:
            flag_msgs.append(f"- {col}: {n_neg} negative values found (should be non-negative).")

# Rating bounds
for col, (lo, hi) in CONFIG["rating_bounds"].items():
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out > 0:
            flag_msgs.append(f"- {col}: {n_out} values outside {lo}–{hi} range.")

# Hard caps (domain sanity checks)
for col, cap in CONFIG["hard_caps"].items():
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n_out = int((s > cap).sum())
        if n_out > 0:
            cap_fmt = f"{cap:,.0f}" if float(cap).is_integer() else f"{cap:,}"
            flag_msgs.append(f"- {col}: {n_out} values > {cap_fmt} (verify outliers).")

#%% 8) High-cardinality categorical checks (raw vs normalized)
cat_cols = [c for c in expected_cols if c in df.columns and df[c].dtype == "object"]
cat_card_raw = pd.Series({c: df[c].nunique(dropna=True) for c in cat_cols}).sort_values(ascending=False)

# Normalized categories: strip, lower, squish spaces
def _normalize_series(x: pd.Series) -> pd.Series:
    return (
        x.dropna().astype(str).str.strip().str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )

cat_card_norm = {}
for c in cat_cols:
    norm = _normalize_series(df[c])
    cat_card_norm[c] = int(norm.nunique())
cat_card_norm = pd.Series(cat_card_norm).sort_values(ascending=False)

print("\nCategorical Cardinality (unique non-null values) — raw (→ normalized):")
for c in cat_cols:
    print(f"{c}: {cat_card_raw.get(c,0)} → {cat_card_norm.get(c,0)}")

#%% 9) Save augmented data (with time columns only; no cleaning yet)
out_csv = OUT_DIR_DATA / "ncr_ride_bookings_week1_with_timecols.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved augmented CSV (with Date_parsed/Hour/Weekday/Month/timestamp) → {out_csv}")

#%% 10) Compose the Data Audit Report (Markdown) + Checklist (+ small JSON summary)
report_lines = []

report_lines.append("# Week 1 — Data Intake & Quality Audit\n")
report_lines.append(f"- File: `{DATA_PATH.name}`  \n- Rows by Cols: **{df.shape[0]:,} x {df.shape[1]}**\n")
if missing_expected:
    report_lines.append(f"- Missing expected columns: {missing_expected}\n")
if extra_cols:
    report_lines.append(f"- Extra columns present: {extra_cols}\n")

report_lines.append("## Schema (dtypes before cleaning)\n")
report_lines.append("```\n" + str(df.dtypes) + "\n```\n")

report_lines.append("## Missing Values Summary (counts + %)\n")
report_lines.append(missing_table.to_markdown())

report_lines.append("\n## Numeric Profile (selected + IQR outliers)\n")
report_lines.append(num_prof.to_markdown())

report_lines.append("\n## Categorical Cardinality\n")
card_df = pd.DataFrame({
    "raw_n_unique": cat_card_raw,
    "normalized_n_unique": cat_card_norm
}).sort_values("raw_n_unique", ascending=False)
report_lines.append(card_df.to_markdown())

report_lines.append("\n## Key Integrity Checks\n")
report_lines.append(f"- Exact duplicate rows: **{n_dupe_rows:,}**")
if "Booking ID" in df.columns:
    report_lines.append(f"- Unique Booking IDs: **{n_unique_bookings:,}**  \n- Possible duplicates by ID: **{dup_bookings:,}**")
report_lines.append(f"- Date NaT: **{n_date_nat:,}**; Hour NaN: **{n_hour_nan:,}**; Timestamp NaT: **{n_ts_nat:,}**")

if flag_msgs:
    report_lines.append("\n## Potential Outlier/Range Flags\n" + "\n".join(flag_msgs))

if issues:
    report_lines.append("\n## Other Issues\n" + "\n".join(issues))

audit_md_path = OUT_DIR_REPORTS / "week1_data_audit.md"
with open(audit_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nWrote data audit report → {audit_md_path}")

# Minimal JSON summary (useful to drop into thesis or CI logs)
summary = {
    "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    "duplicates": {"full_row": int(n_dupe_rows), "unique_booking_ids": int(n_unique_bookings) if "Booking ID" in df.columns else None},
    "date_parse": {"date_nat": int(n_date_nat), "hour_nan": int(n_hour_nan), "timestamp_nat": int(n_ts_nat)},
    "flags": flag_msgs + issues
}
with open(OUT_DIR_REPORTS / "week1_data_audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote JSON summary → {OUT_DIR_REPORTS / 'week1_data_audit_summary.json'}")

# Checklist (explicit actions for Week 2)
checklist = dedent("""
# Week 1 → Week 2 Checklist
- [ ] Impute event columns (0 / "Not Applicable"); finalize numeric imputation plan (KNN vs median).
- [ ] Decide treatment for extreme values in Ride Distance and Booking Value (cap vs remove) — consider using IQR bounds.
- [ ] Confirm handling of rating bounds (clip to [0,5]) and document any changes.
- [ ] Verify Booking ID duplicates; deduplicate with a clear rule if needed.
- [ ] Confirm Date/Time parsing choices and timezone (if applicable); use `timestamp` for resampling.
- [ ] Lock environment (requirements.txt or environment.yml) and commit to repo.
""").strip()

checklist_path = OUT_DIR_REPORTS / "week1_checklist.md"
with open(checklist_path, "w", encoding="utf-8") as f:
    f.write(checklist)
print(f"Wrote checklist → {checklist_path}")
#%%