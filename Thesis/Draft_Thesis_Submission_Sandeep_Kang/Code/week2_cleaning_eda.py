#%% [markdown]
# ============================================
# Week 2 — Cleaning & Core EDA
# File: week2_cleaning_eda.py
# Depends on: week1_data_audit.py → data/interim/ncr_ride_bookings_week1_with_timecols.csv
# ============================================
#
# Purpose
# Apply rigorous cleaning informed by Week 1:
#  - Deduplicate Booking IDs with transparent priority rules
#  - Impute missing values (baseline: median/mode; optional: KNN)
#  - Cap extreme outliers with percentile-based fences; clip ratings
#  - Re-run a post-clean audit and generate baseline EDA plots
#  - Output a thesis-ready cleaning report + cleaned dataset
#
# Deliverables
#  - data/processed/ncr_ride_bookings_cleaned.csv (and parquet if possible)
#  - reports/week2_cleaning_eda.md  (what changed, why, and after-clean checks)
#  - reports/week2_distributions.png
#  - reports/week2_corr_matrix.png
#  - reports/week2_daily_rides.png
#  - reports/week2_daily_rides_7day.png
#  - reports/week2_daily_rides_28day.png

#%% 0) Imports & Paths
from pathlib import Path
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.ticker import MaxNLocator

# Optional (off by default): KNN imputation for core numerics
USE_KNN = False
try:
    from sklearn.impute import KNNImputer
    HAS_KNN = True
except Exception:
    HAS_KNN = False
    USE_KNN = False

ROOT = Path(".").resolve()
IN_DATA = ROOT / "data" / "interim" / "ncr_ride_bookings_week1_with_timecols.csv"
OUT_DIR_DATA = ROOT / "data" / "processed"
OUT_DIR_REPORTS = ROOT / "reports"
OUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

assert IN_DATA.exists(), f"Input not found: {IN_DATA}"

#%% 1) Load dataset
df = pd.read_csv(IN_DATA)
n0 = len(df)
cols0 = df.columns.tolist()
print("Loaded:", IN_DATA.name, "shape:", df.shape)

# Ensure datetime columns are proper
if "Date_parsed" in df.columns:
    df["Date_parsed"] = pd.to_datetime(df["Date_parsed"], errors="coerce")
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

#%% 2) Deduplicate by Booking ID (from Week 1 audit)
# Priority:
#   (1) keep non-null Booking Status
#   (2) keep row with most non-missing values
#   (3) stable order otherwise
if "Booking ID" in df.columns:
    df["_non_missing"] = df.notna().sum(axis=1)
    df["_status_flag"] = df["Booking Status"].notna().astype(int) if "Booking Status" in df.columns else 0
    before = len(df)
    df = (
        df.sort_values(by=["_status_flag", "_non_missing"], ascending=[False, False])
          .drop_duplicates(subset=["Booking ID"], keep="first")
          .sort_index()
          .copy()
    )
    df.drop(columns=["_non_missing", "_status_flag"], errors="ignore", inplace=True)
    print(f"De-duplicated Booking ID: removed {before - len(df):,} rows (now {len(df):,})")

#%% 3) Missing value handling (median/mode baseline; optional KNN for core numerics)
event_num_cols = [
    "Cancelled Rides by Customer",
    "Cancelled Rides by Driver",
    "Incomplete Rides"
]
event_text_cols = [
    "Reason for cancelling by Customer",
    "Driver Cancellation Reason",
    "Incomplete Rides Reason"
]
core_num_cols = [c for c in ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating"] if c in df.columns]
core_cat_cols = [c for c in ["Payment Method"] if c in df.columns]

# Events → 0 / "Not Applicable"
for c in event_num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

for c in event_text_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Not Applicable")

# Categoricals → mode
for c in core_cat_cols:
    if c in df.columns:
        mode_val = df[c].mode(dropna=True)
        mode_val = mode_val.iat[0] if not mode_val.empty else "Unknown"
        df[c] = df[c].fillna(mode_val)

# Core numerics → median or KNN
if USE_KNN and HAS_KNN and core_num_cols:
    imputer = KNNImputer(n_neighbors=5)
    df[core_num_cols] = imputer.fit_transform(df[core_num_cols])
    impute_note = "KNNImputer (k=5) on core numerics"
else:
    for c in core_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    impute_note = "Median for core numerics (baseline)"

#%% 4) Outliers (percentile capping + rating clipping)
# Ratings in [0,5]
for c in [x for x in ["Driver Ratings","Customer Rating"] if x in df.columns]:
    df[c] = df[c].clip(0, 5)

# Cap Ride Distance / Booking Value using 1st–99th percentiles
caps_applied = []
for c in [x for x in ["Ride Distance","Booking Value"] if x in df.columns]:
    s = pd.to_numeric(df[c], errors="coerce")
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    df[c] = s.clip(lower=lo, upper=hi)
    caps_applied.append((c, float(lo), float(hi)))

#%% 5) Post-clean audit (duplicates + missingness)
dupes_after = int(df.duplicated().sum())
miss_counts = df.isna().sum()
miss_pct = (miss_counts / len(df) * 100).round(2)
missing_table = (
    pd.DataFrame({"n_missing": miss_counts, "pct_missing": miss_pct})
      .sort_values(["pct_missing","n_missing"], ascending=False)
)

#%% 6) Core EDA visuals (matplotlib only; formatted axes)
def _format_date_axis(ax):
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

num_for_plots = [c for c in core_num_cols if c in df.columns]
if num_for_plots:
    df[num_for_plots].hist(bins=40, figsize=(14,9))
    plt.tight_layout()
    plt.savefig(OUT_DIR_REPORTS / "week2_distributions.png", dpi=150)
    plt.close()

# Correlation heatmap
if num_for_plots:
    corr = df[num_for_plots].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11,9))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax)
    ax.set_title("Numeric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR_REPORTS / "week2_corr_matrix.png", dpi=150)
    plt.close()

# Daily rides (sorted by date) — raw + 7d/28d rolling with clean axes
if "Date_parsed" in df.columns:
    daily = (
        df[df["Date_parsed"].notna()]
        .groupby("Date_parsed")
        .size()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(daily.index, daily.values)
    ax.set_title("Daily Rides (Raw)")
    ax.set_ylabel("Rides"); ax.set_xlabel("Date")
    _format_date_axis(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR_REPORTS / "week2_daily_rides.png", dpi=150)
    plt.close(fig)

    for win, fname in [(7, "week2_daily_rides_7day.png"),
                       (28, "week2_daily_rides_28day.png")]:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(daily.index, daily.rolling(win, min_periods=1).mean())
        ax.set_title(f"{win}-Day Rolling Average")
        ax.set_ylabel("Rides"); ax.set_xlabel("Date")
        _format_date_axis(ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR_REPORTS / fname, dpi=150)
        plt.close(fig)

#%% 7) Save cleaned dataset (with parquet→csv fallback)
def _safe_save(df_: pd.DataFrame, pq_path: Path, csv_path: Path) -> Path:
    try:
        df_.to_parquet(pq_path, index=False)  # requires pyarrow/fastparquet
        return pq_path
    except Exception:
        df_.to_csv(csv_path, index=False)
        return csv_path

csv_path = OUT_DIR_DATA / "ncr_ride_bookings_cleaned.csv"
pq_path  = OUT_DIR_DATA / "ncr_ride_bookings_cleaned.parquet"
written_path = _safe_save(df, pq_path, csv_path)
print("Saved cleaned dataset →", written_path)

#%% 8) Write the Week 2 Cleaning Report (Markdown)
cap_lines = "\n".join([f"- {c}: capped to [{lo:.2f}, {hi:.2f}] (1st–99th pct)" for c, lo, hi in caps_applied]) or "- None"

report = dedent(f"""
# Week 2 — Cleaning & Core EDA

**Input**: `{IN_DATA.name}`  
**Output**: `{written_path.name}`  
**Rows after de-dup**: {len(df):,} (removed {n0 - len(df):,} from {n0:,})

## Steps Applied
- **Deduplication**: Dropped duplicate `Booking ID`s using a transparent priority:
  (1) keep non-null `Booking Status`, (2) keep the row with the fewest missing values, (3) preserve original order on ties.
- **Imputation**:
  - Events (counts) → 0; event reasons → "Not Applicable".
  - Core categoricals (e.g., Payment Method) → mode.
  - Core numerics → {impute_note}.
- **Outliers**:
{cap_lines}
  - Ratings clipped to [0, 5].

## Post-clean Audit
- Exact duplicate rows remaining: **{dupes_after:,}**
- Missing values (top 15):
""").strip()

with open(OUT_DIR_REPORTS / "week2_cleaning_eda.md", "w", encoding="utf-8") as f:
    f.write(report + "\n\n")
    f.write(missing_table.head(15).to_markdown() + "\n\n")
    f.write(dedent("""
## Visuals
- `week2_distributions.png` — numeric distributions (after cleaning)
- `week2_corr_matrix.png` — numeric correlation heatmap
- `week2_daily_rides.png` — daily ride counts (raw)
- `week2_daily_rides_7day.png` — 7-day rolling average
- `week2_daily_rides_28day.png` — 28-day rolling average
"""))

print("Wrote report →", OUT_DIR_REPORTS / "week2_cleaning_eda.md")