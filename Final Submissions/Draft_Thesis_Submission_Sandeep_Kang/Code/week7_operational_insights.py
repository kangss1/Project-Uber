#%% [markdown]
# ============================================
# Week 7 — Operational Insights (Peak Load & Cancellations)
# File: week7_operational_insights.py
# Depends on: Week 2/3 outputs → data/processed/ncr_ride_bookings_cleaned.(parquet|csv)
# ============================================
#
# Purpose
#   Turn modeling-ready data into actionable ops insights:
#    1) Peak-load matrix (Weekday × Hour) + Top-10 peak slots
#    2) "No Driver Found" cancellation analysis by hour/weekday/vehicle type
#    3) A short, human-readable ops brief with staffing/fleet/pricing cues
#
# Outputs
#   reports/week7_peak_heatmap.png
#   reports/week7_peak_top10.csv
#   reports/week7_cancellation_heatmap.png
#   reports/week7_cancel_by_vehicle.png
#   reports/week7_peak_heatmap_annotated.png
#   reports/week7_fulfillment_efficiency_heatmap.png
#   reports/week7_avg_rides_per_hour.png
#   reports/week7_ops_brief.md
#   data/processed/week7_peak_table.csv
#   data/processed/week7_cancellation_tables.csv
#   data/processed/week7_summary_table.csv
# ============================================

#%% 
# 0) Imports & Paths
from pathlib import Path
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project dirs
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

DATA_PROC = ROOT / "data" / "processed"
DATA_INT  = ROOT / "data" / "interim"
REPORTS   = ROOT / "reports"
for d in (DATA_PROC, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Weekday order for plots/tables
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKDAY_TO_IDX = {d:i for i,d in enumerate(WEEKDAY_ORDER)}

#%% 
# 1) Load cleaned dataset (Week 2 output; fallback to Week 1 interim if needed)
def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the expected input files were found:\n" + "\n".join(str(p) for p in paths))

cand_paths = [
    DATA_PROC / "ncr_ride_bookings_cleaned.parquet",
    DATA_PROC / "ncr_ride_bookings_cleaned.csv",
    DATA_INT  / "ncr_ride_bookings_week1_with_timecols.csv"  # last resort
]
in_path = _find_first_existing(cand_paths)

if in_path.suffix.lower() == ".parquet":
    df = pd.read_parquet(in_path)
else:
    df = pd.read_csv(in_path)

print(f"Loaded: {in_path.name}, shape={df.shape}")

# Ensure time columns exist
if "Date_parsed" in df.columns:
    df["Date_parsed"] = pd.to_datetime(df["Date_parsed"], errors="coerce")
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if "Hour" not in df.columns and "timestamp" in df.columns:
    df["Hour"] = df["timestamp"].dt.hour
if "Weekday" not in df.columns and "Date_parsed" in df.columns:
    df["Weekday"] = df["Date_parsed"].dt.day_name()

# Coerce Hour to int when possible
if "Hour" in df.columns:
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").astype("Int64")

# Guard: keep rows with Weekday & Hour
df = df[df["Weekday"].notna() & df["Hour"].notna()].copy()
df["Weekday"] = df["Weekday"].astype(str)
df["Hour"]    = df["Hour"].astype(int)
print("After Weekday/Hour guard:", df.shape)

#%% 
# 2) Peak-load matrix (Weekday × Hour) + Top-10 peak slots
def _weekday_category(series: pd.Series) -> pd.Series:
    # Map to ordered Categorical; unknowns go to end (rare)
    cat = pd.Categorical(series, categories=WEEKDAY_ORDER, ordered=True)
    return cat

df["Weekday_cat"] = _weekday_category(df["Weekday"])
peak_pivot = (
    df
    .groupby(["Weekday_cat","Hour"])
    .size()
    .rename("rides")
    .reset_index()
    .pivot(index="Weekday_cat", columns="Hour", values="rides")
    .fillna(0)
    .astype(int)
)

# Save raw table for reference
peak_table_csv = DATA_PROC / "week7_peak_table.csv"
peak_pivot.to_csv(peak_table_csv)
print("Saved peak table →", peak_table_csv)

# Top-10 slots by absolute ride count
peak_long = peak_pivot.reset_index().melt(id_vars="Weekday_cat", var_name="Hour", value_name="rides")
peak_long = peak_long.rename(columns={"Weekday_cat":"Weekday"})
top10 = peak_long.sort_values("rides", ascending=False).head(10)
top10_csv = REPORTS / "week7_peak_top10.csv"
top10.to_csv(top10_csv, index=False)
print("Saved Top-10 peak slots →", top10_csv)

# Heatmap with matplotlib (no seaborn)
def plot_heatmap(values_df: pd.DataFrame, title: str, outpath: Path, cmap="viridis", cbar_label="Rides"):
    arr = values_df.values
    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Weekday")
    ax.set_xticks(range(values_df.shape[1]))
    ax.set_xticklabels(values_df.columns.tolist(), rotation=0)
    ax.set_yticks(range(values_df.shape[0]))
    ax.set_yticklabels(list(values_df.index.astype(str)))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

peak_heatmap_png = REPORTS / "week7_peak_heatmap.png"
plot_heatmap(peak_pivot, "Peak Load — Rides by Weekday × Hour", peak_heatmap_png, cbar_label="Rides")
print("Saved →", peak_heatmap_png)

#%% 
# 3) "No Driver Found" cancellation analysis (hour, weekday, vehicle)
def is_no_driver_found(row) -> bool:
    checks = []
    for col in ["Booking Status", "Driver Cancellation Reason", "Reason for cancelling by Customer"]:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).strip().lower()
            checks.append("no driver found" in s)
    return any(checks)

df["no_driver_found"] = df.apply(is_no_driver_found, axis=1)

# Cancellation rate by Weekday × Hour
cell_counts = df.groupby(["Weekday_cat","Hour"]).size().rename("n_total")
cell_ndf    = df.groupby(["Weekday_cat","Hour"])["no_driver_found"].sum().rename("n_ndf")
cancel_mat = pd.concat([cell_counts, cell_ndf], axis=1).reset_index()
cancel_mat["rate_ndf"] = (cancel_mat["n_ndf"] / cancel_mat["n_total"]).fillna(0.0)
cancel_pivot = cancel_mat.pivot(index="Weekday_cat", columns="Hour", values="rate_ndf").fillna(0.0)

cancel_csv = DATA_PROC / "week7_cancellation_tables.csv"
cancel_mat.to_csv(cancel_csv, index=False)
print("Saved cancellation tables →", cancel_csv)

cancel_hm_png = REPORTS / "week7_cancellation_heatmap.png"
plot_heatmap(cancel_pivot, "No Driver Found — Cancellation Rate by Weekday × Hour", cancel_hm_png, cmap="magma", cbar_label="Rate (0–1)")
print("Saved →", cancel_hm_png)

# Cancellation rate by Vehicle Type
veh_col = "Vehicle Type" if "Vehicle Type" in df.columns else None
if veh_col:
    by_veh = (
        df.groupby(veh_col)
          .agg(n_total=("Booking ID","size") if "Booking ID" in df.columns else ("no_driver_found","size"),
               n_ndf=("no_driver_found","sum"))
          .reset_index()
    )
    by_veh["rate_ndf"] = (by_veh["n_ndf"] / by_veh["n_total"]).fillna(0.0)
    by_veh = by_veh.sort_values("rate_ndf", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(by_veh[veh_col].astype(str), by_veh["rate_ndf"])
    ax.set_title("No Driver Found — Cancellation Rate by Vehicle Type")
    ax.set_xlabel("Vehicle Type"); ax.set_ylabel("Rate (0–1)")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30); tick.set_ha("right")
    plt.tight_layout()
    cancel_by_vehicle_png = REPORTS / "week7_cancel_by_vehicle.png"
    plt.savefig(cancel_by_vehicle_png, dpi=150)
    plt.close(fig)
    print("Saved →", cancel_by_vehicle_png)
else:
    by_veh = pd.DataFrame(columns=["Vehicle Type","n_total","n_ndf","rate_ndf"])
    print("Vehicle Type column not found — skipping vehicle cancellation bar.")

cancel_hotspots = (
    cancel_mat.sort_values("rate_ndf", ascending=False)
              .head(10)
              .copy()
)
cancel_hotspots["Weekday"] = cancel_hotspots["Weekday_cat"].astype(str)
cancel_hotspots = cancel_hotspots[["Weekday","Hour","n_total","n_ndf","rate_ndf"]]

#%% 
# 4) Draft Ops Brief (Markdown) with concrete findings
def _slot_pretty(row):
    return f"{row['Weekday']} {int(row['Hour']):02d}:00"

top10_lines = [
    f"- {row['Weekday']} {int(row['Hour']):02d}:00 — {int(row['rides']):,} rides"
    for _, row in top10.iterrows()
]
hotspot_lines = [
    f"- {row['Weekday']} {int(row['Hour']):02d}:00 — rate={row['rate_ndf']:.2%} (N={int(row['n_total'])})"
    for _, row in cancel_hotspots.iterrows()
]

def _recommended_actions():
    actions = []
    if not top10.empty:
        morning_share = (top10["Hour"].between(7,10).sum()) / len(top10)
        evening_share = (top10["Hour"].between(16,20).sum()) / len(top10)
        if morning_share > 0.3:
            actions.append("- Increase driver availability 7–10 AM on high-demand weekdays via pre-shift incentives or login bonuses.")
        if evening_share > 0.3:
            actions.append("- Add surge guardrails and short incentives 5–8 PM for congestion relief and fulfillment.")
    if not by_veh.empty:
        worst = by_veh.iloc[0]
        actions.append(f"- Review {worst[veh_col]} coverage or assignment rules (highest NDF rate at {worst['rate_ndf']:.1%}).")
    if not cancel_hotspots.empty:
        actions.append("- Pilot micro-surge or upfront driver bonuses in the top cancellation windows.")
    if not actions:
        actions.append("- Maintain current staffing; monitor demand and cancellations weekly.")
    return actions

ops_brief = dedent(f"""
# Week 7 — Operational Insights

**Input:** `{in_path.name}`  
**Rows used:** {len(df):,}  
**Artifacts:** 
- `week7_peak_heatmap.png`
- `week7_peak_top10.csv`
- `week7_cancellation_heatmap.png`
- `week7_cancel_by_vehicle.png` (if vehicle type available)

## Peak Demand (Weekday × Hour)
Top 10 peaks by absolute rides:
{chr(10).join(top10_lines) if top10_lines else "- n/a"}

## “No Driver Found” — Cancellation Hotspots
Highest NDF rates (top 10 cells by Weekday × Hour):
{chr(10).join(hotspot_lines) if hotspot_lines else "- n/a"}

## Recommendations (Staffing • Fleet Mix • Pricing)
{chr(10).join(_recommended_actions())}
""").strip()

brief_md = REPORTS / "week7_ops_brief.md"
with open(brief_md, "w", encoding="utf-8") as f:
    f.write(ops_brief)
print("Wrote ops brief →", brief_md)

#%% 
# 5) Enhancements for Richer Insights & Presentation Polish
# These do NOT alter previous results — only add visuals and summaries.

# 5A) Annotated Heatmap for Peak Demand
fig, ax = plt.subplots(figsize=(14, 5.5))
im = ax.imshow(peak_pivot.values, aspect="auto", cmap="viridis", origin="upper")
ax.set_title("Peak Load — Annotated Top 10 Demand Slots")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Weekday")
ax.set_xticks(range(peak_pivot.shape[1]))
ax.set_xticklabels(peak_pivot.columns.tolist())
ax.set_yticks(range(peak_pivot.shape[0]))
ax.set_yticklabels(list(peak_pivot.index.astype(str)))
for _, row in top10.iterrows():
    y = list(peak_pivot.index).index(row["Weekday"])
    x = list(peak_pivot.columns).index(row["Hour"])
    ax.scatter(x, y, s=100, edgecolors="white", facecolors="none", linewidths=2)
    ax.text(x, y, f"{int(row['rides']):,}", color="white", ha="center", va="center", fontsize=8, weight="bold")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Rides")
plt.tight_layout()
plt.savefig(REPORTS / "week7_peak_heatmap_annotated.png", dpi=150)
plt.close(fig)

# 5B) Fulfillment Efficiency Heatmap (1 - NDF Rate)
eff_pivot = (1 - cancel_pivot).clip(0, 1)
fig, ax = plt.subplots(figsize=(14, 5.5))
im = ax.imshow(eff_pivot.values, aspect="auto", cmap="YlGn", origin="upper")
ax.set_title("Fulfillment Efficiency (1 - NDF Rate)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Weekday")
ax.set_xticks(range(eff_pivot.shape[1]))
ax.set_xticklabels(eff_pivot.columns.tolist())
ax.set_yticks(range(eff_pivot.shape[0]))
ax.set_yticklabels(list(eff_pivot.index.astype(str)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Efficiency (0–1)")
plt.tight_layout()
plt.savefig(REPORTS / "week7_fulfillment_efficiency_heatmap.png", dpi=150)
plt.close(fig)

# 5C) Summary Table Export for Slides or Final Report
summary_csv = DATA_PROC / "week7_summary_table.csv"
summary_df = pd.DataFrame({
    "Metric": ["Top Peak Hour", "Peak Weekday", "Highest NDF Hour", "Highest NDF Weekday"],
    "Value": [
        f"{int(top10.iloc[0]['Hour'])}:00",
        top10.iloc[0]['Weekday'],
        f"{int(cancel_hotspots.iloc[0]['Hour'])}:00",
        cancel_hotspots.iloc[0]['Weekday']
    ]
})
summary_df.to_csv(summary_csv, index=False)
print("Saved summary table →", summary_csv)

# 5D) Optional: Trend Line of Average Rides per Hour Across Weekdays
avg_by_hour = df.groupby("Hour")["Booking ID"].count() if "Booking ID" in df.columns else df.groupby("Hour").size()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_by_hour.index, avg_by_hour.values, marker="o")
ax.set_title("Average Rides per Hour (All Weekdays Combined)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Ride Count")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(REPORTS / "week7_avg_rides_per_hour.png", dpi=150)
plt.close(fig)

print("\nEnhanced visuals and summary tables added successfully.")
print("Week 7 operational insights complete (with enhancements).")
#%%