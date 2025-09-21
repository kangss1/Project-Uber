#%% [markdown]
# ============================================
# Week 2 — Cleaning & Core EDA
# File: week2_cleaning_eda.py
# Depends on: week1_data_audit.py → ncr_ride_bookings_week1_with_timecols.csv
# ============================================

#%% 0) Imports & Paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import KNNImputer

# Project paths
ROOT = Path(".").resolve()
IN_DATA = ROOT / "data" / "interim" / "ncr_ride_bookings_week1_with_timecols.csv"   # from Week 1
OUT_DIR_DATA = ROOT / "data" / "processed"
OUT_DIR_REPORTS = ROOT / "reports"
OUT_DIRS = [OUT_DIR_DATA, OUT_DIR_REPORTS]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

#%% 1) Load dataset
df = pd.read_csv(IN_DATA)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

#%% 2) Handle missing values
# Event columns: replace NaN with 0 or "Not Applicable"
event_cols = [
    "Cancelled Rides by Customer", "Reason for cancelling by Customer",
    "Cancelled Rides by Driver", "Driver Cancellation Reason",
    "Incomplete Rides", "Incomplete Rides Reason"
]
for col in event_cols:
    if col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Not Applicable")

# Numeric columns: KNN Imputer (fallback: median)
num_cols = ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating"]
num_cols = [c for c in num_cols if c in df.columns]

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

#%% 3) Remove/clip extreme outliers
# Clip Ride Distance > 200 km
if "Ride Distance" in df.columns:
    df.loc[df["Ride Distance"] > 200, "Ride Distance"] = 200

# Clip Booking Value > 10,000
if "Booking Value" in df.columns:
    df.loc[df["Booking Value"] > 10000, "Booking Value"] = 10000

# Clip Ratings to [0,5]
for col in ["Driver Ratings","Customer Rating"]:
    if col in df.columns:
        df[col] = df[col].clip(0,5)

#%% 4) Initial EDA
# Distributions
df[num_cols].hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.savefig(OUT_DIR_REPORTS / "week2_distributions.png")
plt.close()

# Correlations
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Numeric Correlation Matrix")
plt.savefig(OUT_DIR_REPORTS / "week2_corr_matrix.png")
plt.close()

# Time series plots (Daily rides)
if "Date_parsed" in df.columns:
    daily_counts = df.groupby("Date_parsed").size()
    daily_counts.plot(figsize=(14,7), title="Daily Rides (Raw)")
    plt.ylabel("Rides")
    plt.savefig(OUT_DIR_REPORTS / "week2_daily_rides.png")
    plt.close()

    daily_counts.rolling(7).mean().plot(figsize=(12,6), title="7-Day Rolling Average")
    plt.ylabel("Rides")
    plt.savefig(OUT_DIR_REPORTS / "week2_daily_rides_7day.png")
    plt.close()

    daily_counts.rolling(28).mean().plot(figsize=(12,6), title="28-Day Rolling Average")
    plt.ylabel("Rides")
    plt.savefig(OUT_DIR_REPORTS / "week2_daily_rides_28day.png")
    plt.close()

#%% 5) Save cleaned dataset
out_csv = OUT_DIR_DATA / "ncr_ride_bookings_cleaned.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved cleaned dataset → {out_csv}")

#%% 6) Write EDA Report
with open(OUT_DIR_REPORTS / "week2_cleaning_eda.md", "w", encoding="utf-8") as f:
    f.write("# Week 2 — Cleaning & Core EDA\n")
    f.write(f"- Input file: {IN_DATA.name}\n")
    f.write(f"- Output file: {out_csv.name}\n\n")
    f.write("## Cleaning Steps Applied\n")
    f.write("- Missing event columns imputed (0 / Not Applicable)\n")
    f.write("- Numeric columns imputed with KNN\n")
    f.write("- Outliers clipped (Distance >200km, Value >10,000, Ratings → [0,5])\n\n")
    f.write("## Generated Visuals\n")
    f.write("- week2_distributions.png (numeric distributions)\n")
    f.write("- week2_corr_matrix.png (correlation heatmap)\n")
    f.write("- week2_daily_rides.png (daily rides)\n")
    f.write("- week2_daily_rides_7day.png (7-day rolling)\n")
    f.write("- week2_daily_rides_28day.png (28-day rolling)\n")
    #%%