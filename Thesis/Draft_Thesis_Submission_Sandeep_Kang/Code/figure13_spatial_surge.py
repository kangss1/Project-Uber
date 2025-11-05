"""
figure13_spatial_surge.py
--------------------------------------
Reconstructs the spatial-temporal surge multiplier heatmap (Figure 13)
using Week 7, Week 4, and Week 5 analysis outputs.

Author: Sandeep K.
Course: INFO-I 492 Senior Thesis
"""

#%%
# 1) Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# 2) Paths
project_path = "/Users/sandeepk/Library/Mobile Documents/com~apple~CloudDocs/IUPUI/Senior Thesis INFO-I 492/Project-Uber"
reports_path = os.path.join(project_path, "reports")
graphs_path = os.path.join(
    project_path,
    "Thesis",
    "Draft_Thesis_Submission_Sandeep_Kang",
    "Graphs"
)
os.makedirs(graphs_path, exist_ok=True)

#%%
# 3) Load data
peak = pd.read_csv(os.path.join(reports_path, "week7_peak_top10.csv"))
baseline = pd.read_csv(os.path.join(reports_path, "week4_demand_baselines.csv"))

# If baseline contains overall demand by hour or zone, rename for clarity
baseline.columns = [c.strip().lower() for c in baseline.columns]
peak.columns = [c.strip().lower() for c in peak.columns]

#%%
# 4) Prepare surge map (synthetic join)
# Normalize peak rides by baseline demand (if baseline has matching hour)
if "hour" in baseline.columns and "rides" in baseline.columns:
    merged = peak.merge(baseline, on="hour", suffixes=("_peak", "_base"))
    merged["surge_ratio"] = merged["rides_peak"] / merged["rides_base"].replace(0, np.nan)
else:
    merged = peak.copy()
    merged["surge_ratio"] = merged["rides"] / merged["rides"].max() * 2  # simple normalization

# Create pivot table (zones × hours)
pivot = merged.pivot_table(index="weekday", columns="hour", values="surge_ratio", aggfunc="mean")

# Sort weekdays logically
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = pivot.reindex(weekday_order)

#%%
# 5) Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    pivot,
    cmap="coolwarm",
    linewidths=0.3,
    linecolor="gray",
    cbar_kws={"label": "Relative Surge Multiplier"},
)
plt.title("Figure 13. Spatial Distribution of Surge Multipliers — Delhi NCR", fontsize=13, pad=20)
plt.xlabel("Hour of Day")
plt.ylabel("Weekday")

#%%
# 6) Save figure
save_path = os.path.join(graphs_path, "Figure13_Spatial_Surge_Reconstructed.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print("Saved:", save_path)