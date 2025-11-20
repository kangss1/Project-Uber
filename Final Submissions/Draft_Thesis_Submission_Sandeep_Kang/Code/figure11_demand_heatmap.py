"""
figure11_demand_heatmap.py
--------------------------------------
Combines Week 7 demand analysis visuals into a
composite figure for the thesis (Figure 11).

Uses:
 - week7_peak_heatmap_annotated.png
 - week7_avg_rides_per_hour.png
 - week7_fulfillment_efficiency_heatmap.png
 - week7_peak_top10.csv

Author: Sandeep K.
Course: INFO-I 492 Senior Thesis
"""

#%%
# 1) Import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

#%%
# 2) Define paths
project_path = "/Users/sandeepk/Library/Mobile Documents/com~apple~CloudDocs/IUPUI/Senior Thesis INFO-I 492/Project-Uber"
reports_path = os.path.join(project_path, "reports")
graphs_path = os.path.join(
    project_path,
    "Thesis",
    "Draft_Thesis_Submission_Sandeep_Kang",
    "Graphs"
)

#%%
# 3) Load images and data
heatmap = mpimg.imread(os.path.join(reports_path, "week7_peak_heatmap_annotated.png"))
hourly_trend = mpimg.imread(os.path.join(reports_path, "week7_avg_rides_per_hour.png"))
fulfillment = mpimg.imread(os.path.join(reports_path, "week7_fulfillment_efficiency_heatmap.png"))

top10_path = os.path.join(reports_path, "week7_peak_top10.csv")
top10 = pd.read_csv(top10_path) if os.path.exists(top10_path) else None

#%%
# 4) Create composite figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel A: Annotated Demand Heatmap
axes[0, 0].imshow(heatmap)
axes[0, 0].set_title("A. Hourly and Weekly Demand Intensity")
axes[0, 0].axis("off")

# Panel B: Average Rides per Hour
axes[0, 1].imshow(hourly_trend)
axes[0, 1].set_title("B. Average Rides per Hour")
axes[0, 1].axis("off")

# Panel C: Fulfillment Efficiency
axes[1, 0].imshow(fulfillment)
axes[1, 0].set_title("C. Fulfillment Efficiency by Time Window")
axes[1, 0].axis("off")

# Panel D: Top 10 Demand Periods Table
axes[1, 1].axis("off")
if top10 is not None:
    table_data = top10[["Weekday", "Hour", "rides"]].head(10)
    axes[1, 1].table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center"
    )
    axes[1, 1].set_title("D. Top 10 Demand Periods")
else:
    axes[1, 1].text(0.5, 0.5, "Top 10 Demand Data\nNot Available", ha="center", va="center")

plt.tight_layout()

#%%
# 5) Save final figure
os.makedirs(graphs_path, exist_ok=True)
save_path = os.path.join(graphs_path, "Figure11_Demand_Patterns_Composite.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print("Saved:", save_path)