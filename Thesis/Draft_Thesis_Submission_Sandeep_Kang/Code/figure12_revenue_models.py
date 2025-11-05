"""
figure12_revenue_models.py
--------------------------------------
Combines Week 6 revenue model results for the thesis (Figure 12).

Uses:
 - week6_revenue_metrics.csv
 - week6_revenue_xgb_importance.png
"""

#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths
project_path = "/Users/sandeepk/Library/Mobile Documents/com~apple~CloudDocs/IUPUI/Senior Thesis INFO-I 492/Project-Uber"
reports_path = os.path.join(project_path, "reports")
graphs_path = os.path.join(project_path, "Thesis", "Draft_Thesis_Submission_Sandeep_Kang", "Graphs")
os.makedirs(graphs_path, exist_ok=True)

# Load data
metrics = pd.read_csv(os.path.join(reports_path, "week6_revenue_metrics.csv"))
xgb_importance = mpimg.imread(os.path.join(reports_path, "week6_revenue_xgb_importance.png"))

# Plot composite
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Model Performance
metrics.plot(kind="bar", x="Model", y=["MAE", "RMSE"], ax=axes[0], color=["lightblue", "steelblue"])
axes[0].set_title("(A) Model Performance Metrics")
axes[0].set_ylabel("Error Value")

# Panel B: Feature Importance (image)
axes[1].imshow(xgb_importance)
axes[1].set_title("(B) XGBoost Feature Importance")
axes[1].axis("off")

plt.tight_layout()
save_path = os.path.join(graphs_path, "Figure12_Revenue_Model_Comparison.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print("Saved:", save_path)