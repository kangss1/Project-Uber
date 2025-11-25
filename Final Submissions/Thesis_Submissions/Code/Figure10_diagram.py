import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# Load the images
images = [
    "week5_demand_rf_tuned.png",
    "week5_demand_xgb_tuned.png",
    "week5_demand_xgb_importance.png",
    "week5_demand_xgb_shap_summary.png"
]
titles = ["(A) Random Forest Forecast", "(B) XGBoost Forecast",
          "(C) XGBoost Feature Importance", "(D) SHAP Summary Plot"]

# Display them
for ax, img, title in zip(axes.flatten(), images, titles):
    ax.imshow(mpimg.imread(img))
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("Figure10_Combined_Performance_And_Features.png", bbox_inches="tight", dpi=300)
plt.close()