import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load actual revenue metrics
metrics = pd.DataFrame({
    'Model': ['RF (logY)', 'XGB (logY)'],
    'MAE': [1.1864, 5.5467],
    'RMSE': [3.5352, 17.4577]
})

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel (A): RMSE & MAE Comparison ---
metrics_melted = metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(
    x='Model', y='Value', hue='Metric', data=metrics_melted,
    palette='Blues', ax=axes[0]
)
axes[0].set_title('(A) Model Performance Metrics', fontsize=12, weight='bold')
axes[0].set_ylabel('Error Value')
axes[0].set_xlabel('')
axes[0].legend(title='Metric')

# --- Panel (B): XGBoost Feature Importance ---
xgb_imp = plt.imread("week6_revenue_xgb_importances.png")
axes[1].imshow(xgb_imp)
axes[1].axis('off')
axes[1].set_title('(B) XGBoost Feature Importance', fontsize=12, weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Figure12_Fare_Prediction_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()