import matplotlib.pyplot as plt
import numpy as np

# Models and sample benchmark data (replace with values from your sources)
models = ['ARIMA', 'Random Forest', 'XGBoost', 'LSTM', 'Transformer']
mae = [165, 120, 110, 90, 80]   # Mean Absolute Error (lower is better)
rmse = [210, 160, 145, 120, 105]  # Root Mean Squared Error (lower is better)

# Bar positions
x = np.arange(len(models))
width = 0.35

# Create the figure
plt.figure(figsize=(8.5, 5))
plt.bar(x - width/2, mae, width, label='MAE', color='royalblue')
plt.bar(x + width/2, rmse, width, label='RMSE', color='lightseagreen')

# Add labels and title
plt.xlabel('Forecasting Models', fontsize=12)
plt.ylabel('Error Metrics (lower is better)', fontsize=12)
plt.title('Comparative Performance of Forecasting Models in Ride-Hailing Demand Prediction',
          fontsize=13, pad=15)
plt.xticks(x, models, fontsize=11)
plt.legend(fontsize=10, frameon=True)

# Add gridlines for readability
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save high-resolution figure (no cropping)
plt.savefig("Figure3_ModelPerformanceComparison.png", dpi=400, bbox_inches='tight')
plt.show()