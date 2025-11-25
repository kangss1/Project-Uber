import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load dataset ---
df = pd.read_csv("ncr_ride_bookings_cleaned.csv")

# --- Data cleaning and preprocessing ---
# Drop rows with missing booking or distance values
df = df.dropna(subset=['Booking Value', 'Ride Distance', 'Pickup Location', 'Hour'])

# --- Create an estimated surge multiplier ---
# Assuming that 'Booking Value / Ride Distance' approximates the effective fare rate
# Normalize by dividing each fare rate by the mean fare rate to get a relative surge multiplier
df['Fare_per_km'] = df['Booking Value'] / df['Ride Distance']
df['Surge Multiplier'] = df['Fare_per_km'] / df['Fare_per_km'].mean()

# --- Aggregate by pickup zone and hour ---
surge_summary = (
    df.groupby(['Pickup Location', 'Hour'])['Surge Multiplier']
    .mean()
    .reset_index()
)

# --- Create pivot for heatmap ---
surge_pivot = surge_summary.pivot(
    index='Pickup Location',
    columns='Hour',
    values='Surge Multiplier'
)

# Sort pickup zones by average surge to make the heatmap easier to read
surge_pivot = surge_pivot.loc[
    surge_pivot.mean(axis=1).sort_values(ascending=False).index
]

# --- Visualization ---
plt.figure(figsize=(14, 8))
sns.heatmap(
    surge_pivot,
    cmap='coolwarm',
    linewidths=0.4,
    cbar_kws={'label': 'Average Surge Multiplier'},
)

plt.title(
    "Figure 13. Spatial Distribution of Surge Multipliers — Delhi NCR",
    fontsize=16,
    weight='bold',
)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Pickup Zone", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# --- Save & Show ---
plt.savefig("Figure13_Surge_Distribution_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()