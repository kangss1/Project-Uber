import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the actual dataset
df = pd.read_csv("ncr_ride_bookings_cleaned.csv")

# Ensure timestamp is parsed
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Use only completed or valid rides
# (Assuming 'Booking Status' indicates ride completion)
valid_rides = df[df['Booking Status'].str.contains("Completed", case=False, na=False)]

# Group by weekday and hour to compute average ride count
pivot = (
    valid_rides.groupby(['Weekday', 'Hour'])
    .size()
    .reset_index(name='ride_count')
    .pivot(index='Weekday', columns='Hour', values='ride_count')
)

# Order weekdays correctly
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(days_order)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot,
    cmap="YlGnBu",
    linewidths=0.3,
    cbar_kws={'label': 'Average Ride Requests'}
)

plt.title("Figure 11. Hourly and Weekly Demand Pattern Heatmap", fontsize=13, weight='bold')
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")

plt.tight_layout()
plt.savefig("Figure11_Hourly_Weekly_Demand_Heatmap.png", dpi=300, bbox_inches='tight')
plt.show()