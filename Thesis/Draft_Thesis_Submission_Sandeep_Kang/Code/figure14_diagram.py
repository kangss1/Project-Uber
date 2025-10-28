import pandas as pd
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_csv("ncr_ride_bookings_cleaned.csv")

# --- Preprocessing ---
df['is_completed'] = df['Booking Status'].str.lower().eq('completed').astype(int)
df['is_cancelled'] = df['Booking Status'].str.lower().str.contains('cancel').astype(int)

# --- Aggregate by Hour ---
hourly_stats = (
    df.groupby('Hour')
    .agg(
        total_requests=('Booking ID', 'count'),
        completed=('is_completed', 'sum'),
        cancelled=('is_cancelled', 'sum')
    )
    .reset_index()
)

# --- Compute metrics ---
hourly_stats['Fulfillment Rate (%)'] = 100 * hourly_stats['completed'] / hourly_stats['total_requests']
hourly_stats['Cancellation Rate (%)'] = 100 * hourly_stats['cancelled'] / hourly_stats['total_requests']

# --- Plot new diagnostic chart ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot total demand (bars)
ax1.bar(hourly_stats['Hour'], hourly_stats['total_requests'], color='lightgray', alpha=0.6, label='Total Ride Requests')
ax1.set_xlabel("Hour of Day", fontsize=12)
ax1.set_ylabel("Total Ride Requests", color='gray', fontsize=12)

# Add shaded peak-hour bands
ax1.axvspan(7, 9, color='gray', alpha=0.15)
ax1.axvspan(20, 22, color='gray', alpha=0.15)

# Second y-axis for rates
ax2 = ax1.twinx()
ax2.plot(hourly_stats['Hour'], hourly_stats['Fulfillment Rate (%)'], color='green', linewidth=2.5, label='Fulfillment Rate (%)')
ax2.plot(hourly_stats['Hour'], hourly_stats['Cancellation Rate (%)'], color='red', linestyle='--', linewidth=2.5, label='Cancellation Rate (%)')
ax2.set_ylabel("Rate (%)", fontsize=12)

# --- Aesthetics ---
ax1.set_title("Figure 14. Hourly Ride Demand, Fulfillment Efficiency, and Cancellation Rates", fontsize=15, weight='bold')
ax1.grid(True, alpha=0.3)

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig("Figure14_Demand_Fulfillment_Cancellation.png", dpi=300, bbox_inches='tight')
plt.show()