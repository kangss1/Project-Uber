# ============================================================
# Figure 15: Regional Ride Flow Heatmap
# Shows total ride volumes between major Delhi NCR regions
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the dataset ---
df = pd.read_csv("ncr_ride_bookings_cleaned.csv")

# --- Keep only relevant columns ---
df = df[['Pickup Location', 'Drop Location']].dropna()

# --- Define region groupings (edit or expand if needed) ---
region_map = {
    # --- Central Delhi ---
    'Connaught Place': 'Central',
    'Barakhamba Road': 'Central',
    'Pragati Maidan': 'Central',
    'AIIMS': 'Central',

    # --- South Delhi ---
    'Saket': 'South',
    'Greater Kailash': 'South',
    'Mehrauli': 'South',
    'Badarpur': 'South',

    # --- West Delhi ---
    'Tilak Nagar': 'West',
    'Vishwavidyalaya': 'West',
    'Madipur': 'West',
    'Shivaji Park': 'West',

    # --- North Delhi ---
    'Inderlok': 'North',
    'Kanhaiya Nagar': 'North',
    'Ashok Park Main': 'North',

    # --- Peripheral / NCR Fringe ---
    'Dwarka Sector 21': 'Peripheral',
    'Khandsa': 'Peripheral',
    'Udyog Vihar': 'Peripheral',
    'Jasola': 'Peripheral',
    'Pataudi Chowk': 'Peripheral',
    'Nehru Place': 'Peripheral'
}

# --- Map locations to regions ---
df['Pickup Region'] = df['Pickup Location'].map(region_map)
df['Drop Region'] = df['Drop Location'].map(region_map)

# --- Drop rides where mapping failed ---
df = df.dropna(subset=['Pickup Region', 'Drop Region'])

# --- Aggregate ride counts between regions ---
region_flow = (
    df.groupby(['Pickup Region', 'Drop Region'])
      .size()
      .unstack(fill_value=0)
)

# --- Optional: Sort regions for readability ---
region_order = ['Central', 'South', 'West', 'North', 'Peripheral']
region_flow = region_flow.reindex(index=region_order, columns=region_order)

# --- Plot the heatmap ---
plt.figure(figsize=(7, 6))
sns.heatmap(region_flow, annot=True, fmt='g', cmap='YlOrRd', linewidths=0.5)
plt.title("Figure 15. Regional Ride Flow Heatmap", fontsize=15, weight='bold')
plt.xlabel("Drop Region", fontsize=12)
plt.ylabel("Pickup Region", fontsize=12)
plt.tight_layout()

# --- Save and show ---
plt.savefig("Figure15_Regional_Ride_Flow.png", dpi=300, bbox_inches='tight')
plt.show()