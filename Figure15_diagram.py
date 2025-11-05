"""
figure15_diagram.py
-------------------------------------------------------
Generates Figure 15 — Regional Ride Flow Heatmap
Shows total ride volumes between major Delhi NCR regions.
Uses ride pickup and drop-off data to illustrate 
cross-regional connectivity and emerging suburban mobility.
-------------------------------------------------------
Uses:
 - /data/processed/ncr_ride_bookings_cleaned.csv
Outputs:
 - /Thesis/Draft_Thesis_Submission_Sandeep_Kang/Graphs/Figure15_Regional_Ride_Flow.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# 1. Set paths (run from Project-Uber main directory)
# ---------------------------------------------------
project_dir = os.getcwd()
data_path = os.path.join(project_dir, "data", "processed", "ncr_ride_bookings_cleaned.csv")
graphs_path = os.path.join(
    project_dir,
    "Thesis",
    "Draft_Thesis_Submission_Sandeep_Kang",
    "Graphs"
)
os.makedirs(graphs_path, exist_ok=True)

# ---------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)

# Keep only relevant columns
df = df[['Pickup Location', 'Drop Location']].dropna()

# ---------------------------------------------------
# 3. Define region groupings
# ---------------------------------------------------
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

# Map locations to regions
df['Pickup Region'] = df['Pickup Location'].map(region_map)
df['Drop Region'] = df['Drop Location'].map(region_map)

# Drop rows where mapping failed
df = df.dropna(subset=['Pickup Region', 'Drop Region'])

# ---------------------------------------------------
# 4. Aggregate ride counts between regions
# ---------------------------------------------------
region_flow = (
    df.groupby(['Pickup Region', 'Drop Region'])
      .size()
      .unstack(fill_value=0)
)

# Sort for consistency
region_order = ['Central', 'South', 'West', 'North', 'Peripheral']
region_flow = region_flow.reindex(index=region_order, columns=region_order)

# ---------------------------------------------------
# 5. Plot heatmap
# ---------------------------------------------------
plt.figure(figsize=(7, 6))
sns.heatmap(region_flow, annot=True, fmt='g', cmap='YlOrRd', linewidths=0.5)
plt.title("")
plt.xlabel("Drop Region", fontsize=12)
plt.ylabel("Pickup Region", fontsize=12)
plt.tight_layout()

# ---------------------------------------------------
# 6. Save output
# ---------------------------------------------------
save_path = os.path.join(graphs_path, "Figure15_Regional_Ride_Flow.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Figure15_Regional_Ride_Flow.png saved successfully at:\n{save_path}")