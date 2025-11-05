"""
figure16_diagram.py
-------------------------------------------------------
Generates Figure 16 — Emission Reduction Visualization
Analyzes CO₂ emissions by region and compares baseline
vs optimized ride-hailing scenarios (9% reduction).
-------------------------------------------------------
Uses:
 - /data/processed/ncr_ride_bookings_cleaned.csv
Outputs:
 - /Thesis/Draft_Thesis_Submission_Sandeep_Kang/Graphs/Figure16_Emission_Reduction.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

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

# Filter for completed rides
df = df[df["Booking Status"].str.lower().str.contains("completed", na=False)]

# ---------------------------------------------------
# 3. Estimate CO₂ emissions by vehicle type
# ---------------------------------------------------
emission_factors = {
    "Go Mini": 130,        # small hatchback (CNG/petrol)
    "Go Sedan": 150,       # larger sedan
    "Premier Sedan": 160,
    "Auto": 70,            # three-wheeler CNG
    "Bike": 40,            # two-wheeler
    "eBike": 0,            # electric
    "Uber XL": 180         # SUV/van
}

df["Emission_Factor"] = df["Vehicle Type"].map(emission_factors).fillna(120)
df["Emission_kg"] = (df["Ride Distance"] * df["Emission_Factor"]) / 1000  # convert g → kg

# Optimized scenario (9% reduction)
df["Emission_Optimized_kg"] = df["Emission_kg"] * 0.91

# ---------------------------------------------------
# 4. Map pickup locations to regions
# ---------------------------------------------------
region_map = {}
for z in df["Pickup Location"].unique():
    name = str(z).lower()
    if any(k in name for k in ["connaught", "karol", "aiims", "mandi house", "india gate", "rajiv chowk", "cp"]):
        region_map[z] = "Central"
    elif any(k in name for k in ["rohini", "pitampura", "ashok vihar", "model town", "adarsh nagar"]):
        region_map[z] = "North"
    elif any(k in name for k in ["dwarka", "janakpuri", "uttam nagar", "tilak nagar", "punjabi bagh"]):
        region_map[z] = "West"
    elif any(k in name for k in ["noida", "anand vihar", "mayur vihar", "karkarduma", "preet vihar"]):
        region_map[z] = "East"
    elif any(k in name for k in ["mehrauli", "saket", "malviya", "okhla", "lajpat", "nehru place", "greater kailash"]):
        region_map[z] = "South"
    elif any(k in name for k in ["gurgaon", "faridabad", "manesar", "sohna", "bhiwadi", "bahadurgarh"]):
        region_map[z] = "Peripheral"
    else:
        region_map[z] = "Peripheral"

df["Region"] = df["Pickup Location"].map(region_map)

# ---------------------------------------------------
# 5. Aggregate emissions by region
# ---------------------------------------------------
region_emissions = df.groupby("Region").agg(
    Baseline_Emissions_kg=("Emission_kg", "sum"),
    Optimized_Emissions_kg=("Emission_Optimized_kg", "sum")
).reset_index()

region_emissions["Reduction_%"] = (
    (1 - (region_emissions["Optimized_Emissions_kg"] / region_emissions["Baseline_Emissions_kg"])) * 100
)

# ---------------------------------------------------
# 6. Plot results
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
bar_width = 0.35
regions = region_emissions["Region"]
x = range(len(regions))

plt.bar(x, region_emissions["Baseline_Emissions_kg"], width=bar_width, color="#f4a261", label="Before Optimization")
plt.bar(
    [i + bar_width for i in x],
    region_emissions["Optimized_Emissions_kg"],
    width=bar_width,
    color="#2a9d8f",
    label="After Optimization"
)

# Annotate reduction percentages
for i, val in enumerate(region_emissions["Reduction_%"]):
    plt.text(i + 0.2, region_emissions["Baseline_Emissions_kg"][i] * 0.95,
             f"-{val:.1f}%", ha="center", fontsize=10, color="black")

plt.xticks([i + bar_width / 2 for i in x], regions)
plt.ylabel("Total CO₂ Emissions (kg)", fontsize=12)
plt.title("")
plt.legend()
plt.tight_layout()

# ---------------------------------------------------
# 7. Save output
# ---------------------------------------------------
save_path = os.path.join(graphs_path, "Figure16_Emission_Reduction.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure16_Emission_Reduction.png saved successfully at:\n{save_path}")