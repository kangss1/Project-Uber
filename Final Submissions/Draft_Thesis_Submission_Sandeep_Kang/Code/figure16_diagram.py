import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df = pd.read_csv("ncr_ride_bookings_cleaned.csv")

# --- Filter for completed rides only ---
df = df[df["Booking Status"].str.lower().str.contains("completed")]

# --- Estimate CO₂ emissions based on vehicle type ---
# Emission factors (grams CO₂ per km)
emission_factors = {
    "Go Mini": 130,       # small hatchback (CNG/petrol)
    "Go Sedan": 150,      # larger sedan
    "Premier Sedan": 160,
    "Auto": 70,           # three-wheeler CNG
    "Bike": 40,           # two-wheeler
    "eBike": 0,           # electric
    "Uber XL": 180        # SUV/van
}

df["Emission_Factor"] = df["Vehicle Type"].map(emission_factors).fillna(120)
df["Emission_kg"] = (df["Ride Distance"] * df["Emission_Factor"]) / 1000  # convert g → kg

# --- Simulate optimized scenario ---
# Assume optimization reduces idle time and unnecessary detours by 9% (from your text)
df["Emission_Optimized_kg"] = df["Emission_kg"] * 0.91

# --- Aggregate emissions by region ---
region_map = {}
for z in df["Pickup Location"].unique():
    name = z.lower()
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

region_emissions = df.groupby("Region").agg(
    Baseline_Emissions_kg=("Emission_kg", "sum"),
    Optimized_Emissions_kg=("Emission_Optimized_kg", "sum")
).reset_index()

region_emissions["Reduction_%"] = (
    (1 - (region_emissions["Optimized_Emissions_kg"] / region_emissions["Baseline_Emissions_kg"])) * 100
)

# --- Plot emissions before/after optimization ---
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

# --- Annotate reduction % above bars ---
for i, val in enumerate(region_emissions["Reduction_%"]):
    plt.text(i + 0.2, region_emissions["Baseline_Emissions_kg"][i] * 0.95,
             f"-{val:.1f}%", ha="center", fontsize=10, color="black")

plt.xticks([i + bar_width / 2 for i in x], regions)
plt.ylabel("Total CO₂ Emissions (kg)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("Figure16_Emission_Reduction.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure16_Emission_Reduction.png saved successfully.")