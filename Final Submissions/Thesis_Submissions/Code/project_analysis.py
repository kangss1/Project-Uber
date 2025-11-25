"""
project_analysis.py
--------------------------------------
This script scans all weekly project Python files (week1–week9)
to find:
 - which files each script reads or writes
 - what ML models and metrics are used

It then saves:
 1. script_summary.json        (detailed info)
 2. project_summary.txt        (easy-to-read summary)
 3. summary_table.txt          (formatted summary table)

All results are stored inside:
   /analysis_results

Author: Sandeep K.
Course: INFO-I 492 Senior Thesis
"""

#%%
# 1) Setup project path and prepare output folder
import os, re, json

# Path to the main Project-Uber folder
project_path = "/Users/sandeepk/Library/Mobile Documents/com~apple~CloudDocs/IUPUI/Senior Thesis INFO-I 492/Project-Uber"

# Create results folder (if not already exists)
results_path = os.path.join(project_path, "analysis_results")
os.makedirs(results_path, exist_ok=True)

# Find all week Python scripts (week1 to week9)
week_files = [f for f in os.listdir(project_path) if f.startswith("week") and f.endswith(".py")]
week_files.sort()

#%%
# 2) Loop through all weekly scripts and extract key information
all_data = {}

for file in week_files:
    full_path = os.path.join(project_path, file)
    print(f"Analyzing {file}...")

    info = {"reads": [], "writes": [], "models": [], "metrics": []}

    # Read file contents
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
    except:
        print(f"Could not read {file}")
        continue

    #%%
    # 3) Detect files being read or written (CSV operations)
    info["reads"] = re.findall(r'read_csv\([^)]*[\'"]([^\'"]+)[\'"]', code)
    info["writes"] = re.findall(r'to_csv\([^)]*[\'"]([^\'"]+)[\'"]', code)

    #%%
    # 4) Detect ML models and evaluation metrics
    model_keywords = ["RandomForest", "XGB", "GradientBoost", "LinearRegression", "DecisionTree"]
    metric_keywords = ["rmse", "mae", "r2", "accuracy", "f1"]

    for m in model_keywords:
        if m.lower() in code.lower():
            info["models"].append(m)
    for m in metric_keywords:
        if m.lower() in code.lower():
            info["metrics"].append(m)

    # Save data for this script
    all_data[file] = info

#%%
# 5) Save structured results as JSON
json_path = os.path.join(results_path, "script_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

#%%
# 6) Create readable text summary (project_summary.txt)
summary_lines = ["PROJECT SUMMARY MAP (Week1–Week9)\n"]
for f, info in all_data.items():
    summary_lines.append(f"\n{f}:")
    if info["reads"]:
        summary_lines.append("  Reads: " + ", ".join(info["reads"]))
    if info["writes"]:
        summary_lines.append("  Writes: " + ", ".join(info["writes"]))
    if info["models"]:
        summary_lines.append("  Models: " + ", ".join(info["models"]))
    if info["metrics"]:
        summary_lines.append("  Metrics: " + ", ".join(info["metrics"]))

summary_path = os.path.join(results_path, "project_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

#%%
# 7) Print summary table in terminal
print("\n--- Summary Table ---")
print(f"{'File Name':35s} | {'Models':25s} | {'Metrics'}")
print("-" * 75)

table_lines = []
for f, info in all_data.items():
    models = ", ".join(info["models"]) if info["models"] else "-"
    metrics = ", ".join(info["metrics"]) if info["metrics"] else "-"
    print(f"{f:35s} | {models:25s} | {metrics}")
    table_lines.append(f"{f:35s} | {models:25s} | {metrics}")

# Save the printed table to file
table_path = os.path.join(results_path, "summary_table.txt")
with open(table_path, "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines))

#%%
# 8) Wrap up and show file locations
print("\nDone! Results saved inside:")
print(results_path)
print(" - script_summary.json")
print(" - project_summary.txt")
print(" - summary_table.txt")