#%% [markdown]
# ============================================
# Week 8 — Synthesis & Writing (Methods + Results Integration)
# File: week8_synthesis_methods_results.py
# Depends on: Outputs from Weeks 5–7
#   -> Week 5: Demand forecasting metrics, model tuning results, and leaderboard
#   -> Week 6: Revenue prediction models, feature importance analysis, and diagnostics
#   -> Week 7: Operational insights including peak load matrices and cancellation heatmaps
# ============================================
#
# Purpose
#   The goal of this script is to synthesize all prior analytical work into a unified,
#   thesis-ready representation of the research pipeline. It bridges forecasting,
#   revenue modeling, and operational analytics to form the foundation for the final
#   Methods and Results sections of the capstone paper.
#
#   Specifically, this script:
#     1) Loads and harmonizes quantitative outputs from Weeks 5–7
#     2) Automatically detects key performance metrics (MAE, RMSE, cv_mean_MAE, etc.)
#     3) Generates consolidated result tables and comparative visualizations
#     4) Produces markdown drafts for the thesis Methods and Results chapters
#     5) Ensures consistent formatting, figure labeling, and traceable data lineage
#
# Outputs
#   Data Tables:
#     data/processed/week8_tables_demand.csv        -> Aggregated demand model metrics
#     data/processed/week8_tables_revenue.csv       -> Revenue model performance summary
#     data/processed/week8_tables_ops_top10.csv     -> Operational top-10 summary (Week 7)
#
#   Figures:
#     reports/week8_demand_metrics_bar.png          -> Demand model performance (min MAE)
#     reports/week8_revenue_metrics_bar.png         -> Revenue model comparison
#     reports/week8_ops_thumbgrid.png               -> Thumbnail grid of key operational heatmaps
#
#   Text Artifacts:
#     reports/week8_methods.md                      -> Draft of thesis "Methods" section
#     reports/week8_results.md                      -> Draft of thesis "Results" section
#
# Summary
#   This synthesis layer connects analytical modeling (Weeks 5–6) with operational insight (Week 7),
#   aligning all quantitative findings into a cohesive, reproducible structure. The resulting visual
#   and markdown outputs will directly feed into the final thesis (Week 9), ensuring consistent
#   narrative flow between technical methods, empirical results, and business interpretation.
# ============================================
#
# Data Lineage Summary (Week 1–8)
#   Week 1 -> Data Audit & Schema Validation
#       Input: raw ride-booking dataset (ncr_ride_bookings.csv)
#       Output: audit summary, completeness and validity checks
#
#   Week 2 -> Data Cleaning & Exploratory Data Analysis
#       Input: audited data
#       Output: cleaned dataset (ncr_ride_bookings_cleaned.csv), summary plots, correlations
#
#   Week 3 -> Feature Engineering
#       Input: cleaned data
#       Output: enhanced dataset with temporal, spatial, and categorical features
#
#   Week 4 -> Baseline Modeling
#       Input: feature-engineered data
#       Output: baseline ARIMA and regression benchmarks for forecasting
#
#   Week 5 -> Advanced Demand Forecasting
#       Input: feature-engineered data
#       Output: optimized Random Forest and XGBoost models, leaderboard metrics, error plots
#
#   Week 6 -> Revenue Modeling & Feature Analysis
#       Input: model-ready revenue dataset
#       Output: revenue models (RF, XGB), SHAP plots, feature importance, and performance metrics
#
#   Week 7 -> Operational Insights
#       Input: merged booking and model outputs
#       Output: peak load heatmaps, NDF cancellation analysis, fulfillment efficiency maps
#
#   Week 8 -> Synthesis & Writing (Methods + Results)
#       Input: aggregated metrics, figures, and Week 7 summaries
#       Output: integrated tables, comparative figures, and thesis-ready markdown drafts
#
# ============================================
#%% [markdown]
# ============================================
# Week 8 — Synthesis & Writing (Methods + Results Integration)
# File: week8_synthesis_methods_results.py
# Depends on: Outputs from Weeks 5–7
#   -> Week 5: Demand forecasting metrics, model tuning results, and leaderboard
#   -> Week 6: Revenue prediction models, feature importance analysis, and diagnostics
#   -> Week 7: Operational insights including peak load matrices and cancellation heatmaps
# ============================================
#
# Purpose
#   The goal of this script is to synthesize all prior analytical work into a unified,
#   thesis-ready representation of the research pipeline. It bridges forecasting,
#   revenue modeling, and operational analytics to form the foundation for the final
#   Methods and Results sections of the capstone paper.
#
#   Specifically, this script:
#     1) Loads and harmonizes quantitative outputs from Weeks 5–7
#     2) Automatically detects key performance metrics (MAE, RMSE, cv_mean_MAE, etc.)
#     3) Generates consolidated result tables and comparative visualizations
#     4) Produces markdown drafts for the thesis Methods and Results chapters
#     5) Ensures consistent formatting, figure labeling, and traceable data lineage
#
# Outputs
#   Data Tables:
#     data/processed/week8_tables_demand.csv        -> Aggregated demand model metrics
#     data/processed/week8_tables_revenue.csv       -> Revenue model performance summary
#     data/processed/week8_tables_ops_top10.csv     -> Operational top-10 summary (Week 7)
#
#   Figures:
#     reports/week8_demand_metrics_bar.png          -> Demand model performance (min MAE)
#     reports/week8_revenue_metrics_bar.png         -> Revenue model comparison
#     reports/week8_ops_thumbgrid.png               -> Thumbnail grid of key operational heatmaps
#
#   Text Artifacts:
#     reports/week8_methods.md                      -> Draft of thesis "Methods" section
#     reports/week8_results.md                      -> Draft of thesis "Results" section
#
# Data Lineage Summary (Week 1–8)
#   Week 1 -> audit, Week 2 -> cleaning/EDA, Week 3 -> features, Week 4 -> baselines,
#   Week 5 -> demand models, Week 6 -> revenue models, Week 7 -> ops insights,
#   Week 8 -> this synthesis layer for Methods + Results.
# ============================================

#%% 0) Imports & Paths
from pathlib import Path
from textwrap import dedent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project roots and folders
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(".").resolve()

DATA_PROC = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
for d in (DATA_PROC, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _try_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _pick_metric_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None
    cl = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cl.get(cand.lower())
        if c:
            return c
    # fuzzy fallback
    for c in df.columns:
        lc = c.lower()
        if "mae" in lc or "rmse" in lc:
            return c
    return None

def _coerce(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_model_col(df: pd.DataFrame) -> pd.DataFrame:
    if "model" in df.columns:
        return df
    for cand in ["Model", "estimator", "name", "algo"]:
        if cand in df.columns:
            return df.rename(columns={cand: "model"})
    # fabricate if nothing is present
    df = df.copy()
    df["model"] = [f"Model_{i}" for i in range(len(df))]
    return df

#%% 
# 1) Load artifacts from Weeks 5–7 (with robust fallbacks)

# Demand (Week 5)
# Search both /data/processed and /reports to accommodate project structure
demand_sources = [
    DATA_PROC / "week8_tables_demand.csv",
    DATA_PROC / "week5_demand_leaderboard.csv",
    DATA_PROC / "week5_demand_holdout_metrics.csv",
    REPORTS / "week5_demand_leaderboard.csv",
    REPORTS / "week5_demand_holdout_metrics.csv",
]
demand_df = pd.DataFrame()
demand_source = "N/A"
for p in demand_sources:
    t = _try_read_csv(p)
    if not t.empty:
        demand_df = t.copy()
        demand_source = p.name
        print(f"Loaded demand data from {p}")
        break

# Revenue (Week 6)
revenue_sources = [
    DATA_PROC / "week8_tables_revenue.csv",
    DATA_PROC / "week6_revenue_metrics.csv",
    REPORTS / "week6_revenue_metrics.csv",     # added fallback path
]
revenue_df = pd.DataFrame()
revenue_source = "N/A"
for p in revenue_sources:
    t = _try_read_csv(p)
    if not t.empty:
        revenue_df = t.copy()
        revenue_source = p.name
        break

# Ops Top10 (Week 7)
ops_top10 = _try_read_csv(DATA_PROC / "week8_tables_ops_top10.csv")
if ops_top10.empty:
    ops_top10 = _try_read_csv(REPORTS / "week8_tables_ops_top10.csv")
    
ops_source = "week8_tables_ops_top10.csv" if not ops_top10.empty else "N/A"

# If not present, try to derive from week7_peak_table.csv (has rides by weekday x hour)
if ops_top10.empty:
    peak_tab = _try_read_csv(DATA_PROC / "week7_peak_table.csv")
    if not peak_tab.empty:
        # melt and take top-10 by rides
        longp = peak_tab.rename(columns={"Weekday_cat":"Weekday"})
        longp = longp.melt(id_vars=[c for c in longp.columns if c.lower().startswith("weekday")],
                           var_name="Hour", value_name="rides")
        longp = longp.dropna(subset=["rides"])
        longp["Hour"] = pd.to_numeric(longp["Hour"], errors="coerce").astype("Int64")
        ops_top10 = longp.sort_values("rides", ascending=False).head(10).copy()
        ops_source = "week7_peak_table.csv"
    else:
        ops_top10 = pd.DataFrame()

# Week 7 images (for thumb grid)
ops_imgs = {
    "peak":        REPORTS / "week7_peak_heatmap.png",
    "cancel":      REPORTS / "week7_cancellation_heatmap.png",
    "efficiency":  REPORTS / "week7_fulfillment_efficiency_heatmap.png",
}
ops_imgs = {k: v for k, v in ops_imgs.items() if v.exists()}

print(f"Loaded sources -> Demand={demand_source}, Revenue={revenue_source}, OpsTop10={ops_source}")

#%% 
# 2) Demand bar chart and data consolidation (robust detection)
print("\n#2) Building Week 8 demand tables and bar plot")

demand_plot_png = REPORTS / "week8_demand_metrics_bar.png"
demand_out = DATA_PROC / "week8_tables_demand.csv"

if demand_df.empty:
    print("No demand CSV detected — skipping demand metrics.")
else:
    demand_df = _ensure_model_col(demand_df)
    metric_col = _pick_metric_col(
        demand_df,
        ["cv_mean_MAE", "mean_cv_mae", "mae", "MAE", "RMSE", "rmse", "score", "val_mae"]
    )
    if metric_col is None:
        # fallback to first numeric column if no match
        num_cols = [c for c in demand_df.columns if pd.api.types.is_numeric_dtype(demand_df[c])]
        if num_cols:
            metric_col = num_cols[0]
            print(f"No named metric detected — using fallback column: {metric_col}")
        else:
            print("No numeric columns to plot — skipping demand output.")
            metric_col = None

    if metric_col:
        demand_df = _coerce(demand_df, [metric_col])
        subset = demand_df.dropna(subset=[metric_col])
        if subset.empty:
            print(f"All {metric_col} values are NaN — skipping demand chart.")
        else:
            agg = (
                subset.groupby("model", as_index=False)[metric_col]
                .min()
                .sort_values(metric_col, ascending=True)
            )

            # Write table
            keep_cols = [
                c for c in demand_df.columns
                if any(k in c.lower() for k in ["model", "mae", "rmse", "cv", "score"])
            ]
            week8_demand = demand_df[keep_cols].copy()
            week8_demand["Source"] = demand_source
            week8_demand.to_csv(demand_out, index=False)
            print(f"Wrote -> {demand_out}")

            # Bar plot
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.bar(agg["model"].astype(str), agg[metric_col].values, color="skyblue", edgecolor="black")
            ax.set_title("Week 5 — Demand Models (Lower is Better)")
            ax.set_ylabel(metric_col)
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
                tick.set_ha("right")
            plt.tight_layout()
            plt.savefig(demand_plot_png, dpi=150)
            plt.close(fig)
            print(f"Saved -> {demand_plot_png}")
            
#%% 
# 3) Revenue bar chart and data consolidation

revenue_plot_png = REPORTS / "week8_revenue_metrics_bar.png"
if not revenue_df.empty:
    rev_metric_col = _pick_metric_col(revenue_df, ["MAE", "RMSE"])
    if rev_metric_col:
        revenue_df = _coerce(revenue_df, [rev_metric_col])
        name_col = "Model" if "Model" in revenue_df.columns else ("model" if "model" in revenue_df.columns else None)
        if name_col is None:
            name_col = "_name"
            revenue_df[name_col] = [f"Model {i}" for i in range(len(revenue_df))]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(revenue_df[name_col].astype(str), revenue_df[rev_metric_col].values)
        ax.set_title("Week 6 — Revenue Models (MAE on holdout)")
        ax.set_ylabel(rev_metric_col)
        for t in ax.get_xticklabels():
            t.set_rotation(15); t.set_ha("right")
        plt.tight_layout()
        plt.savefig(revenue_plot_png, dpi=150)
        plt.close(fig)
        print(f"Saved revenue bar -> {revenue_plot_png}")
    else:
        print("Revenue bar skipped: could not detect MAE/RMSE column.")
else:
    print("Revenue bar skipped: no revenue table available.")

# Write consolidated Week 8 revenue table
revenue_out = DATA_PROC / "week8_tables_revenue.csv"
if not revenue_df.empty:
    keep_cols = [
        c for c in revenue_df.columns
        if any(k in c.lower() for k in ["model", "mae", "rmse", "metric", "score"])
    ]
    week8_revenue = revenue_df[keep_cols].copy()
    week8_revenue["Source"] = revenue_source
    week8_revenue.to_csv(revenue_out, index=False)
    print(f"Wrote -> {revenue_out}")
else:
    print("No revenue data to write.")

#%% 
# 4) Ops thumb grid and top-10 writing

thumb_png = REPORTS / "week8_ops_thumbgrid.png"
if len(ops_imgs) >= 2:
    fig, axes = plt.subplots(1, len(ops_imgs), figsize=(16, 4.5))
    if len(ops_imgs) == 1:
        axes = [axes]
    for ax, (k, p) in zip(axes, ops_imgs.items()):
        img = plt.imread(str(p))
        ax.imshow(img); ax.axis("off"); ax.set_title(p.stem, fontsize=12)
    plt.tight_layout(); plt.savefig(thumb_png, dpi=150); plt.close(fig)
    print(f"Saved ops thumb grid -> {thumb_png}")
else:
    print("Thumb grid skipped: need at least two Week 7 figures.")

# Write consolidated Week 8 ops top-10
ops_out = DATA_PROC / "week8_tables_ops_top10.csv"
if not ops_top10.empty:
    ops_top10 = ops_top10.copy()
    ops_top10["Source"] = ops_source
    ops_top10.to_csv(ops_out, index=False)
    print(f"Wrote -> {ops_out}")
else:
    print("No ops top-10 data to write.")

#%% 
# 5) Methods + Results markdown

methods_md = REPORTS / "week8_methods.md"
results_md = REPORTS / "week8_results.md"

methods_txt = dedent(f"""
# Week 8 — Methods (Synthesis)

This script aggregates prior outputs (Week 5 demand, Week 6 revenue, Week 7 operations) and formats
them for the thesis. It auto-detects metric columns (e.g., cv_mean_MAE, MAE, RMSE) to accommodate
week-to-week schema differences and produces standardized tables and figures.

**Inputs**
- Demand: {demand_source}
- Revenue: {revenue_source}
- Ops: {ops_source} plus Week 7 heatmaps in /reports

**Processing**
- Demand: group by model and plot the best (minimum) detected error metric. Save a cleaned table with key metrics and parameters.
- Revenue: plot MAE/RMSE by model on final holdout and save a cleaned summary table.
- Operations: write a top-10 demand window table and compose a thumbnail grid from Week 7 heatmaps.

**Outputs**
- data/processed/week8_tables_demand.csv
- data/processed/week8_tables_revenue.csv
- data/processed/week8_tables_ops_top10.csv
- reports/week8_demand_metrics_bar.png
- reports/week8_revenue_metrics_bar.png
- reports/week8_ops_thumbgrid.png
""").strip()

with open(methods_md, "w", encoding="utf-8") as f:
    f.write(methods_txt)
print(f"Wrote: {methods_md}")

def _tbl_info(df: pd.DataFrame, label: str) -> str:
    return f"- {label}: {len(df):,} rows" if not df.empty else f"- {label}: not found"

results_lines = [
    "# Week 8 — Results (Synthesis)",
    _tbl_info(_try_read_csv(demand_out),  "Demand table"),
    _tbl_info(_try_read_csv(revenue_out), "Revenue table"),
    _tbl_info(_try_read_csv(ops_out),     "Ops Top-10 table"),
    "",
]
if demand_plot_png.exists():
    results_lines.append(f"![]({demand_plot_png.name})")
if revenue_plot_png.exists():
    results_lines.append(f"![]({revenue_plot_png.name})")
if thumb_png.exists():
    results_lines.append(f"![]({thumb_png.name})")

with open(results_md, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
print(f"Wrote: {results_md}")

print("\nWeek 8 synthesis complete.")
#%%