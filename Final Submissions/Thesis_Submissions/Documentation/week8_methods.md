# Week 8 — Methods (Synthesis)

This script aggregates prior outputs (Week 5 demand, Week 6 revenue, Week 7 operations) and formats
them for the thesis. It auto-detects metric columns (e.g., cv_mean_MAE, MAE, RMSE) to accommodate
week-to-week schema differences and produces standardized tables and figures.

**Inputs**
- Demand: week8_tables_demand.csv
- Revenue: week8_tables_revenue.csv
- Ops: week8_tables_ops_top10.csv plus Week 7 heatmaps in /reports

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