# INFO-I 492 Senior Thesis Project  
**Author:** Sandeep Kang  
**Course:** INFO-I 492 – Senior Thesis  
**Institution:** Indiana University Bloomington  
**Term:** Fall 2025  

---

## Project Title  
**Machine Learning-Based Forecasting of Ride-Hailing Demand, Revenue, and Operations: A Case Study of Delhi NCR, India**

---

## 1. Overview  

This repository contains all Python scripts, datasets, reports, and visual outputs used in the Senior Thesis project. The project investigates machine learning–based forecasting methods for demand, revenue, and operational performance in ride-hailing systems within Delhi NCR.  

Each weekly script corresponds to a stage in the analytical pipeline, from data cleaning to model evaluation and visualization. The figure scripts reproduce the figures cited in the thesis (Figures 3–16) using real data from the `/Reports` and `/Data` folders.  

All scripts are modular, documented, and reproducible. They can be run independently or sequentially, provided the required input files exist.  

---

## 2. Dependencies  

Developed with:  
- Python 3.9+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

To install dependencies:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 3. Weekly Scripts (Main Analytical Pipeline)

| Week | Script | Purpose | Input | Output |
|------|---------|----------|--------|---------|
| **Week 1** | `week1_data_audit.py` | Performs an initial audit of the raw dataset, identifying missing values, data types, and summary statistics. | `ncr_ride_bookings.csv` | `week1_audit_summary.csv` |
| **Week 2** | `week2_cleaning_eda.py` | Cleans and preprocesses the dataset, fills missing values, and generates exploratory data analysis plots. | `ncr_ride_bookings.csv` | `ncr_ride_bookings_cleaned.csv`, `week2_corr_matrix.png` |
| **Week 3** | `week3_feature_engineering.py` | Adds analytical features (fare per km, trip duration bins, time-of-day categories). | `ncr_ride_bookings_cleaned.csv` | `ncr_ride_featured.csv` |
| **Week 4** | `week4_baselines.py` | Establishes baseline fare prediction models using Linear Regression and Random Forest; computes MAE and RMSE. | `ncr_ride_featured.csv` | `week4_baseline_predictions.csv` |
| **Week 5** | `week5_advanced.py` | Trains advanced ensemble models (Random Forest, XGBoost) for demand forecasting; generates feature importance plots. | `ncr_ride_featured.csv` | `week5_rf_importance.png`, `week5_demand_xgb_tuned.png` |
| **Week 6** | `week6_revenue_advanced.py` | Extends modeling to fare and revenue prediction; evaluates R² and RMSE. | `ncr_ride_featured.csv` | `week6_revenue_metrics.csv`, `week6_revenue_shap_bar_rf.png` |
| **Week 7** | `week7_operational_insights.py` | Analyzes operational metrics such as ride fulfillment, cancellations, and surge behavior. | `ncr_ride_featured.csv` | `week7_peak_top10.csv`, `week7_ops_brief.md`, `week7_fulfillment_efficiency_heatmap.png` |
| **Week 8** | `week8_synthesis_methods_results.py` | Consolidates prior analyses and generates combined performance comparisons. | Outputs from Weeks 4–7 | `week8_synthesis_results.png` |
| **Week 9** | `week9_interpretation_reflection.py` | Summarizes and interprets findings; prepares discussion and limitation sections. | None | Console summary output, `week9_discussion.md` |
| **Week 10** | Presentation generation | Creates narrated PowerPoint presentation and thesis document organization. | Thesis draft | `Machine_Learning_Based_Forecasting_RideHailing_DelhiNCR.pptx` |
| **Week 11** | Final figure updates and validation | Rebuilds Figures 11–14 with verified Week 7 data; finalizes reproducibility appendix. | `/Reports`, `/Data` | `Figure11_Hourly_Weekly_Demand_Heatmap.png`, etc. |

---

## 4. Figure Scripts (Visualization and Results Reproduction)

The following Python scripts generate the figures used in the thesis (Figures 3–16).  
Each can be executed independently, provided the corresponding data files are available in the `/Reports` or `/Data` folders.

| Script | Description | Input Files | Output |
|---------|--------------|--------------|---------|
| **figure3_diagram.py** | Generates Figure 3 – Model Performance Comparison showing baseline vs. advanced model accuracy. | `week4_baseline_predictions.csv`, `week5_advanced_metrics.csv` | `Figure3_ModelPerformanceComparison.png` |
| **figure5_diagram.py** | Creates Figure 5 – Conceptual relationship between Fairness, Sustainability, and Governance. | None (static diagram) | `Figure5_Fairness_Sustainability_Governance.png` |
| **figure10_diagram.py** | Produces Figure 10 – Combined model performance and feature importance comparison. | `week5_demand_rf_tuned.png`, `week5_demand_xgb_tuned.png`, `week5_demand_xgb_importance.png` | `Figure10_Combined_Performance_And_Features.png` |
| **figure11_demand_heatmap.py** | Generates Figure 11 – Hourly and Weekly Demand Heatmap based on Week 7 ride data. | `week7_peak_top10.csv`, `week7_peak_heatmap_annotated.png` | `Figure11_Hourly_Weekly_Demand_Heatmap.png` |
| **figure12_revenue_models.py** | Builds Figure 12 – Revenue Model Comparison (Random Forest vs. XGBoost). | `week5_demand_baselines.csv`, `week6_revenue_metrics.csv` | `Figure12_Revenue_Model_Comparison.png` |
| **figure13_spatial_surge.py** | Creates Figure 13 – Spatial Surge Heatmap showing surge distribution across Delhi NCR. | `week7_ops_brief.md`, `week7_peak_heatmap.png` | `Figure13_Spatial_Surge_Reconstructed.png` |
| **figure14_ops_efficiency.py** | Produces Figure 14 – Operational Performance Composite (Hourly Demand + Fulfillment and Cancellation Trends). | `week7_peak_top10.csv`, `week7_ops_brief.md`, `week7_fulfillment_efficiency_heatmap.png` | `Figure14_Operational_Performance_Composite_RealHours.png` |
| **Figure15_diagram.py** | Generates Figure 15 – Regional Ride Flow Heatmap showing cross-regional ride patterns. | `/Reports/regional_flow.csv` | `Figure15_Regional_Ride_Flow.png` |
| **figure16_diagram.py** | Builds Figure 16 – Emission Reduction Visualization summarizing sustainable fleet allocation. | `/Reports/emission_reduction_summary.csv` | `Figure16_Emission_Reduction.png` |

---

## 5. How to Run the Code

To reproduce the full analytical and visualization pipeline:  

1. Place all CSV and image files in the `/Reports` or `/Data` folders.  
2. Run the weekly scripts sequentially:  
   ```bash
   python week1_data_audit.py
   python week2_cleaning_eda.py
   ...
   python week9_interpretation_reflection.py
   ```  
3. To regenerate a specific figure:  
   ```bash
   python figure14_ops_efficiency.py
   ```  
4. All figure outputs will be saved automatically to:  
   ```
   /Thesis/Draft_Thesis_Submission_Sandeep_Kang/Graphs/
   ```

---

## 6. Notes  

- File paths assume a flat project structure (`/Code`, `/Data`, `/Reports`, `/Models`).  
- All figures correspond directly to visuals cited in the thesis (Figures 3–16).  
- Each script includes inline comments and can be executed independently.  
- Outputs are reproducible and automatically saved upon execution.  
- For reproducibility details, see **Appendix A – Reproducibility and Data Availability** in the thesis document.

---

## 7. Contact Information  

**Author:** Sandeep Kang  
**Email:** kangss@iu.edu  
**Course:** INFO-I 492 – Senior Thesis  
**Institution:** Indiana University Bloomington  
**Advisor:** [Instructor Name]  
**Term:** Fall 2025  

---

**End of README**  
