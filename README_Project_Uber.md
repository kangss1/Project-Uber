
# Project Uber: Machine Learning-Based Ride-Hailing Optimization (Delhi NCR)

## Overview
Project Uber is a comprehensive research and analytics project focused on optimizing ride-hailing operations within the Delhi NCR region using machine learning. 
The study integrates forecasting, revenue modeling, and operational efficiency with fairness and sustainability evaluation. 
It leverages real-world ride data to identify demand trends, improve fleet allocation, and reduce emissions while maintaining service reliability.

---

## Repository Structure
```
Project-Uber/
│
├── Final Submissions/             # Complete research package
│   ├── Thesis_Submissions/        # Core working directory with all essential research materials
│   │   ├── Code/                  # Python scripts for data analysis, modeling, and figure generation
│   │   ├── Documentation/         # Weekly reports, analyses, and markdown summaries
│   │   ├── Graphs/                # Final figures used in the paper and presentation
│   │   ├── Graphics/              # Supporting visuals and conceptual diagrams
│   │   └── Illustrations/         # Drawio and presentation visuals
│   ├── Project_Uber_Thesis_Paper.docx
│   └── Machine_Learning_Based_Forecasting_RideHailing_DelhiNCR.pptx
│
├── data/                          # Processed and interim datasets used for training and validation
│   ├── interim/
│   └── processed/
│
├── models/                        # Trained models and parameters for forecasting and revenue prediction
│   ├── week5_best_rf.pkl
│   ├── week5_best_xgb.json
│   ├── week6_rf_revenue.pkl
│   └── week6_xgb_revenue.json
│
├── reports/                       # Week-by-week documentation of analytical stages and findings
│   ├── week1_data_audit.md
│   ├── week5_demand_summary.md
│   ├── week6_revenue_summary.md
│   ├── week7_ops_brief.md
│   ├── week8_results.md
│   └── week9_discussion.md
│
├── analysis_results/              # Final summarized outputs of model performance and analysis tables
│   ├── project_summary.txt
│   └── summary_table.txt
│
├── Drawio Files/                  # Editable conceptual diagrams for figures and workflows
├── requirements.txt               # Python dependencies
└── README.md                      # Repository guide and navigation document
```

---

## Environment Setup
To reproduce this project locally:

```bash
# Clone the repository
git clone https://github.com/kangss1/Project-Uber.git
cd Project-Uber

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate    # for Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- shap
- statsmodels
- pmdarima

---

## Weekly Workflow and Reproduction Guide

### Week 1 — Data Intake and Quality Audit
**Script:** `week1_data_audit.py`  
- Load raw dataset and validate structure.  
- Detect missing values, duplicates, and anomalies.  
- Generate a structured data audit report.  
- **Deliverables:** `week1_data_audit.md`, quality checklist.

### Week 2 — Data Cleaning and EDA
**Script:** `week2_cleaning_eda.py`  
- Clean missing and invalid records.  
- Visualize trends, correlations, and distribution patterns.  
- Save cleaned dataset `ncr_ride_bookings_cleaned.csv`.  
- **Deliverables:** Cleaned dataset and EDA visualizations.

### Week 3 — Feature Engineering
**Script:** `week3_feature_engineering.py`  
- Create time-based and rolling features.  
- Encode categorical and scale numeric features.  
- Generate enriched dataset for modeling.  
- **Deliverables:** Feature-engineered dataset.

### Week 4 — Baseline Models
**Script:** `week4_baselines.py`  
- Build ARIMA and Linear Regression baselines.  
- Evaluate MAE/RMSE and residuals.  
- Establish reference models.  
- **Deliverables:** `week4_baselines_summary.md` and residual plots.

### Week 5 — Advanced Demand Forecasting
**Script:** `week5_advanced.py`  
- Train Random Forest and XGBoost models for demand prediction.  
- Perform grid search tuning and model comparison.  
- Generate SHAP and feature importance plots.  
- **Deliverables:** Best model artifacts and demand forecasts.

### Week 6 — Advanced Revenue Modeling
**Script:** `week6_revenue_advanced.py`  
- Build Random Forest and XGBoost revenue prediction models.  
- Analyze feature impacts using SHAP and PDP plots.  
- Segment results by vehicle type and time of day.  
- **Deliverables:** Revenue model summary and visualizations.

### Week 7 — Operational Insights
**Script:** `week7_operational_insights.py`  
- Generate demand and cancellation heatmaps.  
- Examine peak hour performance and service reliability.  
- Relate findings to fleet allocation efficiency.  
- **Deliverables:** Operational performance heatmaps and brief.

### Week 8 — Method Integration and Results
**Script:** `week8_synthesis_methods_results.py`  
- Integrate results from demand, revenue, and ops analysis.  
- Prepare thesis Methods and Results sections.  
- **Deliverables:** Consolidated results tables and draft figures.

### Week 9 — Discussion and Reflection
**Script:** `week9_interpretation_reflection.py`  
- Interpret findings; identify key limitations.  
- Discuss fairness and sustainability implications.  
- **Deliverables:** `week9_discussion.md` and `week9_limitations.md`.

### Week 10 — Finalization and Presentation
**Files:** Thesis Paper & Presentation  
- Compile final thesis and slides.  
- Review figures and confirm metrics.  
- Prepare oral defense materials.  
- **Deliverables:** Final thesis and presentation deck.

---

## Project Scope
- **Demand Forecasting:** Predict hourly and daily ride demand.  
- **Revenue Modeling:** Understand fare variability and contributing factors.  
- **Operational Efficiency:** Analyze fleet utilization and cancellations.  
- **Sustainability Analysis:** Evaluate emissions and fairness across regions.

---

## Key Outcomes
- **Forecast Accuracy:** Ensemble models achieved the lowest MAE and RMSE.  
- **Operational Efficiency:** Average service fulfillment ~93% with fewer peak-hour cancellations.  
- **Fairness:** 12% reduction in regional trip allocation disparities.  
- **Sustainability:** 9% reduction in total fleet emissions.

---

## Citation
Kang, S. (2025). *Machine Learning-Based Forecasting and Optimization of Ride-Hailing Systems in Delhi NCR.* Project Uber Research, 2025.

---

## Contact
**Author:** Sandeep Kang  
**LinkedIn:** [linkedin.com/in/sandeepkang](https://www.linkedin.com/in/sskang79)  
**Year:** 2025  

---
_Project Uber — A data-driven framework for forecasting, optimization, and sustainability in ride-hailing systems._
