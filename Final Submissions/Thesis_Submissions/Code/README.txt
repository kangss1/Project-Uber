==============================================================
INFO-I 492 Senior Thesis Project
Author: Sandeep Kang
Course: INFO-I 492 – Senior Thesis
Institution: Indiana University Bloomington
Term: Fall 2025
==============================================================

Project Title:
Machine Learning-Based Forecasting of Ride-Hailing Demand, Revenue, and Operations: 
A Case Study of Delhi NCR, India

==============================================================
1. Overview
==============================================================
This folder contains all Python scripts used for data processing, 
feature engineering, model training, evaluation, and visualization 
for the Senior Thesis project. Each “week” script corresponds to a 
research milestone, while each “figure” script reproduces visuals 
cited in the thesis (Figures 3, 5, and 10-16).

All scripts include inline documentation, consistent variable naming, 
and modular structure. They can be executed sequentially or individually, 
provided the required input files exist in the working directory.

==============================================================
2. Dependencies
==============================================================
Developed with:
- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn

==============================================================
3. Weekly Scripts (Main Pipeline)
==============================================================

--------------------------------------------------------------
week1_data_audit.py
--------------------------------------------------------------
Purpose:
    Performs an initial audit of the raw dataset, printing column 
    details, missing values, and summary statistics.
Required Input:
    ncr_ride_bookings.csv
Output:
    week1_audit_summary.csv

--------------------------------------------------------------
week2_cleaning_eda.py
--------------------------------------------------------------
Purpose:
    Cleans and preprocesses the ride-hailing dataset, fills missing 
    values, and produces exploratory visualizations.
Required Input:
    ncr_ride_bookings.csv
Output:
    ncr_ride_bookings_cleaned.csv
    week2_fare_distribution.png
    week2_daywise_bookings.png
    week2_corr_heatmap.png

--------------------------------------------------------------
week3_feature_engineering.py
--------------------------------------------------------------
Purpose:
    Adds analytical features such as fare per kilometer, tip ratio, 
    and categorical encodings for day and payment type.
Required Input:
    ncr_ride_bookings_cleaned.csv
Output:
    ncr_ride_featured.csv

--------------------------------------------------------------
week4_baselines.py
--------------------------------------------------------------
Purpose:
    Establishes a baseline fare prediction model using Linear Regression 
    and evaluates MAE, MSE, and R².
Required Input:
    ncr_ride_featured.csv
Output:
    week4_baseline_predictions.csv

--------------------------------------------------------------
week5_advanced.py
--------------------------------------------------------------
Purpose:
    Trains an ensemble model (Random Forest Regressor) and generates 
    feature importance charts to evaluate model performance.
Required Input:
    ncr_ride_featured.csv
Output:
    week5_rf_importance.png

--------------------------------------------------------------
week6_revenue_advanced.py
--------------------------------------------------------------
Purpose:
    Implements Gradient Boosting for fare prediction and measures 
    out-of-sample accuracy (R² score).
Required Input:
    ncr_ride_featured.csv
Output:
    Console output of R² performance metric.

--------------------------------------------------------------
week7_operational_insights.py
--------------------------------------------------------------
Purpose:
    Produces operational metrics such as average fare per day of week, 
    fulfillment efficiency, and visualizes temporal and spatial demand patterns.
Required Input:
    ncr_ride_featured.csv
Output:
    week7_avg_rides_per_hour.png
    week7_fulfillment_efficiency_heatmap.png
    week7_cancel_by_vehicle.png
    week7_peak_heatmap_annotated.png

--------------------------------------------------------------
week8_synthesis_methods_results.py
--------------------------------------------------------------
Purpose:
    Integrates results from baseline and advanced models, comparing 
    actual vs. predicted values and visualizing model fit.
Required Input:
    ncr_ride_featured.csv
    week4_baseline_predictions.csv
Output:
    week8_scatter.png

--------------------------------------------------------------
week9_interpretation_reflection.py
--------------------------------------------------------------
Purpose:
    Provides narrative interpretation and reflection on model results, 
    summarizing key findings in printed output.
Required Input:
    None
Output:
    Console-printed summary of model performance and insights.

--------------------------------------------------------------
project_analysis.py
--------------------------------------------------------------
Purpose:
    Generates a comprehensive summary of dataset trends and model outputs, 
    integrating metrics from all weekly reports. Outputs key insights and 
    high-level operational performance indicators.
Required Input:
    /reports directory contents
Output:
    project_analysis_summary.csv
    Console summary of key demand, revenue, and operational metrics

==============================================================
4. Figure Diagram Scripts
==============================================================

Figure3_ModelPerformanceComparison.py  
    Input: ncr_ride_featured.csv  
    Output: Figure3_ModelPerformanceComparison.png  

Figure5_Fairness_Sustainability_Governance.py  
    Input: None  
    Output: Figure5_Fairness_Sustainability_Governance.png  

Figure10_Combined_Performance_And_Features.py  
    Generates Figure 10 – Combined Performance and Feature Importance  
    Input:
        week5_demand_rf_tuned.png  
        week5_demand_xgb_tuned.png  
        week5_demand_xgb_importance.png  
        week5_demand_xgb_shap_summary.png  
    Output:
        Figure10_Combined_Performance_And_Features.png  

Figure11_Demand_Patterns_Composite.py  
    Generates Figure 11 – Hourly and Weekly Demand Pattern Heatmap  
    Input: week7_peak_top10.csv  
    Output: Figure11_Demand_Patterns_Composite.png  

Figure12_Revenue_Model_Comparison.py  
    Generates Figure 12 – Fare Prediction and Revenue Model Comparison  
    Input: week5_demand_baselines.csv, week6_revenue_metrics.csv  
    Output: Figure12_Revenue_Model_Comparison.png  

Figure13_Spatial_Surge_Reconstructed.py  
    Generates Figure 13 – Spatial Distribution of Surge Multipliers  
    Input: week7_ops_brief.md  
    Output: Figure13_Spatial_Surge_Reconstructed.png  

Figure14_Ops_Efficiency.py  
    Generates Figure 14 – Hourly Ride Demand, Fulfillment, and Cancellation Rates  
    Input: week7_peak_top10.csv, week7_ops_brief.md  
    Output: Figure14_Operational_Performance_Composite_RealHours.png  

Figure15_diagram.py  
    Generates Figure 15 – Regional Ride Flow Heatmap  
    Input: ncr_ride_featured.csv  
    Output: Figure15_Regional_Ride_Flow.png  

Figure16_Emission_Reduction.py  
    Generates Figure 16 – Emission Reduction Potential and Policy Impact  
    Input: ncr_ride_featured.csv  
    Output: Figure16_Emission_Reduction.png  

==============================================================
5. How to Run the Code
==============================================================
To reproduce the full pipeline:

1. Place all CSV files in the working directory.
2. Run scripts sequentially from Week 1 through Week 9:
       python week1_data_audit.py
       python week2_cleaning_eda.py
       python week3_feature_engineering.py
       ...
       python week9_interpretation_reflection.py
3. Figures will be saved automatically as PNG files in the Graphs directory.
4. To regenerate any figure individually:
       python Figure14_Ops_Efficiency.py
       python Figure15_diagram.py
5. To produce summary analysis:
       python project_analysis.py

==============================================================
6. Notes
==============================================================
- File paths assume a flat structure within the Code folder.
- All scripts are well-documented with clear inline comments.
- Figures correspond directly to visuals cited in the thesis 
  (Figures 3, 5, and 10-16).
- Outputs are reproducible and saved automatically upon execution.
- For evaluation, ensure the input CSVs exist in the same directory.
- Updated Figures 11–14 use verified Week 7 report data 
  from the /reports folder for full data consistency.
- Figure 15 remains in its original format and is unchanged from the prior version.
- Non-Python design files such as .drawio diagrams are excluded.

==============================================================
7. Contact Information
==============================================================
Author: Sandeep Kang  
Email: kangss@iu.edu  
Course: INFO-I 492 – Senior Thesis  
Institution: Indiana University Bloomington  
Advisor: [Instructor Name]  
Term: Fall 2025  

==============================================================
END OF README
==============================================================