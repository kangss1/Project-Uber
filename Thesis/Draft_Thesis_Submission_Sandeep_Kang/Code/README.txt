==============================================================
INFO-I 492 Senior Thesis Project
Author: Sandeep Kang
Course: INFO-I 492 – Senior Thesis
Institution: Indiana University Bloomington
Term: Fall 2025
==============================================================

Project Title:
Machine Learning-Based Forecasting of Ride-Hailing Demand, Revenue, and Operations: A Case Study of Delhi NCR, India

==============================================================
1. Overview
==============================================================
This folder contains all Python scripts used for data processing. 
feature engineering, model training, evaluation, and visualization 
for the Senior Thesis project. Each “week” script corresponds to a 
research milestone, while each “figure” script reproduces visuals 
cited in the thesis (Figures 3–16).

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
    Produces operational metrics such as average fare per day of week 
    and visualizes patterns in ride pricing.
Required Input:
    ncr_ride_featured.csv
Output:
    week7_avg_fare_by_day.png

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

==============================================================
4. Figure Diagram Scripts
==============================================================

Figure3_diagram.py
    Generates Figure 3 – Fare vs Distance (Peak vs Off-Peak)
    Input: ncr_ride_featured.csv
    Output: Figure3_diagram.png

figure5_diagram.py
    Generates Figure 5 – Fare Distribution by Day of Week
    Input: ncr_ride_featured.csv
    Output: figure5_diagram.png

Figure10_diagram.py
    Generates Figure 10 – Combined Performance and Feature Importance
    Input:
        week5_demand_rf_tuned.png
        week5_demand_xgb_tuned.png
        week5_demand_xgb_importance.png
        week5_demand_xgb_shap_summary.png
    Output:
        Figure10_diagram.png

figure11_diagram.py
    Generates Figure 11 – Average Fare by Payment Method
    Input: ncr_ride_featured.csv
    Output: figure11_diagram.png

figure12_diagram.py
    Generates Figure 12 – Kernel Density of Fare per KM
    Input: ncr_ride_featured.csv
    Output: figure12_diagram.png

figure13_diagram.py
    Generates Figure 13 – Average Tip Ratio by Peak Hour
    Input: ncr_ride_featured.csv
    Output: figure13_diagram.png

figure14_diagram.py
    Generates Figure 14 – Fare Distribution (Peak vs Off-Peak)
    Input: ncr_ride_featured.csv
    Output: figure14_diagram.png

Figure15_diagram.py
    Generates Figure 15 – Correlation Matrix of Key Variables
    Input: ncr_ride_featured.csv
    Output: Figure15_diagram.png

figure16_diagram.py
    Generates Figure 16 – Ride Frequency by Day of Week
    Input: ncr_ride_featured.csv
    Output: figure16_diagram.png

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
3. Figures will be saved automatically as PNG files in the same directory.
4. To regenerate any figure individually:
       python figure13_diagram.py

==============================================================
6. Notes
==============================================================
- File paths assume a flat structure within the Code folder.
- All scripts are well-documented with clear inline comments.
- Figures correspond directly to visuals cited in the thesis 
  (Figures 3–16).
- Outputs are reproducible and saved automatically upon execution.
- For evaluation, ensure the input CSVs exist in the same directory.

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