# Uber-Project
Uber Ride-Hailing: Forecasting Demand and Revenue for Efficiency in Delhi NC

Week 1 — Data Intake & Quality Audit
	•	Set up project repo/conda env; lock Python libs (pandas, numpy, scikit-learn, statsmodels, xgboost, pmdarima).
	•	Load ncr_ride_bookings.csv, validate schema (types, unique IDs, ranges).
	•	Audit missing values, duplicates, and obvious outliers; document data dictionary.
	•	Parse Date/Time; create Hour, Weekday, Month.
	•	Deliverable: Data audit memo + checklist of issues to fix.

⸻

Week 2 — Cleaning & Core EDA
	•	Handle missing values (event columns → 0/“Not Applicable”; numeric → KNN/median).
	•	Remove/clip extreme outliers (distance/value beyond logical bounds).
	•	Initial EDA: distributions, boxplots, correlations (numeric subset).
	•	Time plots: daily rides + 7/28-day rolling averages.
	•	Deliverable: EDA notebook with baseline plots + cleaned ncr_ride_bookings_cleaned.csv.

⸻

Week 3 — Feature Engineering (Time Series + Revenue)
	•	Build lag features for demand (lag_1, lag_7, lag_14) and rolling means (roll_7, roll_14, roll_28).
	•	Create calendar features: hour, weekday (ordered), month; optional holiday flags.
	•	Encode Vehicle Type; standardize/scale where needed.
	•	Deliverable: Feature set preview table + data pipeline utilities.

⸻

Week 4 — Baseline Models
	•	Demand baseline: ARIMA (grid a few (p,d,q)); hold-out last 30 days.
	•	Revenue baseline: Linear Regression with distance + time + vehicle type.
	•	Record MAE/RMSE; plot actual vs predicted.
	•	Deliverable: Baseline metrics table + residual diagnostics.

⸻

Week 5 — Advanced Demand Models (Trees/Boosting)
	•	Train Random Forest & XGBoost on lagged/rolling features for daily rides.
	•	Hyperparameter search (coarse): n_estimators, max_depth, learning_rate.
	•	Compare to ARIMA; importance plots; SHAP (if feasible).
	•	Deliverable: Demand model comparison (ARIMA vs RF vs XGB) + forecast plots.

⸻

Week 6 — Advanced Revenue Models
	•	Train RF/XGBoost for booking value using Ride Distance, Hour, Weekday, Vehicle Type, Avg VTAT/CTAT.
	•	Evaluate MAE/RMSE; partial dependence for distance and hour.
	•	Segment revenue by vehicle type & time windows; sanity plots.
	•	Deliverable: Revenue model report + feature importance visuals.

⸻

Week 7 — Operational Insights
	•	Peak load matrix: Weekday × Hour heatmap; top-10 peak slots.
	•	Cancellation analysis: “No Driver Found” by hour/weekday/vehicle type.
	•	Translate to actions: staffing windows, fleet mix, pricing cues.
	•	Deliverable: Ops brief (1–2 pages) with actionable recommendations.

⸻

Week 8 — Synthesis & Writing (Methods/Results)
	•	Integrate findings into unified framework (Forecasting → Revenue → Operations).
	•	Draft Methods (data, cleaning, features, model specs, metrics).
	•	Draft Results (tables for metrics; figs for forecasts & importances).
	•	Deliverable: Draft Methods + Results sections (with figures/tables).

⸻

Week 9 — Discussion, Limitations, and Related Work
	•	Discuss accuracy vs interpretability; generalizability to other cities.
	•	Limitations: no customer history, no external data (weather/traffic).
	•	Connect to literature review themes; outline future work (spatial features, external covariates).
	•	Deliverable: Draft Discussion + Limitations; finalize References/APA.

⸻

Week 10 — Finalization & Defense Prep
	•	Full manuscript polish (APA, captions, ToC); proof charts and numbers.
	•	Build slide deck: problem → data → models → results → ops insights.
	•	Rehearse defense; prep anticipated Q&A (methods, ethics, bias).
	•	Deliverable: Final thesis PDF + presentation slides.

⸻
