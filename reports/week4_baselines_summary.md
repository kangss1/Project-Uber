# Week 4 — Baseline Results

## Demand (Daily rides) — last 30 days holdout
                 Model       MAE      RMSE
               RF_lags 15.625500 20.011240
              XGB_lags 16.703832 21.773344
SARIMA(1,1,1)(1,0,1,7) 16.860809 20.878775
        ARIMA(7, 1, 1) 16.944709 20.946193
        ARIMA(2, 1, 1) 17.003585 21.055253
            Naive-last 22.766667 28.276610
      Naive-seasonal-7 25.566667 30.682786
        ARIMA(1, 1, 0) 27.599508 33.588378

Artifacts:
- week4_demand_baselines.csv
- week4_demand_arima.png (best ARIMA)
- week4_demand_sarima.png
- week4_demand_rf_lags.png
- week4_demand_xgb_lags.png

## Revenue (Booking Value; metrics in original units)
             Model        MAE       RMSE
          RF(logY)   1.178785   3.525208
         XGB(logY)   5.423220  17.150464
 Lasso(logY)+scale 197.527640 672.454749
 Ridge(logY)+scale 197.955813 685.590791
Linear(logY)+scale 198.036233 686.966028

Artifacts:
- week4_revenue_baselines.csv
- week4_revenue_rf_residuals_log.png
- week4_revenue_rf_importances.csv
- week4_revenue_rf_importances_plot.png