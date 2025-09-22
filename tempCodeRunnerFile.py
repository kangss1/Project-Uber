# --- SARIMA (weekly seasonality) baseline ---
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sar = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,7),
                  enforce_stationarity=False, enforce_invertibility=False)
    sar_fit = sar.fit(disp=False)
    sar_fcst = sar_fit.forecast(steps=N_TEST)

    mae = mean_absolute_error(test.values, sar_fcst.values)
    rmse = root_mean_squared_error(test.values, sar_fcst.values)
    results_demand.append(("SARIMA(1,1,1)(1,0,1,7)", mae, rmse))

    plt.figure()
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index,  test.values,  label="Test")
    plt.plot(test.index,  sar_fcst.values, label="SARIMA Forecast")
    plt.title("Daily Rides — SARIMA(1,1,1)(1,0,1,7)")
    plt.xlabel("Date"); plt.ylabel("Rides"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "week4_demand_sarima.png", dpi=150)
    plt.close()
except Exception as e:
    print("SARIMA failed:", e)