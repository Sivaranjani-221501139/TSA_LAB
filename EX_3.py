import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ADF Test Function
def adf_test(time_series):
    result = adfuller(time_series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    return result[1] <= 0.05  


def make_stationary(time_series):
    return time_series.diff().dropna()

# Load data and preprocess
time_series = pd.read_csv(r"c:\Users\Lenovo\Downloads\dataset\weatherHistory.csv")['Temperature (C)'].dropna()
time_series = pd.to_numeric(time_series, errors='coerce')

# Check stationarity and process accordingly
if not adf_test(time_series):
    print("\nDifferencing to make it stationary:")
    stationary_series = make_stationary(time_series)
    adf_test(stationary_series)
    plt.plot(stationary_series)
    plt.title('Differenced (Stationary) Time Series')
else:
    print("\nThe time series is already stationary.")
    plt.plot(time_series)
    plt.title('Original Stationary Time Series')

plt.figure(figsize=(10, 6))
plt.show()
