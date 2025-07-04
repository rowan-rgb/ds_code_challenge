#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:18:48 2025

@author: rowandavies
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
pivoted = pd.read_csv("pivoted_weekly_counts.csv", index_col=0)

# Split
n_rows = pivoted.shape[0]
split_idx = int(n_rows * 0.8)
pivoted_train = pivoted.iloc[:split_idx]
pivoted_test = pivoted.iloc[split_idx:]

n_outputs = 4

pred_rows = []
y_test_list = []

# For each test hex
for idx, row in pivoted_test.iterrows():
    series = row.values

    # Use all but last n_outputs weeks for training
    train_series = series[:-n_outputs]
    test_series = series[-n_outputs:]

    y_test_list.append(test_series)

    # Fit SARIMAX
    # You can tune (p,d,q) as needed
    try:
        model = SARIMAX(train_series, order=(1,0,0), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        forecast = result.forecast(steps=n_outputs)
    except Exception as e:
        print(f"SARIMAX failed for hex {idx}: {e}")
        forecast = np.full(n_outputs, np.nan)

    pred_rows.append(forecast)

# Convert to arrays
preds_array = np.array(pred_rows)
y_test = np.array(y_test_list)

# Flatten
y_true_flat = y_test.flatten()
y_pred_flat = preds_array.flatten()

# Compute metrics
mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
r2 = r2_score(y_true_flat, y_pred_flat)
nonzero_mask = y_true_flat != 0
if nonzero_mask.sum() > 0:
    mape = (np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask]) / y_true_flat[nonzero_mask])).mean() * 100
else:
    mape = np.nan

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
plt.plot(
    [y_true_flat.min(), y_true_flat.max()],
    [y_true_flat.min(), y_true_flat.max()],
    'r--', label='Perfect prediction'
)
plt.xlabel("True Counts")
plt.ylabel("Predicted Counts")
plt.title("Predicted vs True Counts (SARIMAX)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()