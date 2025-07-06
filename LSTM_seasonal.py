#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:48:08 2025

@author: rowandavies
"""

import pandas as pd
import matplotlib.pyplot as plt
#pip install folium
#pip install folium==0.14.0
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
#Let's start with model training
#Step 1
#Create weekly counts
#Let's try a more complicated prediction model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LSTMForecast(nn.Module):
    def __init__(self, hidden_size=128, num_layers=4):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,       # one feature per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        # x shape: (batch_size, seq_len=13, 1)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_size)
        output = self.fc(last_hidden)  # shape: (batch_size, 4)
        return output

# Instantiate model
model = LSTMForecast(hidden_size=64, num_layers=2)

pivoted = pd.read_csv("pivoted_weekly_counts.csv", index_col=0)

pivoted = pivoted.apply(pd.to_numeric, errors="coerce")
pivoted = pivoted.fillna(0)
pivoted = pivoted.sort_index(axis=1)
pivoted.columns = list(range(1, pivoted.shape[1]+1))

# Create sin and cos encodings
n_weeks = pivoted.shape[1]
weeks = np.arange(1, n_weeks + 1)
week_radians = (weeks - 1) / 52 * 2 * np.pi
sin_week = np.sin(week_radians)
cos_week = np.cos(week_radians)

# Build sin and cos columns
sin_columns = {}
cos_columns = {}
for i, week in enumerate(weeks):
    sin_columns[f"{week}_sin"] = sin_week[i]
    cos_columns[f"{week}_cos"] = cos_week[i]

sin_df = pd.DataFrame(sin_columns, index=pivoted.index)
cos_df = pd.DataFrame(cos_columns, index=pivoted.index)

# Concatenate
pivoted_extended = pd.concat([pivoted, sin_df, cos_df], axis=1)

# Order columns nicely
ordered_cols = []
for week in weeks:
    ordered_cols.append(week)               # count column
    ordered_cols.append(f"{week}_sin")      # sin
    ordered_cols.append(f"{week}_cos")      # cos

pivoted_extended = pivoted_extended[ordered_cols]

# Preview
print(pivoted_extended.head())

# Train/test split
n_rows = pivoted_extended.shape[0]
split_idx = int(n_rows * 0.8)
pivoted_train = pivoted_extended.iloc[:split_idx]
pivoted_test = pivoted_extended.iloc[split_idx:]

# Sequence generation
# Configurable parameter: how many historical weeks to use
weeks_historical = 48
n_timesteps = weeks_historical 
n_outputs = 4

X_list = []
y_list = []

data = pivoted_extended

# Use only count columns to define valid weeks
n_weeks = pivoted.shape[1]
weeks = list(range(1, n_weeks + 1))

# Normalization bounds
X_min = pivoted.values.min()
X_max = pivoted.values.max()

# Loop over each row (hex)
for idx, row in data.iterrows():
    max_start = n_weeks - n_timesteps - n_outputs + 1
    if max_start < 1:
        continue

    for start in range(max_start):
        end_input = start + n_timesteps
        end_output = end_input + n_outputs

        x_seq = []
        for week in range(start + 1, end_input + 1):
            count = row[week]
            norm_count = (count - X_min) / (X_max - X_min + 1e-8)
            sin = row[f"{week}_sin"]
            cos = row[f"{week}_cos"]
            x_seq.append([norm_count, sin, cos])

        if len(x_seq) < n_timesteps:
            continue
        x_seq = np.array(x_seq)

        y_counts = []
        for week in range(end_input + 1, end_output + 1):
            count = row[week]
            norm_count = (count - X_min) / (X_max - X_min + 1e-8)
            y_counts.append(norm_count)

        if len(y_counts) < n_outputs:
            continue
        y_seq = np.array(y_counts)

        X_list.append(x_seq)
        y_list.append(y_seq)

X = np.array(X_list)
y = np.array(y_list)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Model
class LSTMForecast(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(
            input_size=3,   # count + sin + cos
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output

model = LSTMForecast()

# Training tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

batch_size = 32
num_samples = X_tensor.shape[0]
n_epochs = 14

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []
model.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_X = X_tensor[batch_indices]
        batch_y = y_tensor[batch_indices]

        optimizer.zero_grad()
        preds = model(batch_X)

        if torch.isnan(preds).any():
            print("NaNs detected in predictions!")
            break

        loss = criterion(preds, batch_y)

        if torch.isnan(loss):
            print("NaN loss detected!")
            break

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / num_samples
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.6f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("LSTM_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

torch.save(model.state_dict(), "LSTM_forecast_model.pth")

# Inference
model.eval()
X_test_list = []
y_test_list = []

for idx, row in pivoted_test.iterrows():
    x_seq = []
    for week in range(n_weeks - n_outputs - n_timesteps + 1, n_weeks - n_outputs + 1):
        count = row[week]
        norm_count = (count - X_min) / (X_max - X_min + 1e-8)
        sin = row[f"{week}_sin"]
        cos = row[f"{week}_cos"]
        x_seq.append([norm_count, sin, cos])

    x_seq = np.array(x_seq)

    y_counts = []
    for week in range(n_weeks - n_outputs + 1, n_weeks + 1):
        count = row[week]
        norm_count = (count - X_min) / (X_max - X_min + 1e-8)
        y_counts.append(norm_count)

    y_seq = np.array(y_counts)

    X_test_list.append(x_seq)
    y_test_list.append(y_seq)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

pred_rows = []

with torch.no_grad():
    for row in X_test:
        row_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
        pred = model(row_tensor)
        pred_np = pred.numpy().squeeze(0)
        pred_rescaled = pred_np * (X_max - X_min) + X_min
        pred_rounded = np.round(pred_rescaled, decimals=0)
        pred_rows.append(pred_rounded)

preds_array = np.array(pred_rows)
y_true_flat = (y_test * (X_max - X_min) + X_min).flatten()
y_pred_flat = preds_array.flatten()

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

plt.figure(figsize=(6,6))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
plt.plot(
    [y_true_flat.min(), y_true_flat.max()],
    [y_true_flat.min(), y_true_flat.max()],
    'r--', label='Perfect prediction'
)
plt.xlabel("True Counts")
plt.ylabel("Predicted Counts")
plt.title("Predicted vs True Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
