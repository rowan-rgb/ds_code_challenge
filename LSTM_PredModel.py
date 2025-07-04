#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:18:48 2025

@author: rowandavies
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMForecast(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
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

model = LSTMForecast(hidden_size=64, num_layers=2)

pivoted = pd.read_csv("pivoted_weekly_counts.csv", index_col=0)

pivoted = pivoted.apply(pd.to_numeric, errors="coerce")

pivoted = pivoted.fillna(0)

# Optional: sort columns (weeks) numerically
pivoted = pivoted.sort_index(axis=1)
pivoted.columns = list(range(1, pivoted.shape[1]+1))
# Get the number of rows
n_rows = pivoted.shape[0]
# Compute split index
split_idx = int(n_rows * 0.8)
# Split
pivoted_train = pivoted.iloc[:split_idx]
pivoted_test = pivoted.iloc[split_idx:]

n_timesteps = 26
n_outputs = 4
X_list = []
y_list = []
data = pivoted_train.values
#change to pivoted when ready to train complete model

for row in data:
    max_start = data.shape[1] - n_timesteps - n_outputs + 1
    for start in range(max_start):
        end_input = start + n_timesteps
        end_output = end_input + n_outputs
        
        x_seq = row[start:end_input]
        y_seq = row[end_input:end_output]
        
        X_list.append(x_seq)
        y_list.append(y_seq)

X = np.array(X_list)
y = np.array(y_list)

# Convert X and y to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Batch size
batch_size = 32

# Number of samples
num_samples = X_tensor.shape[0]

# Number of epochs
n_epochs = 15

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []           # Stores average epoch losses
batch_loss_history = []     # Stores every batch loss

model.train()
for epoch in range(n_epochs):

    running_loss = 0.0
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_X = X_tensor[batch_indices]
        batch_X = batch_X.unsqueeze(-1)

        batch_y = y_tensor[batch_indices]

        optimizer.zero_grad()
        preds = model(batch_X)

        # Check for NaNs in predictions
        if torch.isnan(preds).any():
            print("NaNs detected in predictions!")
            print("Preds:", preds)
            break

        loss = criterion(preds, batch_y)

        if torch.isnan(loss):
            print("NaN loss detected!")
            print("Batch_X stats:", torch.min(batch_X), torch.max(batch_X))
            print("Batch_y stats:", torch.min(batch_y), torch.max(batch_y))
            print("Preds stats:", torch.min(preds), torch.max(preds))
            break

        loss.backward()
        optimizer.step()

        # Save the individual batch loss
        batch_loss_history.append(loss.item())

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / num_samples
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("LSTM_epoch_loss.png", dpi=300)
plt.show()

torch.save(model.state_dict(), "LSTM_forecast_model.pth")

model.eval()

# Create X_test and y_test from pivoted_test
test_data = pivoted_test.values  # shape: (num_hex, 53)
n_timesteps = 26
n_outputs = 4

X_test_list = []
y_test_list = []

for row in test_data:
    x_seq = row[-(n_timesteps + n_outputs):-n_outputs]
    y_seq = row[-n_outputs:]
    X_test_list.append(x_seq)
    y_test_list.append(y_seq)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

# Inference
pred_rows = []

with torch.no_grad():
    for row in X_test:
        row_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(row_tensor)
        pred_np = pred.numpy().squeeze(0)
        pred_rounded = np.round(pred_np, decimals=0)
        pred_rows.append(pred_rounded)

# Convert to array
preds_array = np.array(pred_rows)

# Flatten arrays
y_true_flat = y_test.flatten()
y_pred_flat = preds_array.flatten()

# Compute metrics
mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
r2 = r2_score(y_true_flat, y_pred_flat)

# MAPE calculation (avoid division by zero)
nonzero_mask = y_true_flat != 0
if nonzero_mask.sum() > 0:
    mape = (np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask]) / y_true_flat[nonzero_mask])).mean() * 100
else:
    mape = np.nan

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Combine into DataFrame for inspection
compare_df = pd.DataFrame(
    {
        "hex_index": pivoted_test.index if isinstance(pivoted_test.index, pd.Index) else np.arange(len(y_test)),
        "y_true": list(y_test),
        "y_pred": list(preds_array)
    }
)

# Sample 10 random rows for preview
sample_df = compare_df.sample(n=10, random_state=42)

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
plt.title("Predicted vs True Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Now that the model is trained, 
#use the best performing model to predict the callouts 
#for each hex for the four weeks beyond the dataset.
pivoted_inference = pivoted.iloc[:, -26:]
model.eval()

# Store predictions here
pred_rows = []
with torch.no_grad():
    for idx, row in pivoted_inference.iterrows():
        # Convert the row to a numpy array
        row_array = row.values.astype(np.float32)
        # Reshape to (1, seq_len, 1)
        input_tensor = torch.tensor(row_array).unsqueeze(0).unsqueeze(-1)
        # Model prediction
        pred = model(input_tensor)
        # Convert to numpy
        pred_np = pred.numpy().squeeze(0)
        pred_rows.append(pred_np)
# Convert predictions to array or DataFrame
predictions = np.array(pred_rows)
predictions = np.round(predictions, decimals=0)

pred_df = pd.DataFrame(
    predictions,
    index=pivoted_inference.index,
    columns=[f"week_{i+1}_pred" for i in range(predictions.shape[1])]
)

# Inspect
print(pred_df.head())
pred_df.to_csv("predictions.csv", index=True)
