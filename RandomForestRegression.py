import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


pivoted = pd.read_csv("pivoted_weekly_counts.csv", index_col=0)

# ---------------------------
# Prepare supervised learning samples
# ---------------------------

n_lags = 48
n_forecast = 4

X_list = []
y_list = []
area_list = []

# For each area (row)
for area_idx, row in pivoted.iterrows():
    series = row.values
    max_start = len(series) - n_lags - n_forecast + 1

    # Slide window
    for start in range(max_start):
        end_lag = start + n_lags
        end_forecast = end_lag + n_forecast

        X_seq = series[start:end_lag]
        y_seq = series[end_lag:end_forecast]

        X_list.append(X_seq)
        y_list.append(y_seq)
        area_list.append(area_idx)

X = np.array(X_list)
y = np.array(y_list)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Model: Random Forest
# ---------------------------
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# ---------------------------
# Predict
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Evaluate
# ---------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# MAPE (handle divide by zero)
y_true_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
nonzero_mask = y_true_flat != 0
if nonzero_mask.sum() > 0:
    mape = (
        np.abs(
            (y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask])
            / y_true_flat[nonzero_mask]
        ).mean()
        * 100
    )
else:
    mape = np.nan

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# ---------------------------
# Scatter plot of predictions
# ---------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
plt.plot(
    [y_true_flat.min(), y_true_flat.max()],
    [y_true_flat.min(), y_true_flat.max()],
    'r--', label='Perfect prediction'
)
plt.xlabel("True Counts")
plt.ylabel("Predicted Counts")
plt.title("RandomForest Predictions vs True Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()