#pip install tqdm
import os
import time
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm  # progress bar
import matplotlib.pyplot as plt  # plotting

# -------------- Load manifest lists --------------
yes_manifest_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool/yes/manifest"
no_manifest_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool/no/manifest"

def load_filenames(manifest_url):
    response = requests.get(manifest_url)
    response.raise_for_status()
    return [line.strip() for line in response.text.splitlines() if line.strip()]

yes_index = load_filenames(yes_manifest_url)
no_index = load_filenames(no_manifest_url)

print(f"Found {len(yes_index)} 'yes' images.")
print(f"Found {len(no_index)} 'no' images.")

# -------------- Split data dynamically --------------
total_yes = len(yes_index)
total_no = len(no_index)

train_yes_count = int(total_yes * 0.10)
val_yes_count = int(total_yes * 0.05)
train_no_count = int(total_no * 0.10)
val_no_count = int(total_no * 0.05)

train_yes = yes_index[:train_yes_count]
val_yes = yes_index[train_yes_count:train_yes_count + val_yes_count]

train_no = no_index[:train_no_count]
val_no = no_index[train_no_count:train_no_count + val_no_count]

train_samples = [(fname, 1) for fname in train_yes] + [(fname, 0) for fname in train_no]
val_samples = [(fname, 1) for fname in val_yes] + [(fname, 0) for fname in val_no]

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# -------------- Batch generator with progress bar --------------
base_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool"

def batch_generator(samples, batch_size=16, img_size=(128,128), description="Loading"):
    for i in tqdm(range(0, len(samples), batch_size), desc=description):
        batch = samples[i:i+batch_size]
        X_batch, y_batch = [], []
        for fname, label in batch:
            subfolder = "yes" if label == 1 else "no"
            url = f"{base_url}/{subfolder}/{fname}"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB").resize(img_size)
                X_batch.append(np.array(img))
                y_batch.append(label)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        if X_batch:
            yield np.array(X_batch)/255.0, np.array(y_batch)

# -------------- Model --------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -------------- Download all batches with progress bar --------------
data_load_train_start = time.time()

train_batches = list(batch_generator(train_samples, batch_size=32, description="Training Data"))
X_train = np.concatenate([x for x,_ in train_batches])
y_train = np.concatenate([y for _,y in train_batches])

data_load_train_end = time.time()

data_load_val_start = time.time()

val_batches = list(batch_generator(val_samples, batch_size=32, description="Validation Data"))
X_val = np.concatenate([x for x,_ in val_batches])
y_val = np.concatenate([y for _,y in val_batches])

data_load_val_end = time.time()

# -------------- Custom Callback to Track Per-Batch Metrics --------------
class BatchMetricsLogger(Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get("loss"))
        self.batch_accuracies.append(logs.get("accuracy"))

batch_logger = BatchMetricsLogger()

# -------------- Training --------------
train_start = time.time()

history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[batch_logger]
)

train_end = time.time()

# -------------- Evaluation --------------
predict_start = time.time()

y_pred_prob = model.predict(X_val).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

predict_end = time.time()

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# -------------- Print Durations --------------
train_data_duration = data_load_train_end - data_load_train_start
val_data_duration = data_load_val_end - data_load_val_start
train_duration = train_end - train_start
predict_duration = predict_end - predict_start

print(f"\nData loading time (Training): {train_data_duration:.2f} seconds")
print(f"Data loading time (Validation): {val_data_duration:.2f} seconds")
print(f"Training time: {train_duration:.2f} seconds")
print(f"Prediction time: {predict_duration:.2f} seconds")

# -------------- Plot Training History (Per Epoch) --------------
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Accuracy per epoch
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Epoch Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# Loss per epoch
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Epoch Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

# -------------- Plot Per-Batch Metrics --------------
batch_steps = range(1, len(batch_logger.batch_losses) + 1)

plt.figure(figsize=(12, 5))

# Per-batch accuracy
plt.subplot(1, 2, 1)
plt.plot(batch_steps, batch_logger.batch_accuracies, label="Batch Training Accuracy")
plt.title("Per-Batch Training Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.legend()

# Per-batch loss
plt.subplot(1, 2, 2)
plt.plot(batch_steps, batch_logger.batch_losses, label="Batch Training Loss")
plt.title("Per-Batch Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
