import os
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from random import shuffle, random, choice
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
ROTATION_PROB = 0.5
ROTATION_ANGLES = [90, 180, 270]

# ----- MANIFEST URLS -----
yes_manifest_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool/yes/manifest"
no_manifest_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool/no/manifest"
base_url = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/images/swimming-pool"

# ----- LOAD FILENAMES -----
def load_filenames(url):
    r = requests.get(url)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]

yes_index = load_filenames(yes_manifest_url)
no_index = load_filenames(no_manifest_url)

print(f"Found {len(yes_index)} 'yes' images and {len(no_index)} 'no' images.")

# ----- SPLIT DATA -----
train_yes_count = 2000
val_yes_count = 400
train_no_count = 2000
val_no_count = 400

np.random.seed(42)
yes_shuffled = np.random.permutation(yes_index)
no_shuffled = np.random.permutation(no_index)

train_yes = yes_shuffled[:train_yes_count]
val_yes = yes_shuffled[train_yes_count:train_yes_count + val_yes_count]
train_no = no_shuffled[:train_no_count]
val_no = no_shuffled[train_no_count:train_no_count + val_no_count]

train_samples = [(f, 1) for f in train_yes] + [(f, 0) for f in train_no]
val_samples = [(f, 1) for f in val_yes] + [(f, 0) for f in val_no]
shuffle(train_samples)
shuffle(val_samples)

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# ----- DATA GENERATOR -----
def data_generator(samples, batch_size=BATCH_SIZE, augment=False):
    while True:
        shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            X_batch, y_batch = [], []
            for fname, label in batch:
                subfolder = "yes" if label == 1 else "no"
                url = f"{base_url}/{subfolder}/{fname}"
                try:
                    r = requests.get(url, timeout=10)
                    r.raise_for_status()
                    img = Image.open(BytesIO(r.content)).convert("RGB").resize(IMG_SIZE)
                    if augment and random() < ROTATION_PROB:
                        img = img.rotate(choice(ROTATION_ANGLES))
                    arr = preprocess_input(np.array(img))
                    X_batch.append(arr)
                    y_batch.append(label)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
            if X_batch:
                yield np.array(X_batch), to_categorical(y_batch, num_classes=2)

# ----- MODEL -----
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----- CUSTOM CALLBACK FOR BATCH LOSSES -----
class BatchLogger(Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get("loss"))

batch_logger = BatchLogger()

# ----- TRAINING -----
steps_per_epoch = len(train_samples) // BATCH_SIZE
validation_steps = len(val_samples) // BATCH_SIZE

print("Starting training...")
train_start = time.time()

history = model.fit(
    data_generator(train_samples, augment=True),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(val_samples),
    validation_steps=validation_steps,
    callbacks=[batch_logger]
)

train_end = time.time()
training_duration = train_end - train_start
print(f"Training completed in {training_duration:.2f} seconds.")

# ----- EVALUATION -----
X_val = []
y_val = []

print("Downloading validation set...")
val_download_start = time.time()
for fname, label in tqdm(val_samples, desc="Validation Images"):
    subfolder = "yes" if label == 1 else "no"
    url = f"{base_url}/{subfolder}/{fname}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB").resize(IMG_SIZE)
        arr = preprocess_input(np.array(img))
        X_val.append(arr)
        y_val.append(label)
    except Exception as e:
        print(f"Skipping {fname}: {e}")

val_download_end = time.time()
validation_download_duration = val_download_end - val_download_start
print(f"Validation data download completed in {validation_download_duration:.2f} seconds.")

X_val = np.array(X_val)
y_val = np.array(y_val)

print("Predicting validation set...")
predict_start = time.time()
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
predict_end = time.time()
prediction_duration = predict_end - predict_start
print(f"Prediction completed in {prediction_duration:.2f} seconds.")

print(classification_report(y_val, y_pred, target_names=["No Pool", "Pool"]))

# ----- PLOT LOSSES -----
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Epoch Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], label="Epoch Training Loss")
plt.plot(epochs_range, history.history["val_loss"], label="Epoch Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Batch Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, len(batch_logger.batch_losses) + 1), batch_logger.batch_losses, label="Batch Training Loss")
plt.title("Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ----- PRINT SUMMARY DURATIONS -----
print("\nSummary of Durations:")
print(f"Total Training Time: {training_duration:.2f} seconds")
print(f"Validation Download Time: {validation_download_duration:.2f} seconds")
print(f"Prediction Time: {prediction_duration:.2f} seconds")