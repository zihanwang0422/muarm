import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle  # For saving scalers

# use sklearn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use tensorflow for building the model
from tensorflow import keras
from tensorflow.keras import layers


# Student ID for naming outputs; set environment variable SID to override
SID = os.environ.get("SID", "sid_placeholder")

# Read dataset
csv_filename = "ik_dataset.csv"
df = pd.read_csv(csv_filename)

# Step 1: Split dataset

# Split into features (X) and targets (y)
feature_cols = [
    "reference_joint_1",
    "reference_joint_2",
    "reference_joint_3",
    "reference_joint_4",
    "reference_joint_5",
    "reference_joint_6",
    "target_quat_x",
    "target_quat_y",
    "target_quat_z",
    "target_quat_w",
    "target_pos_x",
    "target_pos_y",
    "target_pos_z",
]
target_cols = [f"target_joint_{i+1}" for i in range(6)]

X = df[feature_cols]
y = df[target_cols]

# Split into train and test (you can use train_test_split from sklearn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Step 2: Normalize the data (you can use StandardScaler from sklearn)
# If you use StandardScaler, you need to fit the scaler on the train data and transform the test data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Save the scalers
mean_X = scaler_X.mean_
std_X = scaler_X.scale_
mean_y = scaler_y.mean_
std_y = scaler_y.scale_

with open(f"{SID}_scalers.pkl", "wb") as f:
    pickle.dump((mean_X, std_X, mean_y, std_y), f)

# Step 3: Define the model
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(6, activation="linear"),
    ]
)

# TODO: Define learning rate
learning_rate = 1e-3

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mse",  # Mean squared error for regression
    metrics=["mae"],  # Mean absolute error
)

print(model.summary())

# Step 4: Train the model
# Set up callbacks to save the best model and stop early if not improving
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{SID}_ik_model.keras",  # TODO: Define model checkpoint file name
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,  # TODO: Define patience
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train_scaled,
    y_train_scaled,  # Use normalized features AND normalized targets
    batch_size=128,  # TODO Define batch size
    epochs=300,  # TODO Define number of epochs
    validation_data=(
        X_test_scaled,
        y_test_scaled,
    ),
    callbacks=callbacks,
    verbose=1,
)

# Find best epoch (based on validation loss)
best_epoch = np.argmin(history.history["val_loss"])
best_val_loss = history.history["val_loss"][best_epoch]
best_val_mae = history.history["val_mae"][best_epoch]
best_train_loss = history.history["loss"][best_epoch]
best_train_mae = history.history["mae"][best_epoch]
# Print training statistics
print("\n" + "=" * 60)
print(f"\nBest Epoch ({best_epoch + 1}):")
print(f"  Training Loss: {best_train_loss:.6f}")
print(f"  Validation Loss: {best_val_loss:.6f}")
print(f"  Training MAE: {best_train_mae:.6f} rad ({np.rad2deg(best_train_mae):.4f}°)")
print(f"  Validation MAE: {best_val_mae:.6f} rad ({np.rad2deg(best_val_mae):.4f}°)")
print("=" * 60)
