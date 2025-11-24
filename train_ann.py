# train_ann.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

from utils.features import extract_url_features, FEATURE_NAMES

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "phiusiil_with_qr_minimal.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_url_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "url_scaler.pkl")

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

assert "url" in df.columns, "Dataset must have a 'url' column"
assert "label" in df.columns, "Dataset must have a 'label' column"

# Kaggle: label 1 = legitimate, 0 = phishing
# Here we define: y = 1 → phishing, 0 → legitimate
df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""].reset_index(drop=True)

df["y_phish"] = (df["label"] == 0).astype(int)
y = df["y_phish"].values

print("Total samples:", len(df))
print("Class counts (0=legit, 1=phishing):")
print(df["y_phish"].value_counts())

# Optional subsample
MAX_SAMPLES = 50000
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    y = df["y_phish"].values
    print(f"Subsampled to {MAX_SAMPLES} rows for training.")

# -----------------------------
# Feature extraction
# -----------------------------
def fe_row(u):
    return extract_url_features(u)

print("Extracting lexical features...")
feature_dicts = df["url"].apply(fe_row)
X_full = pd.DataFrame(list(feature_dicts))[FEATURE_NAMES]

print("Feature columns:", X_full.columns.tolist())
print("X_full shape:", X_full.shape)

# -----------------------------
# Train / validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_full.values,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -----------------------------
# ANN model
# -----------------------------
input_dim = X_train_scaled.shape[1]

ann_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")   # output = P(phishing=1)
])

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ann_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Training ANN...")
history = ann_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=15,
    batch_size=256,
    callbacks=[early_stop],
    verbose=2
)

# -----------------------------
# Evaluation
# -----------------------------
print("\nEvaluating on validation set...")
y_val_prob = ann_model.predict(X_val_scaled).ravel()
y_val_pred = (y_val_prob >= 0.5).astype(int)

print("\nClassification report (0=legit, 1=phishing):")
print(classification_report(y_val, y_val_pred))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))

acc = accuracy_score(y_val, y_val_pred)
print("Exact validation accuracy:", acc)

# -----------------------------
# Save model + scaler
# -----------------------------
print("\nSaving ANN model to:", ANN_MODEL_PATH)
ann_model.save(ANN_MODEL_PATH)

print("Saving scaler to:", SCALER_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Done.")
