# train_url_models.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils.features import extract_url_features, FEATURE_NAMES


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "phiusiil_with_qr_minimal.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LR_MODEL_PATH = os.path.join(MODEL_DIR, "url_lr_model.pkl")
LR_SCALER_PATH = os.path.join(MODEL_DIR, "url_lr_scaler.pkl")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "url_rf_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "url_xgb_model.pkl")


# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

assert "url" in df.columns, "Dataset must have a 'url' column"
assert "label" in df.columns, "Dataset must have a 'label' column"

# Kaggle: label 1 = legitimate, label 0 = phishing
# Our target: y = 1 → phishing, 0 → legitimate
df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""].reset_index(drop=True)

df["y_phish"] = (df["label"] == 0).astype(int)
y = df["y_phish"].values

print("Total samples:", len(df))
print("Class counts (0=legit, 1=phishing):")
print(df["y_phish"].value_counts())

# Optional subsample for speed
MAX_SAMPLES = 60000
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    y = df["y_phish"].values
    print(f"Subsampled to {MAX_SAMPLES} rows.")


# -----------------------------
# Feature extraction
# -----------------------------
print("Extracting lexical features using utils.features...")
feat_dicts = df["url"].apply(extract_url_features)
X_full = pd.DataFrame(list(feat_dicts))[FEATURE_NAMES]

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


# =========================================================
# 1. Logistic Regression (good baseline, calibrated probs)
# =========================================================
print("\n=== Training Logistic Regression (URL features) ===")

scaler_lr = StandardScaler()
X_train_scaled = scaler_lr.fit_transform(X_train)
X_val_scaled = scaler_lr.transform(X_val)

lr = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced"  # handle any imbalance
)

lr.fit(X_train_scaled, y_train)

y_val_prob_lr = lr.predict_proba(X_val_scaled)[:, 1]  # P(phishing)
y_val_pred_lr = (y_val_prob_lr >= 0.5).astype(int)

print("\nLogistic Regression report (0=legit, 1=phishing):")
print(classification_report(y_val, y_val_pred_lr))
print("Confusion matrix (LR):")
print(confusion_matrix(y_val, y_val_pred_lr))
print("Accuracy (LR):", accuracy_score(y_val, y_val_pred_lr))
print("ROC-AUC (LR):", roc_auc_score(y_val, y_val_prob_lr))

print("Saving LR model + scaler...")
joblib.dump(lr, LR_MODEL_PATH)
joblib.dump(scaler_lr, LR_SCALER_PATH)


# =========================================================
# 2. Random Forest (nonlinear, feature importance)
# =========================================================
print("\n=== Training Random Forest ===")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42
)

rf.fit(X_train, y_train)

y_val_prob_rf = rf.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_prob_rf >= 0.5).astype(int)

print("\nRandom Forest report:")
print(classification_report(y_val, y_val_pred_rf))
print("Confusion matrix (RF):")
print(confusion_matrix(y_val, y_val_pred_rf))
print("Accuracy (RF):", accuracy_score(y_val, y_val_pred_rf))
print("ROC-AUC (RF):", roc_auc_score(y_val, y_val_prob_rf))

print("Saving RF model...")
joblib.dump(rf, RF_MODEL_PATH)

# Feature importance from RF
importances = rf.feature_importances_
feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
print("\nRandom Forest feature importances:")
for name, imp in feat_imp:
    print(f"{name:25s}: {imp:.4f}")


# =========================================================
# 3. XGBoost (optional) – often very strong on tabular data
# =========================================================
if HAS_XGB:
    print("\n=== Training XGBoost (if installed) ===")
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    y_val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
    y_val_pred_xgb = (y_val_prob_xgb >= 0.5).astype(int)

    print("\nXGBoost report:")
    print(classification_report(y_val, y_val_pred_xgb))
    print("Confusion matrix (XGB):")
    print(confusion_matrix(y_val, y_val_pred_xgb))
    print("Accuracy (XGB):", accuracy_score(y_val, y_val_pred_xgb))
    print("ROC-AUC (XGB):", roc_auc_score(y_val, y_val_prob_xgb))

    print("Saving XGB model...")
    joblib.dump(xgb, XGB_MODEL_PATH)

    # Feature importance (gain-based)
    xgb_imp = xgb.feature_importances_
    feat_imp_xgb = sorted(zip(FEATURE_NAMES, xgb_imp), key=lambda x: x[1], reverse=True)
    print("\nXGBoost feature importances:")
    for name, imp in feat_imp_xgb:
        print(f"{name:25s}: {imp:.4f}")
else:
    print("\nXGBoost not installed; skipping XGB training.")

print("\nDone training URL models.")
