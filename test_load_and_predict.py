# test_load_and_predict.py
import os
import joblib
import numpy as np
import tensorflow as tf
from pyzbar.pyzbar import decode as qr_decode
from PIL import Image

# Import your feature extractor
from utils.features import extract_url_features


# ----------------------------------------------------
# 1. Load ANN model + scaler
# ----------------------------------------------------
MODEL_DIR = "models"

ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_url_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "url_scaler.pkl")

print("\n=== Checking Models ===")
print("Working directory:", os.getcwd())
print("Model folder exists:", os.path.exists(MODEL_DIR))

ann_model = None
scaler = None

try:
    ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
    print("Loaded ANN model:", ANN_MODEL_PATH)
except Exception as e:
    print("ERROR loading ANN model:", e)

try:
    scaler = joblib.load(SCALER_PATH)
    print("Loaded scaler:", SCALER_PATH)
except Exception as e:
    print("ERROR loading scaler:", e)

if ann_model is None or scaler is None:
    print("❌ ANN or Scaler missing — cannot continue.")
    exit()


# ----------------------------------------------------
# 2. Helper: predict URL category using ANN
# ----------------------------------------------------
def predict_url(url_text):
    """
    ANN output = P(legitimate)
    We compute: prob_malicious = 1 - P(legitimate)
    """

    feats = extract_url_features(url_text)
    feats_arr = np.array(feats).reshape(1, -1)

    X_scaled = scaler.transform(feats_arr)

    p_legit = float(ann_model.predict(X_scaled)[0][0])
    p_malicious = 1.0 - p_legit

    if p_malicious >= 0.70:
        label = "Phishing URL"
    elif p_malicious <= 0.30:
        label = "Safe URL"
    else:
        label = "Suspicious URL"

    return label, p_legit, p_malicious


# ----------------------------------------------------
# 3. Test: decode QR image and run ANN prediction
# ----------------------------------------------------
# Change this path to any test QR image
TEST_IMAGE = "test_qr.png"

print("\n=== Testing QR Decoding & ANN Prediction ===")
print("QR test image:", TEST_IMAGE, "exists:", os.path.exists(TEST_IMAGE))

if not os.path.exists(TEST_IMAGE):
    print("❌ Test QR image not found. Place any QR at:", TEST_IMAGE)
    exit()

img = Image.open(TEST_IMAGE).convert("RGB")
decoded_list = qr_decode(img)

if not decoded_list:
    print("❌ No QR detected in the image.")
    exit()

url_text = decoded_list[0].data.decode("utf-8")
print("Decoded QR URL:", url_text)

label, p_legit, p_malicious = predict_url(url_text)

print("\n=== ANN Prediction ===")
print("URL:", url_text)
print("P(legitimate):", round(p_legit, 4))
print("P(malicious):", round(p_malicious, 4))
print("Final label:", label)
