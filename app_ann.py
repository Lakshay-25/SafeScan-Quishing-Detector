import os
import io
import json
import csv
import datetime
import traceback

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from urllib.parse import urlparse
from pyzbar.pyzbar import decode as qr_decode
from PIL import Image

import numpy as np
import joblib
import tensorflow as tf

from utils.features import extract_url_features, FEATURE_NAMES

# ---------- Config ----------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
LOG_PATH = os.path.join(MODEL_DIR, "predictions_log.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ---------- Error for large uploads ----------
@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    return render_template(
        "result.html",
        error="Uploaded image is too large. Lower camera resolution/quality or upload a smaller file."
    ), 413


# ---------- Load ANN model + scaler ----------
ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_url_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "url_scaler.pkl")

ann_model = None
scaler = None

try:
    ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
    print("Loaded ANN URL model from:", ANN_MODEL_PATH)
except Exception:
    print("Failed to load ANN model from:", ANN_MODEL_PATH)
    traceback.print_exc()

try:
    scaler = joblib.load(SCALER_PATH)
    print("Loaded URL scaler from:", SCALER_PATH)
except Exception:
    print("Failed to load scaler from:", SCALER_PATH)
    traceback.print_exc()

print("Model loading finished. ann_model:", bool(ann_model), "scaler:", bool(scaler))

# ---------- Decision thresholds ----------
# prob_malicious ∈ [0,1]
SAFE_THRESH = 0.30   # you can tune
PHISH_THRESH = 0.70  # you can tune

def decision_from_prob(prob_malicious, thresh_safe=SAFE_THRESH, thresh_phish=PHISH_THRESH):
    """
    Map probability of malicious into a label + confidence.
    prob_malicious: probability URL is phishing.
    """
    if prob_malicious >= thresh_phish:
        return "Phishing URL", prob_malicious
    elif prob_malicious <= thresh_safe:
        return "Safe URL", 1.0 - prob_malicious
    else:
        # middle band → suspicious
        return "Suspicious URL", max(prob_malicious, 1.0 - prob_malicious)


# ---------- QR decoding ----------
def decode_qr_from_pil(img_pil):
    decoded = qr_decode(img_pil)
    result = []
    for d in decoded:
        try:
            text = d.data.decode("utf-8")
        except Exception:
            text = d.data.decode("latin-1")
        result.append(text)
    return result


# ---------- Logging ----------
def log_prediction(image_path, decoded_url, prob_malicious, label, model_used="ann_url"):
    header = ["timestamp", "image_path", "decoded_url", "model_used", "prob_malicious", "label"]
    row = [
        datetime.datetime.utcnow().isoformat(),
        image_path or "",
        decoded_url,
        model_used,
        float(prob_malicious),
        label
    ]
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ---------- ANN prediction helper ----------
def predict_url_with_ann(url_text):
    try:
        feats = extract_url_features(url_text)  # dict → convert in correct order
        x = np.array([[feats[k] for k in FEATURE_NAMES]], dtype=float)

        x_scaled = scaler.transform(x)

        # ANN output = P(phishing)
        p_phish = float(ann_model.predict(x_scaled)[0][0])

        print("\n=== DEBUG ANN PREDICTION ===")
        print("URL:", url_text)
        print("Raw ANN output (P(phishing)):", p_phish)
        print("=============================\n")

        # decision
        label, conf = decision_from_prob(p_phish)

        return label, conf, p_phish, feats

    except Exception as e:
        print("ERROR in ANN prediction:", e)
        return "Error", 0.0, 0.0, {}



# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # Accept uploaded file OR base64 image from webcam
    file = request.files.get("file")
    pil_img = None
    saved_path = None

    if file and file.filename:
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)
        pil_img = Image.open(saved_path).convert("RGB")
    else:
        data = request.form.get("image_base64")
        if not data:
            return render_template("result.html", error="No image provided")
        import base64
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")

    # 1) Decode QR code(s)
    decoded_texts = decode_qr_from_pil(pil_img)
    if not decoded_texts:
        return render_template(
            "result.html",
            error="No QR code detected in the image. Try a clearer image or another QR."
        )

    url_text = decoded_texts[0].strip()

    # 2) ANN prediction on URL
    label, confidence, prob_malicious, feat_dict = predict_url_with_ann(url_text)

    # 3) Log prediction
    log_prediction(saved_path, url_text, prob_malicious, label, model_used="ann_url")

    # 4) Render result page
    return render_template(
        "result.html",
        url=url_text,
        label=label,
        confidence=round(confidence, 3),
        prob=round(prob_malicious, 3),
        features=feat_dict
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
