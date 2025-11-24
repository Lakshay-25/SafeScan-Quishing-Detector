# app.py
import os
import io
import csv
import base64
import datetime
import traceback

from urllib.parse import urlparse

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from pyzbar.pyzbar import decode as qr_decode
from PIL import Image

import numpy as np
import joblib
import tensorflow as tf

from utils.features import extract_url_features, FEATURE_NAMES

# ============================================================
# 1. PATHS & FLASK CONFIG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LOG_PATH = os.path.join(MODEL_DIR, "predictions_log.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


# ============================================================
# 2. WHITELISTED SAFE DOMAINS
# ============================================================
WHITELIST_DOMAINS = {
    "github.com",
    "google.com",
    "openai.com",
    "microsoft.com",
    "apple.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "wikipedia.org",
    "amazon.com",
    "reddit.com",
    "cloudflare.com",
    "medium.com",
    "stackoverflow.com",
    "python.org",
}


def is_whitelisted(url: str) -> bool:
    """
    Returns True if URL host is in our trusted allow-list.
    Prevents absurd false positives for major sites.
    """
    try:
        host = urlparse(url if url.startswith("http") else "http://" + url).netloc
        host = host.lower().strip()
        return any(host == d or host.endswith("." + d) for d in WHITELIST_DOMAINS)
    except Exception:
        return False


# ============================================================
# 3. LOAD ANN MODEL + SCALER
# ============================================================
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


# ============================================================
# 4. LOAD LOGISTIC REGRESSION MODEL + SCALER
# ============================================================
LR_MODEL_PATH = os.path.join(MODEL_DIR, "url_lr_model.pkl")
LR_SCALER_PATH = os.path.join(MODEL_DIR, "url_lr_scaler.pkl")

lr_model = None
lr_scaler = None

try:
    lr_model = joblib.load(LR_MODEL_PATH)
    print("Loaded Logistic Regression URL model from:", LR_MODEL_PATH)
except Exception:
    print("WARNING: Could not load Logistic Regression model.")
    traceback.print_exc()

try:
    lr_scaler = joblib.load(LR_SCALER_PATH)
    print("Loaded Logistic Regression scaler from:", LR_SCALER_PATH)
except Exception:
    print("WARNING: Could not load Logistic Regression scaler.")
    traceback.print_exc()


# ============================================================
# 5. DECISION LOGIC (SAFE / SUSPICIOUS / PHISHING)
# ============================================================
# prob_malicious = P(phishing)
SAFE_THRESH = 0.25
PHISH_THRESH = 0.85


def decision_from_prob(prob_malicious, thresh_safe=SAFE_THRESH, thresh_phish=PHISH_THRESH):
    """
    Map probability of malicious into label + confidence.
    prob_malicious: probability URL is phishing.
    """
    if prob_malicious >= thresh_phish:
        return "Phishing URL", prob_malicious
    elif prob_malicious <= thresh_safe:
        return "Safe URL", 1.0 - prob_malicious
    else:
        return "Suspicious URL", max(prob_malicious, 1.0 - prob_malicious)


# ============================================================
# 6. QR DECODING
# ============================================================
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


# ============================================================
# 7. LOGGING
# ============================================================
def log_prediction(image_path, decoded_url, prob_malicious, label, model_used="lr_url"):
    header = ["timestamp", "image_path", "decoded_url", "model_used", "prob_malicious", "label"]
    row = [
        datetime.datetime.utcnow().isoformat(),
        image_path or "",
        decoded_url,
        model_used,
        float(prob_malicious),
        label,
    ]
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ============================================================
# 8. ANN PREDICTION HELPER (fallback / comparison)
# ============================================================
def predict_url_with_ann(url_text: str):
    """
    Use ANN to predict whether URL is phishing or legitimate.

    Training: y=1 means phishing, y=0 legitimate.
    So ANN output is P(phishing).
    """
    if ann_model is None or scaler is None:
        return "No model available", 0.0, 0.0, {}

    # Whitelist override
    if is_whitelisted(url_text):
        feats_dict = extract_url_features(url_text)
        print("\n[WHITELIST / ANN] Auto-safe for:", url_text)
        return "Safe URL", 1.0, 0.0, feats_dict

    feats_dict = extract_url_features(url_text)
    x = np.array([[feats_dict[name] for name in FEATURE_NAMES]], dtype=float)
    x_scaled = scaler.transform(x)

    p_phish = float(ann_model.predict(x_scaled)[0][0])
    prob_malicious = p_phish

    label, confidence = decision_from_prob(prob_malicious)

    print("\n=== DEBUG ANN PREDICTION ===")
    print("URL:", url_text)
    print("Features:", feats_dict)
    print("P(phishing):", prob_malicious)
    print("Label:", label, "Confidence:", confidence)
    print("================================\n")

    return label, confidence, prob_malicious, feats_dict


# ============================================================
# 9. LOGISTIC REGRESSION PREDICTION (main model)
# ============================================================
def predict_url_with_lr(url_text: str):
    """
    Main prediction function used by the web app.
    Uses Logistic Regression on lexical URL features.
    """
    if lr_model is None or lr_scaler is None:
        # fallback to ANN if LR not available
        return predict_url_with_ann(url_text)

    # Whitelist override first
    if is_whitelisted(url_text):
        feats_dict = extract_url_features(url_text)
        print("\n[WHITELIST / LR] Auto-safe for:", url_text)
        return "Safe URL", 1.0, 0.0, feats_dict

    feats_dict = extract_url_features(url_text)
    x = np.array([[feats_dict[name] for name in FEATURE_NAMES]], dtype=float)

    x_scaled = lr_scaler.transform(x)
    p_phish = float(lr_model.predict_proba(x_scaled)[0][1])  # P(class=1 = phishing)
    prob_malicious = p_phish

    label, confidence = decision_from_prob(prob_malicious)

    # ---------- SHORTENER RULE ----------
    # If URL is from a shortener domain, never call it "Safe".
    # If model says "Phishing", downgrade to "Suspicious".
    if feats_dict.get("is_shortener", 0) == 1:
        if label == "Safe URL":
            label = "Suspicious URL"
        elif label == "Phishing URL":
            label = "Suspicious URL"

    print("\n=== DEBUG LR PREDICTION ===")
    print("URL:", url_text)
    print("Features:", feats_dict)
    print("P(phishing):", prob_malicious)
    print("Final Label:", label, "Confidence:", confidence)
    print("================================\n")

    # Always return exactly 4 values
    return label, confidence, prob_malicious, feats_dict


# ============================================================
# 10. ERROR HANDLER FOR LARGE UPLOADS
# ============================================================
@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    return render_template(
        "result.html",
        error="Uploaded image is too large. Lower camera resolution/quality or upload a smaller file."
    ), 413


# ============================================================
# 11. ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        pil_img = None
        saved_path = None

        # ------------- FILE UPLOAD -------------
        if file and file.filename:
            filename = secure_filename(file.filename)
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(saved_path)
            pil_img = Image.open(saved_path).convert("RGB")
        else:
            # ------------- WEBCAM (BASE64) -------------
            data = request.form.get("image_base64")
            if not data:
                return render_template("result.html", error="No image provided.")

            try:
                encoded = data.split(",", 1)[1]
            except Exception:
                return render_template("result.html", error="Invalid image data received.")

            decoded_bytes = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(decoded_bytes)).convert("RGB")

        # ------------- QR DECODE -------------
        decoded_texts = decode_qr_from_pil(pil_img)
        if not decoded_texts:
            return render_template(
                "result.html",
                error="No QR code detected in the image. Try a clearer image or another QR."
            )

        url_text = decoded_texts[0].strip()

        # ------------- MODEL PREDICTION (LR) -------------
        label, confidence, prob_malicious, feat_dict = predict_url_with_lr(url_text)

        # ------------- LOG PREDICTION -------------
        log_prediction(saved_path, url_text, prob_malicious, label, model_used="lr_url")

        # Optional: relative QR image path (if you want to show preview)
        qr_image_rel = None
        if saved_path:
            # file is saved as static/uploads/<filename>
            qr_image_rel = "uploads/" + os.path.basename(saved_path)

        # Trusted flag for badge
        is_trusted = is_whitelisted(url_text)

        # ------------- RENDER RESULT -------------
        return render_template(
            "result.html",
            url=url_text,
            label=label,
            confidence=round(confidence, 3),
            prob=round(prob_malicious, 3),
            features=feat_dict,
            is_trusted=is_trusted,
            qr_image=qr_image_rel,  # use in template if you want
        )

    except Exception as e:
        print("Error in /analyze:", e)
        traceback.print_exc()
        return render_template(
            "result.html",
            error=f"Error while processing URL: {e}",
        )


# ============================================================
# 12. MAIN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
