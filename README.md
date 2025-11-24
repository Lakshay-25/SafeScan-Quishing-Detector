# SafeScan – Quishing Detector (QR Code Phishing Detection)

SafeScan is a machine-learning powered **QR code security scanner** that detects hidden phishing (quishing) URLs inside QR codes.  
It scans a QR image or webcam feed, extracts the URL, analyzes it using ML models, and classifies it as:

- ✅ **Safe URL**
- ⚠️ **Suspicious URL**
- ❌ **Phishing URL**

SafeScan helps protect users from QR-based phishing attacks — a growing cyber threat where attackers embed malicious links inside QR codes.

---

## 🚀 Features

### 🔍 QR Code Processing
- Upload QR image  
- OR scan in real-time using device **webcam**
- Decodes QR → URL using **pyzbar + Pillow**

### 🧠 Machine Learning URL Analysis
- Extracts **16 lexical URL features**
- Uses trained ML models (Logistic Regression + ANN)
- Outputs **P(phishing)** and final classification

### 🛡️ Security Enhancements
- Built-in **trusted whitelist** (Google, GitHub, OpenAI, etc.)
- Automatic downgrading of URL shorteners (bit.ly, tinyurl, etc.)
- Risk rating badges:
  - 🔴 High risk
  - 🟡 Medium risk
  - 🟢 Low risk

### 🎨 Modern Web UI
- Clean dark-themed UI  
- Risk badges, model confidence, feature table  
- QR preview  
- Responsive webcam scanning  

---

## 🧠 System Architecture

SafeScan-Quishing-Detector/
│
├─ app.py
├─ train_ann.py
├─ train_url_models.py
│
├─ utils/
│   └─ features.py
│
├─ models/
│   ├─ ann_url_model.h5
│   ├─ url_lr_model.pkl
│   └─ url_lr_scaler.pkl
│
├─ templates/
│   ├─ index.html
│   └─ result.html
│
├─ static/
│   └─ css/
│       └─ style.css
│
├─ datasets/
│   └─ phiusiil_with_qr_minimal.csv
│
├─ logs/
│   └─ predictions_log.csv
│
├─ README.md
├─ LICENSE
└─ requirements.txt



# 📊 Dataset & Labels
Dataset: PhiUSIIL Phishing URL Dataset (Kaggle)

Original labels:
1 → Legitimate
0 → Phishing
In this project, we map to:
1 → Phishing (y_phish = 1)
0 → Safe / Legitimate (y_phish = 0)
This mapping is used consistently for ML training and evaluation.

# 🔬 Feature Engineering (16 Lexical Features)

- For each URL, we extract string-based (lexical) features without visiting the site, such as:
- url_length – total length of URL
- hostname_length – length of domain/hostname
- path_length – length of path /a/b/c
- num_dots – number of .
- num_hyphens – number of -
- num_digits – count of numeric characters
- num_special_chars – count of @ # ? % = & _
- num_subdomains – number of subdomain levels
- has_https – 1 if URL starts with https://, else 0
- https_in_domain – 1 if "https" appears inside domain name (often suspicious)
- contains_ip – 1 if IP address used instead of hostname
- contains_at – 1 if @ appears in URL
- contains_double_slash – extra // (often used to obfuscate)
- url_entropy – Shannon entropy (randomness/obfuscation)
- url_token_count – tokens when splitting on . / ? = & _ -
- is_shortener – 1 if domain is a known shortener (bit.ly, tinyurl, t.co, etc.)
- Implemented in utils/features.py.

# 🧮 Models Trained

We trained and compared multiple models on the URL features:

| Model               | Purpose                    | Notes                                          |
| ------------------- | -------------------------- | ---------------------------------------------- |
| ANN (Keras)         | URL classification         | High accuracy but overconfident on some URLs   |
| Logistic Regression | **Final production model** | More stable, interpretable, better calibration |
| Random Forest       | Feature importance         | Used mainly for analysis and comparison        |

# 🎯 Final Chosen Model

Logistic Regression + StandardScaler
- Good accuracy (~96–97% on validation)
- Fast and simple
- Well-behaved on real-world URLs and QR tests

# 🖥️ Running SafeScan Locally
### 1️⃣ Create Virtual Environment
- python -m venv .venv
### Windows:
- .venv\Scripts\activate
### macOS / Linux:
- source .venv/bin/activate

# 2️⃣ Install Requirements
- pip install -r requirements.txt
- (Make sure requirements.txt includes Flask, scikit-learn, numpy, pandas, pyzbar, Pillow, tensorflow, tldextract, etc.)

# 3️⃣ Run the App
python app.py

Then open in browser:
http://127.0.0.1:5000/

### You can now:

Upload QR images

Scan QR codes via webcam

See the URL classification & analysis

# 🔐 Security Logic (Decision Layer)
## Thresholds

The Logistic Regression model outputs P(phishing).
We apply thresholds:

- P(phishing) ≤ 0.25 → ✅ Safe URL

- P(phishing) ≥ 0.85 → ❌ Phishing URL

- Otherwise → ⚠️ Suspicious URL

## Whitelist Check

Certain well-known domains are treated as safe even if the model is unsure, e.g.:

- google.com

- github.com

- openai.com

- microsoft.com

- wikipedia.org

- etc.

If the URL’s host is in the whitelist, it is marked as Safe URL and a “Trusted Domain” badge is shown on the result page.

## Shortener Rule

If is_shortener == 1 (e.g. bit.ly, tinyurl.com, etc.):

- URL is treated as at least Suspicious

- Even if the model predicts safe, it will be downgraded to Suspicious URL, because link shorteners hide the true destination.

# 🧑‍💻 My Personal Contributions

In this project, I implemented:

- Selection and understanding of the Quishing (QR phishing) problem.

- Data preparation using the PhiUSIIL Phishing URL Dataset.

- Full ML pipeline:

- Data cleaning

- Train/validation split

- Feature extraction (16 lexical features)

- Model training and hyperparameter tuning

- Evaluation using accuracy, F1-score, and confusion matrix.

- Training multiple models:

- ANN (Keras)

- Logistic Regression (final model)

- Random Forest (feature importance analysis)

- Model saving/loading with joblib and Keras.

- Development of the Flask web application:

- Routes for / and /analyze

- Handling file uploads and webcam captures (base64 images)

- QR code decoding using pyzbar and PIL

- Integrating ML predictions into the web pipeline.

- Creating the frontend UI:

- index.html with upload + webcam scanner

- result.html with risk badges, confidence score, and feature table

- Implementing:

- Whitelist logic for trusted domains

- Shortener handling as suspicious

- Logging of predictions for debugging

