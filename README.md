# SafeScan – Quishing Detector (QR Code Phishing Detection)

SafeScan is a machine-learning powered **QR code security scanner** that detects hidden phishing (quishing) URLs inside QR codes.  
It scans a QR image or webcam feed, extracts the URL, analyzes it using ML models, and classifies it as:

- ✅ **Safe URL**
- ⚠️ **Suspicious URL**
- ❌ **Phishing URL**

SafeScan helps protect users from QR-based phishing attacks — a growing cyber threat where attackers embed malicious links inside QR codes.

---

## 🚀 Features

### 🔍 **QR Code Processing**
- Upload QR image  
- OR scan in real-time using device **webcam**
- Decodes QR → URL using **pyzbar + Pillow**

### 🧠 **Machine Learning URL Analysis**
- Extracts **16 lexical URL features**
- Uses trained ML models (Logistic Regression + ANN)
- Outputs **P(phishing)** and classification

### 🛡️ **Security Enhancements**
- Built-in **trusted whitelist** (Google, GitHub, OpenAI, etc.)
- Automatic downgrading of URL shorteners (bit.ly, tinyurl, etc.)
- Risk rating badges:
  - 🔴 High risk
  - 🟡 Medium risk
  - 🟢 Low risk

### 🎨 **Modern Web UI**
- Clean UI with dark theme  
- Risk badges, model confidence, feature table  
- QR preview  
- Responsive webcam scanning  

---

## 🧠 System Architecture

```mermaid
flowchart LR
    A[User Browser] -->|Upload QR / Webcam| B[Flask Backend]
    B --> C[QR Decoder<br>(pyzbar + PIL)]
    C -->|URL| D[URL Feature Extractor]
    D -->|16 Features| E[Scaler]
    E -->|Scaled| F[Logistic Regression Model]
    F -->|P(Phishing)| G[Decision Layer]
    G --> H[Result Page]

flowchart TD
    A[Dataset (PhiUSIIL)] --> B[Cleaning & Preprocessing]
    B --> C[Feature Engineering<br>16 Lexical Features]
    C --> D[Train/Validation Split]
    D --> E[Scaling (StandardScaler)]
    E --> F1[Train ANN]
    E --> F2[Train Logistic Regression]
    E --> F3[Train Random Forest]
    F1 --> G[Evaluate]
    F2 --> G
    F3 --> G
    G --> H[Select Best Model<br>(Logistic Regression)]
    H --> I[Deploy in Flask App]

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
│   └─ css/style.css
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
```



PhiUSIIL Phishing URL Dataset
Label meaning in dataset:
1 → Legitimate
0 → Phishing

Mapped internally as:
1 = phishing
0 = safe

🔬 Feature Engineering (16 Lexical Features)
Examples:
URL length
Number of dots / hyphens
Presence of IP address
URL entropy
HTTPS flag
Token count
Shortener detection
Extracted using utils/features.py.

🧮 Models Trained
Model	               Purpose	                Notes
ANN (Keras)	           URL classification	    High accuracy, but overconfident on unseen URLs
Logistic Regression    Final production model	Best real-world behavior
Random Forest	       Feature importance	    Optional

🎯 Final Model
Logistic Regression + StandardScaler
→ Best balance of stability, speed, generalization.

```

🖥️ Running SafeScan Locally 
1️⃣ Create Virtual Environment

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the App
python app.py

Open browser:
http://127.0.0.1:5000/

🔐 Security Logic (Decision Layer)
Thresholds
SAFE ≤ 0.25
PHISH ≥ 0.85
Else: Suspicious

Whitelist Check
Trusted domains auto-safe:
google.com
github.com
openai.com
microsoft.com
etc.
Shortener Rule

If URL is shortened → classify as at least Suspicious.

🧑‍💻 My Personal Contributions
I implemented:
Full ML pipeline (cleaning → features → training → evaluation)
Feature engineering (16 URL features)
ANN model and Logistic Regression model
Model saving/loading with joblib + Keras
Flask backend (/ & /analyze )
Webcam QR scanning (JavaScript + getUserMedia)
Result page UI with risk badges & trusted indicators
Whitelist + shortener rule
Debugging, testing, and real QR verification
Complete documentation and GitHub setup
```