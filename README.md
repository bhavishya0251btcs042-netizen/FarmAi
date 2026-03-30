<div align="center">

<img src="logo/logo.png" alt="FarmAI Logo" width="140"/>

<h1>🌾 FarmAI</h1>

<p><em>AI-Powered Smart Farming Assistant</em></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
  <img src="https://img.shields.io/badge/Android-PWA-3DDC84?style=for-the-badge&logo=android&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-Academic-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Android-lightgrey?style=flat-square"/>
</p>

<br/>

> **FarmAI** helps farmers make smarter decisions using machine learning.  
> Get crop recommendations, detect plant diseases, and optimize fertilizer usage — all in one place.

<br/>

<img src="logo/results_dark_aesthetic.png" alt="FarmAI Preview" width="75%" style="border-radius: 12px;"/>

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🌱 Crop Recommendation
Predicts the best crop to grow based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall using a trained Random Forest model.

</td>
<td width="50%">

### 🔬 Disease Detection
Upload a leaf image and get instant disease diagnosis with treatment recommendations and fertilizer advice.

</td>
</tr>
<tr>
<td width="50%">

### 💊 Fertilizer Suggestion
Analyzes soil nutrient deficiencies and recommends the right fertilizer to maximize yield.

</td>
<td width="50%">

### 📊 History Tracking
Every prediction is saved per user — crop history and disease history accessible anytime.

</td>
</tr>
<tr>
<td width="50%">

### 📉 Bias & Fairness Auditing
Includes a dedicated auditing script (`fairness_audit.py`) to evaluate class distribution biases and measure model fairness across diverse environmental subgroups.

</td>
<td width="50%">

### 🧠 Explainable AI (XAI)
Powered by SHAP (Shapley Additive exPlanations) to dynamically explain why a specific crop is recommended, highlighting the exact contribution of each soil/climate feature.

</td>
</tr>
<tr>
<td width="50%">

### 🔐 User Authentication
Secure register, login, profile management, and password change with bcrypt encryption.

</td>
<td width="50%">

### 📱 Android PWA
Installable as a Progressive Web App on Android devices — works like a native app.

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:------|:----------:|:--------|
| 🐍 Backend | FastAPI (Python) | REST API & HTML serving |
| 🤖 ML Model | Random Forest (scikit-learn) | Crop prediction |
| 🗄️ Database | SQLite + SQLAlchemy | User data & history |
| 🔒 Auth | Passlib + bcrypt | Password hashing |
| 🎨 Frontend | HTML, CSS, JavaScript | User interface |
| ⚡ Server | Uvicorn (ASGI) | High-performance serving |

</div>

---

## 📁 Project Structure

```
FarmAi/
│
├── 🐍 crop.py                    # Main FastAPI backend & all API routes
├── 🤖 crop_model.joblib          # Pre-trained Random Forest model
├── ⚖️ fairness_audit.py          # Bias detection and fairness evaluation script
│
├── 📂 Crop_recommendation/
│   └── Crop_recommendation.csv   # Training dataset (10200 samples, 102 crops)
│
├── 📂 logo/                      # App branding & background images
│   └── crops/                    # Individual crop images
│
├── 📂 static/uploads/            # User profile pictures
│
├── 📂 android_app/               # PWA version for Android
├── 📂 android_native/            # Native Android wrapper (Java)
│
├── 🌐 crop ui.html               # Main crop prediction page
├── 🌐 login.html                 # Login page
├── 🌐 register.html              # Registration page
├── 🌐 results.html               # Prediction results
├── 🌐 disease.html               # Disease detection page
├── 🌐 fertilizer.html            # Fertilizer recommendation
├── 🌐 history.html               # User history
├── 🌐 profile.html               # User profile
│
├── 📋 requirements.txt           # Python dependencies
└── 🚀 start_farmai.bat           # One-click Windows launcher
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Bhavishaya789/FarmAi.git
cd FarmAi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python -m uvicorn crop:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at **http://127.0.0.1:8000** 🎉

### ⚡ Windows Quick Start

Just double-click **`start_farmai.bat`** — it installs dependencies and launches the app automatically.

---

## 📡 API Reference

<details>
<summary><b>Click to expand API endpoints</b></summary>

<br/>

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| `POST` | `/predict` | Crop recommendation with top-N results |
| `POST` | `/predict-disease` | Plant disease detection from image |
| `POST` | `/predict-fertilizer` | Fertilizer suggestion by NPK levels |
| `POST` | `/register` | Create new user account |
| `POST` | `/login` | User authentication |
| `POST` | `/change-password` | Update user password |
| `PUT` | `/users/{username}` | Update profile info |
| `POST` | `/users/{username}/profile-picture` | Upload profile picture |
| `GET` | `/get-user-history/{username}` | Fetch prediction history |
| `GET` | `/health` | Server health check |

</details>

---

## 🤖 ML Model Details

<div align="center">

| Property | Value |
|:---------|:------|
| Algorithm | Random Forest Classifier |
| Training Samples | 2,200 |
| Number of Crops | 22 |
| Input Features | N, P, K, Temperature, Humidity, pH, Rainfall |
| Output | Top-N crops with confidence scores |

</div>

**Supported Crops:** Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 👨‍💻 Team

<div align="center">

| Name | Roll Number |
|:-----|:-----------:|
| Bhavishya Kumar | 0251BTCS042 |
| Atharv Pandey | 0251BTCS048 |
| Dipanshu | 0251BTCS140 |
| Aditya Singh | 0251BTCS081 |

</div>

---

<div align="center">

<img src="logo/logo.png" width="50"/>

**FarmAI** — Built with ❤️ for farmers

*Academic Project | Computer Science Engineering*

</div>
