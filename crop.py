# FarmAI Backend - crop.py
# Bhavishya kumar - 0251BTCS042
# Atharv pandey  - 0251BTCS048
# Dipanshu       - 0251BTCS140
# Aditya singh   - 0251BTCS081

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os, shutil, random, string
from disease_model import predict_disease_from_image
import requests as http_requests
from typing import Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------
# CONFIG
# ----------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/farmai")
JWT_SECRET  = os.getenv("JWT_SECRET", "changeme_secret_please_change")
JWT_ALGO    = "HS256"
JWT_EXPIRE  = 60 * 24  # minutes
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

# ----------------------------------------
# MONGODB
# ----------------------------------------
mongo_client     = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
db_mongo         = mongo_client["farmai"]
users_col        = db_mongo["users"]
crop_hist_col    = db_mongo["crop_history"]
disease_hist_col = db_mongo["disease_history"]
otp_col          = db_mongo["otp_store"]

# ----------------------------------------
# AUTH HELPERS
# ----------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p): return pwd_context.hash(p)
def verify_password(p, h): return pwd_context.verify(p, h)

def create_token(username: str):
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE)
    return jwt.encode({"sub": username, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGO)

from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token or token expired")

def generate_otp():
    return "".join(random.choices(string.digits, k=6))

# ----------------------------------------
# ML MODEL
# ----------------------------------------
def train_and_persist_model():
    csv_path = "Crop_recommendation/Crop_recommendation.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X, y = df[feature_cols], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    m.fit(X, y)
    joblib.dump({"model": m, "features": feature_cols, "accuracy": acc}, "crop_model.joblib")
    return m, feature_cols, acc

def load_model():
    if os.path.exists("crop_model.joblib"):
        try:
            data = joblib.load("crop_model.joblib")
            if "accuracy" in data:
                return data["model"], data["features"], data["accuracy"]
        except:
            pass
    return train_and_persist_model()

model, feature_names, model_accuracy = load_model()

# ----------------------------------------
# APP
# ----------------------------------------
app = FastAPI()

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/logo",   StaticFiles(directory="logo"),   name="logo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------------------
# PYDANTIC MODELS
# ----------------------------------------
class UserLogin(BaseModel):
    username: str
    password: str

class PasswordChange(BaseModel):
    username: str
    current_password: str
    new_password: str

class OTPVerify(BaseModel):
    username: str
    otp: str

class PredictRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_type: str = "Not specified"
    top_n: int = 5

class FertilizerRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    crop: str

class CropHistorySave(BaseModel):
    username: str
    crop_name: str
    confidence: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class DiseaseHistorySave(BaseModel):
    username: str
    disease_name: str
    confidence: float
    treatment: str

class UserUpdate(BaseModel):
    address: Optional[str] = None

# ----------------------------------------
# AUTH ROUTES  (plain def — FastAPI runs these in threadpool automatically)
# ----------------------------------------

@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    address: Optional[str] = Form(None),
    profile_picture: Optional[UploadFile] = File(None)
):
    if users_col.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if users_col.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    picture_path = None
    if profile_picture and profile_picture.filename:
        ext = profile_picture.filename.split(".")[-1]
        fname = f"{username}_profile.{ext}"
        with open(f"static/uploads/{fname}", "wb") as f:
            shutil.copyfileobj(profile_picture.file, f)
        picture_path = f"uploads/{fname}"

    otp = generate_otp()
    otp_col.delete_many({"username": username})
    otp_col.insert_one({
        "username": username,
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=10)
    })

    users_col.insert_one({
        "username": username,
        "email": email,
        "hashed_password": hash_password(password),
        "address": address,
        "profile_picture": picture_path,
        "verified": False,
        "created_at": datetime.utcnow()
    })

    return {"status": "registered", "otp": otp, "email": email, "username": username}


@app.post("/verify-otp")
def verify_otp(req: OTPVerify):
    record = otp_col.find_one({"username": req.username})
    if not record:
        raise HTTPException(status_code=400, detail="No OTP found. Please register again.")
    if record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    if datetime.utcnow() > record["expires"]:
        raise HTTPException(status_code=400, detail="OTP expired. Please register again.")

    users_col.update_one({"username": req.username}, {"$set": {"verified": True}})
    otp_col.delete_many({"username": req.username})

    user = users_col.find_one({"username": req.username})
    token = create_token(req.username)
    return {
        "message": "Email verified",
        "access_token": token,
        "user": {
            "username": user["username"],
            "email": user["email"],
            "address": user.get("address"),
            "profile_picture": user.get("profile_picture")
        }
    }


@app.post("/resend-otp")
def resend_otp(data: dict):
    username = data.get("username")
    user = users_col.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    otp = generate_otp()
    otp_col.delete_many({"username": username})
    otp_col.insert_one({
        "username": username,
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=10)
    })
    return {"otp": otp, "email": user["email"], "username": username}


@app.post("/login")
def login(user: UserLogin):
    db_user = users_col.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not db_user.get("verified", False):
        raise HTTPException(status_code=403, detail="Email not verified. Please verify your account.")
    token = create_token(user.username)
    return {
        "message": "Login successful",
        "access_token": token,
        "user": {
            "username": db_user["username"],
            "email": db_user.get("email"),
            "address": db_user.get("address"),
            "profile_picture": db_user.get("profile_picture")
        }
    }


@app.post("/change-password")
def change_password(req: PasswordChange):
    db_user = users_col.find_one({"username": req.username})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, db_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    users_col.update_one(
        {"username": req.username},
        {"$set": {"hashed_password": hash_password(req.new_password)}}
    )
    return {"message": "Password updated successfully"}


@app.put("/users/{username}")
def update_profile(username: str, req: UserUpdate):
    users_col.update_one({"username": username}, {"$set": {"address": req.address}})
    user = users_col.find_one({"username": username})
    return {
        "username": user["username"],
        "address": user.get("address"),
        "profile_picture": user.get("profile_picture")
    }


@app.post("/users/{username}/profile-picture")
def update_profile_picture(username: str, file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    fname = f"{username}_profile.{ext}"
    os.makedirs("static/uploads", exist_ok=True)
    with open(f"static/uploads/{fname}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    users_col.update_one({"username": username}, {"$set": {"profile_picture": f"uploads/{fname}"}})
    return {"profile_picture": f"uploads/{fname}"}

# ----------------------------------------
# HISTORY ROUTES
# ----------------------------------------

@app.post("/save-crop-history")
def save_crop_history(req: CropHistorySave):
    crop_hist_col.insert_one({
        "username": req.username, "crop_name": req.crop_name,
        "confidence": req.confidence, "temperature": req.temperature,
        "humidity": req.humidity, "ph": req.ph, "rainfall": req.rainfall,
        "timestamp": datetime.utcnow()
    })
    return {"status": "success"}


@app.post("/save-disease-history")
def save_disease_history(req: DiseaseHistorySave):
    disease_hist_col.insert_one({
        "username": req.username, "disease_name": req.disease_name,
        "confidence": req.confidence, "treatment": req.treatment,
        "timestamp": datetime.utcnow()
    })
    return {"status": "success"}


@app.get("/get-user-history/{username}")
def get_user_history(username: str):
    crops = list(crop_hist_col.find({"username": username}, {"_id": 0}).sort("timestamp", -1).limit(50))
    diseases = list(disease_hist_col.find({"username": username}, {"_id": 0}).sort("timestamp", -1).limit(50))
    return {"crops": crops, "diseases": diseases}

# ----------------------------------------
# ML ROUTES
# ----------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, current_user: str = Depends(get_current_user)):
    warnings = []
    if req.temperature > 60:
        warnings.append("Extreme heat detected.")
    if req.ph < 0 or req.ph > 14:
        warnings.append("pH must be 0-14.")
    row = {
        "N": req.nitrogen, "P": req.phosphorus, "K": req.potassium,
        "temperature": req.temperature, "humidity": req.humidity,
        "ph": req.ph, "rainfall": req.rainfall
    }
    X = pd.DataFrame([row], columns=feature_names)
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    idx = np.argsort(probs)[::-1][:req.top_n]
    
    # ----------------------------------------------------
    # EXPLAINABILITY (SHAP)
    # ----------------------------------------------------
    explanation = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        
        # Random Forest SHAP values are usually a list of arrays (one for each class)
        top_idx = idx[0]
        if isinstance(shap_vals, list):
            class_shap = shap_vals[top_idx][0]
        elif len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, top_idx]
        else:
            class_shap = shap_vals[0]
            
        for i, fname in enumerate(feature_names):
            explanation.append({
                "feature": fname, 
                "contribution": float(class_shap[i])
            })
        
        # Sort by absolute contribution to show most impactful features first
        explanation = sorted(explanation, key=lambda x: abs(x["contribution"]), reverse=True)
    except Exception as e:
        explanation = [{"error": f"SHAP explanation failed: {str(e)}"}]

    return {
        "metadata": {
            "accuracy": model_accuracy, 
            "soil_type": req.soil_type, 
            "warnings": warnings or None,
            "explainability": {
                "top_crop_explained": classes[idx[0]],
                "feature_contributions": explanation
            }
        },
        "suggestions": [{"crop": classes[i], "probability": float(probs[i])} for i in idx]
    }


@app.post("/predict-fertilizer")
def predict_fertilizer(req: FertilizerRequest, current_user: str = Depends(get_current_user)):
    ideal = {"N": 40, "P": 40, "K": 40}
    n_diff = ideal["N"] - req.nitrogen
    p_diff = ideal["P"] - req.phosphorus
    k_diff = ideal["K"] - req.potassium
    if n_diff > 10:
        rec = "Urea or Ammonium Nitrate to boost Nitrogen."
    elif p_diff > 10:
        rec = "DAP or Single Superphosphate for Phosphorus."
    elif k_diff > 10:
        rec = "MOP or Potassium Sulfate to increase Potassium."
    else:
        rec = "Soil is optimal. Use balanced 10-10-10 compost."
    return {"crop": req.crop, "recommendation": rec, "status": "Verified"}


@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user)
):
    contents = await file.read()
    result = predict_disease_from_image(contents, crop=crop or "")
    if "error" in result:
        return result
    return {
        "disease":    result["disease"],
        "confidence": result["confidence"],
        "treatment":  result["treatment"],
        "fertilizer": result["fertilizer"],
        "method":     result.get("method", "")
    }

# ----------------------------------------
# CHATBOT ROUTE (Grok API)
# ----------------------------------------

class ChatMessage(BaseModel):
    message: str
    history: list = []

SYSTEM_PROMPT = """You are FarmAI Assistant, an expert agricultural advisor built into the FarmAI platform.

CRITICAL FACTS — NEVER get these wrong:
- FarmAI's crop recommendation model supports EXACTLY 22 crops (not 3, not 5, not any other number)
- The model recommends the TOP 5 best crops from these 22 based on soil and climate inputs
- The complete list of 22 supported crops is:
  Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Blackgram,
  Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange,
  Papaya, Coconut, Cotton, Jute, Coffee

If anyone asks "how many crops does FarmAI support/recommend/suggest" — answer: 22 crops.
If anyone asks to list the crops — list all 22 names above.
If anyone asks what the model predicts — it picks the best 5 from 22 crops.

ABOUT FARMAI:
- Uses Random Forest ML model trained on soil & climate data
- Inputs: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Model accuracy: ~99%
- Also has: Plant Disease Detection (Gemini AI), Fertilizer Advisor, History Tracking

You also help farmers with crop selection, disease treatment, fertilizer advice,
irrigation, pest control, harvesting tips, and any agriculture-related knowledge.
Be friendly, concise, practical. Keep responses under 200 words unless more detail is needed."""

@app.post("/chat")
def chat(req: ChatMessage):
    if not GROK_API_KEY:
        return {"reply": "Chatbot is not configured. Please add GROK_API_KEY to .env file."}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 400, "temperature": 0.7},
            timeout=20
        )
        if resp.status_code == 400:
            return {"reply": f"API Error: {resp.json().get('error', {}).get('message', 'Bad request.')}"}
        if resp.status_code == 401:
            return {"reply": "Invalid API key. Please update GROK_API_KEY in your .env file."}
        resp.raise_for_status()
        return {"reply": resp.json()["choices"][0]["message"]["content"]}
    except http_requests.exceptions.Timeout:
        return {"reply": "Sorry, the response took too long. Please try again."}
    except Exception as e:
        return {"reply": f"Connection error: {str(e)[:80]}"}


@app.post("/chat-file")
async def chat_file(
    file: UploadFile = File(...),
    message: str = Form(default="Analyze this and give agricultural advice.")
):
    import base64
    contents = await file.read()
    fname = file.filename.lower()
    ext = fname.rsplit(".", 1)[-1] if "." in fname else ""

    # ---- IMAGE: use Gemini Vision ----
    if ext in ("jpg", "jpeg", "png", "webp", "gif"):
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            return {"reply": "Gemini API key not configured for image analysis."}
        b64 = base64.b64encode(contents).decode("utf-8")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        prompt = f"""{SYSTEM_PROMPT}

User uploaded an image and asks: "{message}"
Analyze the image from an agricultural perspective. Identify:
- What crop or plant is shown
- Any visible diseases, pests, or deficiencies
- Health status
- Recommended treatment or action
Be specific and practical."""

        try:
            resp = http_requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}",
                json={"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime, "data": b64}}]}],
                      "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}},
                timeout=25
            )
            resp.raise_for_status()
            reply = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {"reply": reply}
        except Exception as e:
            return {"reply": f"Image analysis failed: {str(e)[:80]}"}

    # ---- PDF: extract text then send to Groq ----
    elif ext == "pdf":
        try:
            import pypdf
            import io as _io
            reader = pypdf.PdfReader(_io.BytesIO(contents))
            text = "\n".join(p.extract_text() or "" for p in reader.pages[:5]).strip()
            if not text:
                return {"reply": "Could not extract text from this PDF. It may be a scanned image PDF."}
            text = text[:3000]  # limit tokens
        except Exception as e:
            return {"reply": f"Could not read PDF: {str(e)[:80]}"}

        user_msg = f'User uploaded a document and asks: "{message}"\n\nDocument content:\n{text}'

    # ---- TXT / CSV ----
    elif ext in ("txt", "csv"):
        try:
            text = contents.decode("utf-8", errors="ignore")[:3000]
        except:
            return {"reply": "Could not read this file."}
        user_msg = f'User uploaded a text file and asks: "{message}"\n\nFile content:\n{text}'

    else:
        return {"reply": f"Unsupported file type '.{ext}'. Please upload an image (JPG/PNG), PDF, or TXT file."}

    # Send document text to Groq
    if not GROK_API_KEY:
        return {"reply": "Chatbot API key not configured."}
    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                  "max_tokens": 500, "temperature": 0.5},
            timeout=20
        )
        resp.raise_for_status()
        return {"reply": resp.json()["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"reply": f"Analysis failed: {str(e)[:80]}"}

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.7
            },
            timeout=20
        )
        if resp.status_code == 400:
            err = resp.json().get("error", {})
            return {"reply": f"API Error: {err.get('message', 'Bad request. Check your API key.')}"}
        if resp.status_code == 401:
            return {"reply": "Invalid API key. Please update GROK_API_KEY in your .env file."}
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        return {"reply": reply}
    except http_requests.exceptions.Timeout:
        return {"reply": "Sorry, the response took too long. Please try again."}
    except Exception as e:
        return {"reply": f"Sorry, I'm having trouble connecting. Please try again. ({str(e)[:80]})"}


# ----------------------------------------
# HTML SERVING
# ----------------------------------------

@app.get("/")
def serve_home(): return FileResponse("crop ui.html")

@app.get("/login")
@app.get("/login-page")
@app.get("/login.html")
def serve_login(): return FileResponse("login.html")

@app.get("/register")
@app.get("/register-page")
@app.get("/register.html")
def serve_register(): return FileResponse("register.html")

@app.get("/verify")
@app.get("/verify.html")
def serve_verify(): return FileResponse("verify.html")

@app.get("/profile")
@app.get("/profile-page")
@app.get("/profile.html")
def serve_profile(): return FileResponse("profile.html")

@app.get("/results")
@app.get("/results-page")
@app.get("/results.html")
def serve_results(): return FileResponse("results.html")

@app.get("/change-password")
@app.get("/change-password-page")
@app.get("/change_password.html")
def serve_change_password(): return FileResponse("change_password.html")

@app.get("/disease")
@app.get("/disease-page")
@app.get("/disease.html")
def serve_disease(): return FileResponse("disease.html")

@app.get("/history")
@app.get("/history-page")
@app.get("/history.html")
def serve_history(): return FileResponse("history.html")

@app.get("/fertilizer")
@app.get("/fertilizer-page")
@app.get("/fertilizer.html")
def serve_fertilizer(): return FileResponse("fertilizer.html")

@app.get("/chatbot.html")
def serve_chatbot(): return FileResponse("chatbot.html")
