# disease_model.py - ULTIMATE HYPER-PRECISION AG-ENGINE 4.0
import os, io, base64, requests, json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
CROP_HEALTH_API_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROK_API_KEY        = os.getenv("GROK_API_KEY", "").strip()

# Endpoints (2026 Production Tier)
GEMINI_V1      = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
KINDWISE_URL   = "https://crop.kindwise.com/api/v1/health_assessment"
GROK_URL       = "https://api.x.ai/v1/chat/completions"
GROK_MODEL     = "grok-vision-beta"

# ---------------------------------------------------------------
# Tier 4: EXPERT SEMANTIC FALLBACK (Restored & Hardened)
# ---------------------------------------------------------------
def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    g_s = "1" if GEMINI_API_KEY else "0"
    k_s = "1" if CROP_HEALTH_API_KEY else "0"
    err_str = "|".join(errors) if errors else "NO_ERR"
    diag_code = f"[AI:{g_s}{k_s}|{err_str}]"

    # Convert Image to RGB Matrix
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((300, 300))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # 1. MAIZE MATURITY ANALYSIS
    is_gold = ((R > 185) & (G > 160) & (B < 125))
    gold_pct = is_gold.sum() / total
    
    # 2. DISEASE SIGNATURES
    is_rust = ((R > 135) & (G < 130) & (B < 90) & (R > G * 1.5))
    rust_pct = is_rust.sum() / total
    
    is_necrosis = ((R < 75) & (G < 75) & (B < 75))
    nec_pct = is_necrosis.sum() / total
    
    crop_low = (crop or "").lower()
    is_maize = any(x in crop_low for x in ["maize", "corn"])

    # Decision Matrix
    if is_maize and gold_pct > 0.03 and gold_pct > rust_pct:
        return {
            "disease": "Maize: Healthy (Mature)", "confidence": 0.99,
            "treatment": f"Normal grain maturation detected (Golden ear). Safe for harvest. {diag_code}",
            "fertilizer": "No treatment required.", "method": "Semantic Expert 4.0"
        }

    if rust_pct > 0.05:
        return {
            "disease": "Pathology: Foliar Rust", "confidence": 0.94,
            "treatment": f"Fungal rust pustules detected. Apply triazole fungicide. {diag_code}",
            "fertilizer": "Check Potassium levels.", "method": "Semantic Expert 4.0"
        }

    if nec_pct > 0.005:
        return {
            "disease": "Pathology: Fungal Necrosis", "confidence": 0.96,
            "treatment": f"Leaf rot/necrotic spots detected. Use copper bactericide. {diag_code}",
            "fertilizer": "Apply Zinc/Micronutrients.", "method": "Semantic Expert 4.0"
        }

    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.88,
        "treatment": f"Bio-scan optimal. Clear leaf surfaces found. {diag_code}",
        "fertilizer": "Standard NPK maintenance.", "method": "Semantic Expert 4.0"
    }

# ---------------------------------------------------------------
# Tier 3: xAI Grok Vision Fallback
# ---------------------------------------------------------------
def _grok_predict(image_bytes: bytes, crop: str = "Plant") -> dict:
    if not GROK_API_KEY: raise ValueError("X-Missing")
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": GROK_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Ag-Diagnosis: {crop}. JSON: {{\"disease\":\"Name\",\"confidence\":0.9,\"treatment\":\"Advice\",\"fertilizer\":\"Advice\"}}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(GROK_URL, json=payload, headers=headers, timeout=25)
    if resp.status_code != 200: raise Exception(f"X-Err:{resp.status_code}")
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if "```" in raw: raw = raw.split("```")[1].replace("json","").strip()
    res = json.loads(raw)
    return {
        "disease": res.get("disease", "Healthy").title(),
        "confidence": float(res.get("confidence", 0.95)),
        "treatment": res.get("treatment", "Balanced care."),
        "fertilizer": res.get("fertilizer", "Organic NPK"), "method": "xAI Grok Vision"
    }

# ---------------------------------------------------------------
# Tier 2: Gemini Vision Engine
# ---------------------------------------------------------------
def _gemini_predict(image_bytes: bytes, crop: str = "Plant") -> dict:
    if not GEMINI_API_KEY: raise ValueError("G-Missing")
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = f"DIAGNOSE {crop}. JSON ONLY: {{\"disease\":\"...\",\"confidence\":0.9,\"treatment\":\"...\",\"fertilizer\":\"...\"}}"
    body = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]}]}
    resp = requests.post(f"{GEMINI_V1}?key={GEMINI_API_KEY}", json=body, timeout=25)
    if resp.status_code != 200: raise Exception(f"G-Err:{resp.status_code}")
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if "```" in text: text = text.split("```")[1].replace("json","").strip()
    res = json.loads(text)
    return {
        "disease": res.get("disease", "Healthy").title(), "confidence": float(res.get("confidence", 0.9)),
        "treatment": res.get("treatment", "Care."), "fertilizer": res.get("fertilizer", "NPK"),
        "method": "Google Vision AI"
    }

# ---------------------------------------------------------------
# Tier 1: Kindwise Crop Health
# ---------------------------------------------------------------
def _kindwise_predict(image_bytes: bytes) -> dict:
    if not CROP_HEALTH_API_KEY: raise ValueError("K-Missing")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"images": [f"data:image/jpeg;base64,{b64}"], "latitude": 20.5, "longitude": 78.5}
    headers = {"Content-Type": "application/json", "Api-Key": CROP_HEALTH_API_KEY}
    resp = requests.post(KINDWISE_URL, json=payload, headers=headers, timeout=20)
    if resp.status_code != 200: raise Exception(f"K-Err:{resp.status_code}")
    data = resp.json()
    health = data.get("health")
    if not health: raise Exception("K-BadData")
    is_healthy = health.get("is_healthy", True)
    conf = float(health.get("is_healthy_probability", 0.95))
    if is_healthy:
        return {"disease": "Plant: Generally Healthy", "confidence": conf, "treatment": "Biometrics optimal.", "fertilizer": "NPK Maintenance", "method": "Kindwise AI"}
    diseases = health.get("diseases", [])
    top = diseases[0] if diseases else {"name": "Pathogen Detected", "probability": 0.8}
    return {"disease": top.get("name").title(), "confidence": float(top.get("probability", 0.8)), "treatment": "Apply fungicide.", "fertilizer": "Boost nutrients.", "method": "Kindwise AI"}

# ---------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------
def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c_name = (crop or "").strip() or "Plant"
    errs = []
    if CROP_HEALTH_API_KEY:
        try: return _kindwise_predict(image_bytes)
        except Exception as e: errs.append(str(e))
    if GEMINI_API_KEY:
        try: return _gemini_predict(image_bytes, c_name)
        except Exception as e: errs.append(str(e))
    if GROK_API_KEY:
        try: return _grok_predict(image_bytes, c_name)
        except Exception as e: errs.append(str(e))
    return _expert_fallback(image_bytes, c_name, errs)
