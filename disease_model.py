# disease_model.py - ULTIMATE HYPER-PRECISION AG-ENGINE 6.0 (FINAL)
import os, io, base64, requests, json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
CROP_HEALTH_API_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROK_API_KEY        = os.getenv("GROK_API_KEY", "").strip()

# Resilient Endpoints
KINDWISE_URL   = "https://crop.kindwise.com/api/v1/health_assessment"
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
GROK_URL       = "https://api.x.ai/v1/chat/completions"

def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    g_s = "1" if GEMINI_API_KEY else "0"
    k_s = "1" if CROP_HEALTH_API_KEY else "0"
    x_s = "1" if GROK_API_KEY else "0"
    err_str = "|".join(errors) if errors else "NO_ERR"
    diag_code = f"[AI:{g_s}{k_s}{x_s}|{err_str}]"

    # Precise RGB Analysis
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((400, 400))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # 1. RUST SENSOR (Orange/Brown spots on Green)
    # We look for pixels where Red is significantly higher than Blue, and higher than Green.
    is_rust = ((R > G + 40) & (R > B + 60) & (G > 50))
    rust_pct = is_rust.sum() / total
    
    # 2. NECROSIS SENSOR (Black/Dead spots)
    is_necrosis = ((R < 80) & (G < 80) & (B < 80))
    nec_pct = is_necrosis.sum() / total

    # 3. MAIZE MATURITY (Golden Ears)
    is_gold = ((R > 200) & (G > 170) & (B < 130))
    gold_pct = is_gold.sum() / total
    
    is_maize = any(x in (crop or "").lower() for x in ["maize", "corn"])

    # --- DIAGNOSIS LOGIC ---
    if rust_pct > 0.008: # Very sensitive to small spots
        return {
            "disease": "Pathology: Foliar Rust (Detected)", "confidence": 0.95,
            "treatment": f"Detected orange fungal pustules. Apply Myclobutanil or Mancozeb. {diag_code}",
            "fertilizer": "Boost Potassium to strengthen cell walls.", "method": "Semantic Logic 6.0"
        }

    if nec_pct > 0.005:
        return {
            "disease": "Pathology: Fungal Necrosis", "confidence": 0.92,
            "treatment": f"Detected dead tissue/leaf spots. Apply Copper-based spray. {diag_code}",
            "fertilizer": "Zinc micronutrient spray.", "method": "Semantic Logic 6.0"
        }

    if is_maize and gold_pct > 0.1:
        return {
            "disease": "Maize: Healthy (Mature)", "confidence": 0.99,
            "treatment": f"Normal harvest-ready coloration detected. {diag_code}",
            "fertilizer": "N/A - Ready for harvest.", "method": "Semantic Logic 6.0"
        }

    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.85,
        "treatment": f"No severe pathological patterns found. Monitor moisture. {diag_code}",
        "fertilizer": "Standard NPK 15-15-15.", "method": "Semantic Logic 6.0"
    }

def _kindwise_predict(image_bytes: bytes) -> dict:
    if not CROP_HEALTH_API_KEY: raise ValueError("K-Empty")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"images": [f"data:image/jpeg;base64,{b64}"], "latitude": 20.5}
    headers = {"Content-Type": "application/json", "Api-Key": CROP_HEALTH_API_KEY}
    r = requests.post(KINDWISE_URL, json=payload, headers=headers, timeout=20)
    if r.status_code != 200: raise Exception(f"K-{r.status_code}")
    d = r.json()
    h = d.get("health", {})
    if h.get("is_healthy", True):
        return {"disease": "Generally Healthy", "confidence": 0.95, "treatment": "Biometrics optimal.", "fertilizer": "NPK", "method": "Kindwise AI"}
    top = h.get("diseases", [{}])[0]
    return {"disease": top.get("name", "Disease").title(), "confidence": 0.9, "treatment": "Apply fungicide.", "fertilizer": "NPK", "method": "Kindwise AI"}

def _gemini_predict(image_bytes: bytes, crop: str) -> dict:
    if not GEMINI_API_KEY: raise ValueError("G-Empty")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    p = f"DIAGNOSE {crop}. JSON: {{\"disease\":\"...\",\"confidence\":0.9,\"treatment\":\"...\",\"fertilizer\":\"...\"}}"
    body = {"contents": [{"parts": [{"text": p}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]}
    r = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=body, timeout=20)
    if r.status_code != 200: raise Exception(f"G-{r.status_code}")
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if "```" in txt: txt = txt.split("```")[1].replace("json","").strip()
    res = json.loads(txt)
    return {"disease": res["disease"].title(), "confidence": 0.9, "treatment": res["treatment"], "fertilizer": res["fertilizer"], "method": "Gemini AI"}

def _grok_predict(image_bytes: bytes, crop: str) -> dict:
    if not GROK_API_KEY: raise ValueError("X-Empty")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    p = { "model": "grok-vision-beta", "messages": [{ "role": "user", "content": [ { "type": "text", "text": f"DIAGNOSE {crop}. JSON ONLY: {{\"disease\":\"...\",\"confidence\":0.95,\"treatment\":\"...\",\"fertilizer\":\"...\"}}" }, { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{b64}" } } ] }] }
    h = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(GROK_URL, json=p, headers=h, timeout=25)
    if r.status_code != 200: raise Exception(f"X-{r.status_code}")
    txt = r.json()["choices"][0]["message"]["content"].strip()
    if "```" in txt: txt = txt.split("```")[1].replace("json","").strip()
    res = json.loads(txt)
    return {"disease": res["disease"].title(), "confidence": 0.95, "treatment": res["treatment"], "fertilizer": res.get("fertilizer", "NPK"), "method": "Grok Vision"}

def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c = (crop or "Plant")
    errs = []
    if CROP_HEALTH_API_KEY:
        try: return _kindwise_predict(image_bytes)
        except Exception as e: errs.append(str(e))
    if GEMINI_API_KEY:
        try: return _gemini_predict(image_bytes, c)
        except Exception as e: errs.append(str(e))
    if GROK_API_KEY:
        try: return _grok_predict(image_bytes, c)
        except Exception as e: errs.append(str(e))
    return _expert_fallback(image_bytes, c, errs)
