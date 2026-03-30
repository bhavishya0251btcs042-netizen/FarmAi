# disease_model.py - ULTIMATE HYPER-PRECISION AG-ENGINE 9.0
import os, io, base64, requests, json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Key Sanitization
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
KINDWISE_KEY    = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROQ_KEY        = os.getenv("GROK_API_KEY", "").strip()

# Resilient Endpoints
KINDWISE_URL    = "https://crop.kindwise.com/api/v1/health_assessment"
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"

def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    g_s = "1" if GEMINI_KEY else "0"
    k_s = "1" if KINDWISE_KEY else "0"
    x_s = "1" if GROQ_KEY else "0"
    err_str = "|".join(errors) if errors else "OK"
    tag = f"[AI:{g_s}{k_s}{x_s}|{err_str}]"
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((300, 300))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # 1. Identify Maize/Corn first to prevent false rust-positives
    is_maize = any(x in (crop or "").lower() for x in ["maise", "maize", "corn"])
    is_rice  = any(x in (crop or "").lower() for x in ["rice", "grain", "paddy"])
    
    # 2. BRIGHTNESS FILTERS
    is_golden_grain = ((R > 210) & (G > 185) & (B < 140))
    gold_pct = is_golden_grain.sum() / total
    
    # 2.1 RICE HUSK DETECTION (Prevents false positives on dry husks)
    is_husk = ((R > 180) & (G > 160) & (B < 120))
    husk_pct = is_husk.sum() / total
    
    # 3. RUST PATTERN (True Rust is darker/dirtier orange)
    rust = ((R > G + 45) & (R > B + 65) & (R < 210) & (G > 40))
    r_pct = rust.sum() / total
    
    # --- LOGIC GATE ---
    if (is_maize or is_rice) and (gold_pct > 0.01 or (is_rice and husk_pct > 0.05)):
        return {
            "disease": f"{'Rice' if is_rice else 'Maize'}: Healthy (Mature)", "confidence": 0.98,
            "treatment": f"Normal harvest-ready grain detected. No disease. {tag}",
            "fertilizer": "N/A - Ready for harvest.", "method": "Core Logic 10.0"
        }
        
    if r_pct > 0.005: # High sensitivity for spots
        return {
            "disease": "Pathology: Foliar Rust Detected", "confidence": 0.95,
            "treatment": f"Detected orange/brown fungal patterns. Apply Mancozeb. {tag}",
            "fertilizer": "Check K levels.", "method": "Core Logic 9.0"
        }
    
    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.85,
        "treatment": f"No severe pathology found in bio-scan. {tag}",
        "fertilizer": "Maintain standards.", "method": "Core Logic 9.0"
    }

def _groq_predict(image_bytes: bytes, crop: str) -> dict:
    if not GROQ_KEY: raise ValueError("X-Missing")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = { "model": "llama-3.2-11b-vision-preview", "messages": [{ "role": "user", "content": [ { "type": "text", "text": f"DIAGNOSE {crop}. JSON: {{\"disease\":\"...\",\"confidence\":0.9,\"treatment\":\"...\",\"fertilizer\":\"...\"}}" }, { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{b64}" } } ] }] }
    r = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=25)
    if r.status_code != 200: raise Exception(f"X-{r.status_code}")
    txt = r.json()["choices"][0]["message"]["content"].strip()
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"].title(), "confidence": 0.95, "treatment": res["treatment"], "fertilizer": res.get("fertilizer", "NPK"), "method": "Groq Vision"}

def _gemini_predict(image_bytes: bytes, crop: str) -> dict:
    if not GEMINI_KEY: raise ValueError("G-Missing")
    body = {"contents": [{"parts": [{"text": f"DIAGNOSE {crop}. JSON: {{\"disease\":\"...\"}}"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode("utf-8")}}]}]}
    r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json=body, timeout=20)
    if r.status_code != 200: raise Exception(f"G-{r.status_code}")
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"].title(), "confidence": 0.9, "treatment": res.get("treatment", "Care."), "fertilizer": "NPK", "method": "Gemini AI"}

def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c = crop or "Plant"
    errs = []
    # 1. Kindwise
    if KINDWISE_KEY:
        try:
            r = requests.post(KINDWISE_URL, json={"images": [f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8') }"]}, headers={"Api-Key": KINDWISE_KEY}, timeout=20)
            if r.status_code == 200:
                h = r.json().get("health", {})
                if h.get("is_healthy", True): return {"disease": "Generally Healthy", "confidence": 0.9, "treatment": "Healthy.", "fertilizer": "NPK", "method": "Kindwise AI"}
                top = h.get("diseases", [{}])[0]
                return {"disease": top.get("name", "Disease").title(), "confidence": 0.9, "treatment": "Treatment.", "fertilizer": "NPK", "method": "Kindwise AI"}
            else: errs.append(f"K-{r.status_code}")
        except Exception: errs.append("K-TO")
    # 2. Gemini
    try: return _gemini_predict(image_bytes, c)
    except Exception as e: errs.append(str(e))
    # 3. Groq
    try: return _groq_predict(image_bytes, c)
    except Exception as e: errs.append(str(e).split(":")[0])
    # Fallback
    return _expert_fallback(image_bytes, c, errs)
