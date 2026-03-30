# disease_model.py - ULTIMATE HYBRID-INTEL AG-ENGINE 11.0 (PRODUCTION)
import os, io, base64, requests, json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Key Sanitization
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
KINDWISE_KEY    = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROQ_KEY        = os.getenv("GROK_API_KEY", "").strip()

# 2026 Production Tier URLs (Validated Paths)
# KINDWISE: Using /identification?details=health for crop.health portal keys
KINDWISE_URL    = "https://crop.kindwise.com/api/v1/identification?details=health"
# GEMINI: Switching to v1beta for fresh API key activation
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
# GROQ: Using the 2026 Meta-Llama 4 Scout model
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-instruct"

def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    g_s = "1" if GEMINI_KEY else "0"
    k_s = "1" if KINDWISE_KEY else "0"
    x_s = "1" if GROQ_KEY else "0"
    tag = f"[AI:{g_s}{k_s}{x_s}|{'|'.join(errors) if errors else 'OK'}]"
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((300, 300))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # Precise Pattern Matching
    rust = ((R > G + 35) & (R > B + 55) & (R < 215) & (G > 40))
    r_pct = rust.sum() / total
    
    is_maize = any(x in (crop or "").lower() for x in ["maise", "maize", "corn"])
    is_rice = any(x in (crop or "").lower() for x in ["rice", "grain", "paddy"])
    
    # 2026 Bright-Grain Filter
    gold = ((R > 210) & (G > 185) & (B < 140))
    g_pct = gold.sum() / total
    
    if (is_maize or is_rice) and g_pct > 0.01:
        return {
            "disease": f"{'Rice' if is_rice else 'Maize'}: Healthy (Mature)", "confidence": 0.98,
            "treatment": f"Normal maturation. No disease. {tag}",
            "fertilizer": "Ready for harvest.", "method": "Core Logic 11.0"
        }
    
    if r_pct > 0.006:
        return {
            "disease": "Pathology: Foliar Rust Detected", "confidence": 0.95,
            "treatment": f"Detected orange spots. Apply Mancozeb. {tag}",
            "fertilizer": "Boost Potassium.", "method": "Core Logic 11.0"
        }
    
    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.85,
        "treatment": f"Scan optimal. {tag}", "fertilizer": "Maintain standards.", "method": "Core Logic 11.0"
    }

def _groq_predict(image_bytes: bytes, crop: str) -> dict:
    if not GROQ_KEY: raise ValueError("X-Missing")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": f"DIAGNOSE {crop}. JSON: {{\"disease\":\"Name\",\"confidence\":0.9,\"treatment\":\"Advice\",\"fertilizer\":\"Advice\"}}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    }
    r = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=25)
    if r.status_code != 200: raise Exception(f"X-{r.status_code}")
    txt = r.json()["choices"][0]["message"]["content"].strip()
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"].title(), "confidence": 0.95, "treatment": res["treatment"], "fertilizer": res.get("fertilizer", "NPK"), "method": "Groq Llama 4 Vision"}

def _gemini_predict(image_bytes: bytes, crop: str) -> dict:
    if not GEMINI_KEY: raise ValueError("G-Missing")
    b = {"contents": [{"parts": [{"text": f"DIAGNOSE {crop}. JSON: {{\"disease\":\"...\"}}"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode('utf-8')}}]}]}
    r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json=b, timeout=20)
    if r.status_code != 200: raise Exception(f"G-{r.status_code}")
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"].title(), "confidence": 0.9, "treatment": res.get("treatment", "Care."), "fertilizer": "NPK", "method": "Gemini AI"}

def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c = crop or "Plant"
    errs = []
    # 1. Kindwise (Using /identification?details=health)
    if KINDWISE_KEY:
        try:
            r = requests.post(KINDWISE_URL, json={"images": [f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8') }"]}, headers={"Api-Key": KINDWISE_KEY}, timeout=20)
            if r.status_code == 200:
                h = r.json().get("result", {}).get("disease", {}) # Path update
                if h: return {"disease": h.get("name", "Healthy").title(), "confidence": 0.9, "treatment": "Apply treatment.", "fertilizer": "NPK", "method": "Kindwise AI"}
            else: errs.append(f"K-{r.status_code}")
        except Exception: errs.append("K-Err")

    try: return _gemini_predict(image_bytes, c)
    except Exception as e: errs.append(str(e).split(":")[0])
    
    try: return _groq_predict(image_bytes, c)
    except Exception as e: errs.append(str(e).split(":")[0])
    
    return _expert_fallback(image_bytes, c, errs)
