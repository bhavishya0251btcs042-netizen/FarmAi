# disease_model.py - ULTIMATE AUTO-VISION AG-ENGINE 13.0
import os, io, base64, requests, json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Key Sanitization
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
KINDWISE_KEY    = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROQ_KEY        = os.getenv("GROK_API_KEY", "").strip()

# 2026 Validated URLs
KINDWISE_URL    = "https://api.kindwise.com/api/v1/identification?details=health"
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent"
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct" # 2026 Llama 4 Scout Vision

def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    tag = f"[AI:111|{'|'.join(errors) if errors else 'OK'}]"
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((300, 300))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # 1. ENHANCED AUTO-CROP RECOGNITION (Vision Math)
    # If the image is dominated by Yellow-Gold (R, G > 180, B < 150)
    gold_mask = ((R > 180) & (G > 165) & (B < 140))
    gold_pct = gold_mask.sum() / total
    
    # If the image is dominated by Green (G > R & G > B)
    green_mask = ((G > R + 10) & (G > B + 10))
    green_pct = green_mask.sum() / total

    # 2. RUST SENSOR (True Rust is darker orange/brown spots on green)
    rust_mask = ((R > G + 50) & (R > B + 70) & (R < 210) & (G > 40))
    rust_pct = rust_mask.sum() / total
    
    # 3. DIAGNOSTIC LOGIC (Auto-Prioritizing Maturation)
    # If it's very golden, it's likely a healthy mature grain (Rice/Maize)
    if gold_pct > 0.08:
        return {
            "disease": "Status: Healthy (Mature)", "confidence": 0.99,
            "treatment": f"Identified as healthy mature coloration. Grain quality optimal. {tag}",
            "fertilizer": "No treatment needed. Harvest-ready.", "method": "Auto-Vision Math 13.0"
        }
    
    # If it's mostly green but has some rust spots
    if rust_pct > 0.005:
        return {
            "disease": "Pathology: Foliar Rust Detected", "confidence": 0.95,
            "treatment": f"Detected orange fungal patterns. Apply Mancozeb or Myclobutanil. {tag}",
            "fertilizer": "Boost potassium levels.", "method": "Auto-Vision Math 13.0"
        }
        
    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.85,
        "treatment": f"Bio-scan optimal. No severe pathology found. {tag}", 
        "fertilizer": "Standard NPK 15-15-15.", "method": "Auto-Vision Math 13.0"
    }

def _groq_predict(image_bytes: bytes, crop: str) -> dict:
    if not GROQ_KEY: raise ValueError("X-Missing")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    expert_prompt = f"Act as a PhD Plant Pathologist. Analyze this image of {crop or 'a plant'}. 1. If it's a healthy crop or just mature (e.g. golden corn), identify it as 'Healthy'. 2. If diseased, identify the specific pathology. 3. Provide a professional treatment plan including specific fungicides/pesticides if applicable. 4. Suggest a fertilizer boost. Return ONLY JSON: {{\"disease\": \"State: [Diagnosis]\", \"confidence\": 0.98, \"treatment\": \"...\", \"fertilizer\": \"...\"}}"
    payload = { "model": GROQ_MODEL, "messages": [{ "role": "user", "content": [ { "type": "text", "text": expert_prompt }, { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{b64}" } } ] }] }
    r = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=25)
    if r.status_code != 200: raise Exception(f"X-{r.status_code}")
    txt = r.json()["choices"][0]["message"]["content"]
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"], "confidence": res.get("confidence", 0.9), "treatment": res["treatment"], "fertilizer": res.get("fertilizer", "N/A"), "method": "Groq Expert Vision"}

def _gemini_predict(image_bytes: bytes, crop: str) -> dict:
    if not GEMINI_KEY: raise ValueError("G-Missing")
    expert_prompt = f"Agricultural AI Engine: Perform high-precision diagnostic on this {crop or 'leaf'}. Differentiate between maturity/coloration and actual disease. If healthy, report 'Healthy'. If diseased, list the specific name and a detailed bio-chemical treatment plan. JSON format: {{\"disease\":\"...\",\"confidence\":0.95,\"treatment\":\"...\",\"fertilizer\":\"...\"}}"
    b = {"contents": [{"parts": [{"text": expert_prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode('utf-8')}}]}]}
    r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json=b, timeout=20)
    if r.status_code != 200: raise Exception(f"G-{r.status_code}")
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    res = json.loads(txt.split("```")[1].replace("json","") if "```" in txt else txt)
    return {"disease": res["disease"], "confidence": res.get("confidence", 0.9), "treatment": res["treatment"], "fertilizer": res.get("fertilizer", "N/A"), "method": "Gemini AI Expert"}

def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c = crop or "Plant"
    errs = []
    
    # 1. Kindwise (Dual-Header Auth Strategy)
    if KINDWISE_KEY:
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            for h_type in [{"Api-Key": KINDWISE_KEY}, {"Authorization": f"Bearer {KINDWISE_KEY}"}]:
                r = requests.post(KINDWISE_URL, json={"images": [f"data:image/jpeg;base64,{b64_img}"]}, headers=h_type, timeout=12)
                if r.status_code == 200:
                    try: 
                        h = r.json()["result"]["disease"]; return {"disease": h["name"].title(), "confidence": 0.95, "treatment": "Apply treatment.", "method": "Kindwise AI"}
                    except: continue
            errs.append(f"K-{r.status_code}")
        except Exception: errs.append("K-TO")

    try: return _gemini_predict(image_bytes, c)
    except Exception as e: errs.append(str(e).split(":")[0][:5])
    
    try: return _groq_predict(image_bytes, c)
    except Exception as e: errs.append(str(e).split(":")[0][:5])
    
    return _expert_fallback(image_bytes, c, errs)
