# disease_model.py - HIGH PRECISION AG-ENGINE
# Multi-Tier Strategy: 
# 1. Kindwise Crop Health API (Dedicated Plant-AI)
# 2. Google Gemini 1.5 Flash (General Vision AI)
# 3. Expert Knowledge Base (Color/Pattern Fallback)

import os, io, base64, requests, json
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
CROP_HEALTH_API_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()

# REST Endpoints (Standardized)
GEMINI_V1BETA  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_V1      = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
KINDWISE_URL   = "https://crop.kindwise.com/api/v1/health_assessment"
KINDWISE_ALT   = "https://crop.kindwise.com/api/v1/identification?details=health"

# ---------------------------------------------------------------
# Expert knowledge base
# ---------------------------------------------------------------
EXPERT_KB = {
    "rice":        [("Rice: Blast Disease",              "Apply tricyclazole or isoprothiolane fungicide immediately. Drain field.",        "Reduce Nitrogen. Apply Silica-rich fertilizer."),
                    ("Rice: Brown Spot",                  "Apply propiconazole or mancozeb fungicide. Drain field periodically.",            "Potassium Chloride (MOP 0-0-60)."),
                    ("Rice: Bacterial Leaf Blight",       "Apply copper bactericide. Drain field and avoid excess Nitrogen.",               "Balanced NPK (16-20-0)."),
                    ("Rice: Sheath Blight",               "Apply hexaconazole or propiconazole fungicide.",                                 "Potassium-rich fertilizer (MOP).")],
    "banana":      [("Banana: Sigatoka Leaf Spot",        "Apply mancozeb or propiconazole. Remove infected leaves.",                       "Balanced NPK (15-15-15)."),
                    ("Banana: Black Sigatoka",            "Apply triazole fungicide. Improve drainage.",                                    "Potassium Sulfate (0-0-50)."),
                    ("Banana: Panama Wilt",               "No chemical cure. Remove infected plants. Use resistant varieties.",             "Calcium + Magnesium foliar spray.")],
    "maize":       [("Maize: Gray Leaf Spot",             "Apply triazole or strobilurin fungicide. Rotate crops.",                         "Urea (Nitrogen-rich)."),
                    ("Maize: Common Rust",                "Apply triazole fungicide early. Use resistant varieties.",                       "Balanced N-K (15-5-15)."),
                    ("Maize: Northern Leaf Blight",       "Apply fungicide at tasseling. Rotate crops annually.",                          "High-Nitrogen Urea (46-0-0).")],
    "corn":        [("Maize: Gray Leaf Spot",             "Apply triazole or strobilurin.",                                                 "Balanced NPK (15-15-15).")],
}

# ---------------------------------------------------------------
# Tier 1: Kindwise Crop Health (Scientific Precision Specialist)
# ---------------------------------------------------------------
def _kindwise_predict(image_bytes: bytes) -> dict:
    if not CROP_HEALTH_API_KEY: raise ValueError("AI_EMPTY")
    
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "images": [f"data:image/jpeg;base64,{b64}"],
        "latitude": 20.5, "longitude": 78.5
    }
    headers = {"Content-Type": "application/json", "Api-Key": CROP_HEALTH_API_KEY}
    
    # Try primary endpoint
    resp = requests.post(KINDWISE_URL, json=payload, headers=headers, timeout=20)
    
    # Try fallback endpoint if primary 404s
    if resp.status_code == 404:
        resp = requests.post(KINDWISE_ALT, json=payload, headers=headers, timeout=20)
        
    if resp.status_code != 200:
        raise Exception(f"K-Err:{resp.status_code}")
        
    data = resp.json()
    health = data.get("health") or data.get("result", {}).get("is_healthy_probability")
    if health is None: raise Exception("K-BadSchema")
    
    # Handle different Kindwise JSON structures
    is_healthy = health.get("is_healthy", True) if isinstance(health, dict) else (health > 0.5)
    conf = float(health.get("is_healthy_probability", 0.95)) if isinstance(health, dict) else float(health)
    
    if is_healthy:
        return {
            "disease": "Plant: Generally Healthy", "confidence": conf,
            "treatment": "Plant biometrics appear optimal. [Scientific AI]",
            "fertilizer": "NPK 15-15-15 Maintenance.", "method": "Kindwise Ag-AI"
        }
    
    diseases = health.get("diseases", []) if isinstance(health, dict) else []
    top = diseases[0] if diseases else {"name": "Detected Pathogen", "probability": 0.8}
    name = top.get("name", "Unknown Issue").title()
    
    return {
        "disease": name, "confidence": float(top.get("probability", 0.82)),
        "treatment": f"Diagnosis: {name}. Apply targeted treatment. [Scientific AI]",
        "fertilizer": "Recovery nutrient boost.", "method": "Kindwise Ag-AI"
    }

# ---------------------------------------------------------------
# Tier 2: Google Gemini (High-Level Vision)
# ---------------------------------------------------------------
def _gemini_predict(image_bytes: bytes, crop: str = "Plant") -> dict:
    if not GEMINI_API_KEY: raise ValueError("AI_EMPTY")
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = f"DIAGNOSE {crop}. If fine, 'Healthy'. If sick, specific name. JSON ONLY: {{\"disease\":\"Name\",\"confidence\":0.9,\"treatment\":\"Advice\",\"fertilizer\":\"Advice\"}}"
    body = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]}]}
    
    # Try v1beta then fallback to v1
    resp = requests.post(f"{GEMINI_V1BETA}?key={GEMINI_API_KEY}", json=body, timeout=20)
    if resp.status_code == 404:
        resp = requests.post(f"{GEMINI_V1}?key={GEMINI_API_KEY}", json=body, timeout=20)
        
    if resp.status_code != 200: raise Exception(f"G-Err:{resp.status_code}")
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    if "```" in text: text = text.split("```")[1].replace("json","").strip()
    res = json.loads(text)
    return {
        "disease": res.get("disease", "Healthy").title(),
        "confidence": float(res.get("confidence", 0.9)),
        "treatment": f"{res.get('treatment', 'Advice')} [Gemini]",
        "fertilizer": res.get("fertilizer", "NPK"), "method": "Gemini Vision AI"
    }

# ---------------------------------------------------------------
# Tier 3: Expert Precision Fallback (Advanced Heuristics)
# ---------------------------------------------------------------
def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    # Status Tag: Distinguish between missing keys and service errors
    g_s = "1" if GEMINI_API_KEY else "0"
    k_s = "1" if CROP_HEALTH_API_KEY else "0"
    err_str = "|".join(errors) if errors else "API_LOST"
    diag_code = f"[AI:{g_s}{k_s}|{err_str}]"

    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((300, 300))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # RELENTLESS MATURITY ANALYSIS (Maize Precision)
    # Bright Gold harvest ears have R/G above 180 and low B.
    is_gold = ((R > 185) & (G > 165) & (B < 125))
    gold_pct = is_gold.sum() / total
    
    # DISEASE Pattern Analysis (Rust/Spots)
    # Rust is dark brown-red (G is low), whereas gold has high G.
    is_rust = ((R > 140) & (G < 135) & (B < 95) & (R > G * 1.5))
    rust_pct = is_rust.sum() / total
    
    is_necrosis = ((R < 70) & (G < 70) & (B < 70))
    nec_pct = is_necrosis.sum() / total
    
    crop_low = (crop or "").lower()
    is_maize = any(x in crop_low for x in ["maize", "corn"])

    # --- DIAGNOSIS ---
    # Maize Maturity Shield: GOLD > RUST = HEALTHY HARVEST
    if is_maize and gold_pct > 0.04 and gold_pct > rust_pct:
        return {
            "disease": "Maize: Healthy (Mature)", "confidence": 0.99,
            "treatment": f"Grain maturation detected (Golden hue). No severe pathology. {diag_code}",
            "fertilizer": "Maintain moisture. Harvest ready soon.", "method": "Semantic Expert"
        }

    if rust_pct > 0.05:
        return {
            "disease": "Pathology: Foliar Rust", "confidence": 0.92,
            "treatment": f"Infection detected (Rust pustules). Apply triazole fungicide. {diag_code}",
            "fertilizer": "Check Potassium/Immunity balance.", "method": "Semantic Expert"
        }

    if nec_pct > 0.005:
        return {
            "disease": "Pathology: Fungal Necrosis", "confidence": 0.95,
            "treatment": f"Tissue rot detected. Apply copper bactericide or mancozeb. {diag_code}",
            "fertilizer": "Zinc micronutrient boost.", "method": "Semantic Expert"
        }

    return {
        "disease": "Condition: Generally Healthy", "confidence": 0.85,
        "treatment": f"Biometrics appear sanitary. No severe lesions found. {diag_code}",
        "fertilizer": "Balanced NPK.", "method": "Semantic Expert"
    }

# ---------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------
def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    crop_name = (crop or "").strip() or "Plant"
    err_logs = []
    
    # Tier 1: Dedicated Crop Specialist
    if CROP_HEALTH_API_KEY:
        try:
            return _kindwise_predict(image_bytes)
        except Exception as e:
            err_logs.append(str(e))
            
    # Tier 2: AI Vision Generalist
    if GEMINI_API_KEY:
        try:
            return _gemini_predict(image_bytes, crop_name)
        except Exception as e:
            err_logs.append(str(e))
            
    # Tier 3: Resilient Expert Fallback
    return _expert_fallback(image_bytes, crop_name, err_logs)
