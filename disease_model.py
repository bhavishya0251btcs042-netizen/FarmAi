# disease_model.py - FarmAI Expert Diagnostics Engine v14.0
# High-precision, multi-tier AI plant pathology system
import os, io, base64, requests, json, re
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
KINDWISE_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GROQ_KEY     = os.getenv("GROK_API_KEY", "").strip()

GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
KINDWISE_URL = "https://crop.kindwise.com/api/v1/identification"

# ---------------------------------------------------------------------------
# COMPREHENSIVE DISEASE TREATMENT DATABASE
# Used by all tiers to enrich results with proper treatment plans
# ---------------------------------------------------------------------------
DISEASE_DB = {
    "rust":           {"treatment": "Apply Mancozeb 75WP (2g/L water) or Propiconazole 25EC (1ml/L). Spray every 10-14 days. Remove infected leaves immediately.", "fertilizer": "Reduce Nitrogen, boost Potassium (K) with MOP at 3kg/acre."},
    "blight":         {"treatment": "Apply Copper Oxychloride (3g/L) or Metalaxyl + Mancozeb. Remove and burn affected plant parts. Improve drainage.", "fertilizer": "Apply balanced NPK 12-32-16 to boost immunity."},
    "smut":           {"treatment": "Treat seeds with Carboxin + Thiram (2g/kg seed) before planting. Remove galls before they burst. No cure after infection.", "fertilizer": "Apply Zinc Sulfate (25kg/ha) to reduce susceptibility."},
    "mosaic":         {"treatment": "No cure. Remove infected plants. Control aphid vectors with Imidacloprid (0.5ml/L). Use virus-resistant varieties.", "fertilizer": "Apply Boron (0.5kg/acre) to boost plant cell wall strength."},
    "wilt":           {"treatment": "Apply Carbendazim (1g/L) or Trichoderma viride as soil drench. Improve field drainage. Use resistant varieties.", "fertilizer": "Apply Calcium Nitrate to strengthen root systems."},
    "rot":            {"treatment": "Apply Mancozeb or Copper-based fungicides. Remove infected tissue. Avoid waterlogging. Improve air circulation.", "fertilizer": "Add Phosphorus (P) with DAP to strengthen root health."},
    "spot":           {"treatment": "Spray Chlorothalonil (2g/L) or Mancozeb 75WP every 14 days. Remove and destroy spotted leaves.", "fertilizer": "Foliar spray of Zinc + Manganese micronutrients."},
    "mildew":         {"treatment": "Apply Sulfur-based fungicide (3g/L) or Azoxystrobin. Improve air circulation. Avoid overhead irrigation.", "fertilizer": "Reduce Nitrogen (N). Apply Potassium Silicate as foliar spray."},
    "anthracnose":    {"treatment": "Apply Carbendazim (1g/L) or Copper Hydroxide. Collect and destroy fallen leaves. Avoid overhead irrigation.", "fertilizer": "Apply balanced NPK. Add Calcium foliar spray."},
    "canker":         {"treatment": "Prune infected branches 15cm below visible infection. Apply Copper Oxychloride paste on cuts. Spray Bordeaux Mixture.", "fertilizer": "Apply Calcium + Boron to prevent cell structure damage."},
    "healthy":        {"treatment": "No treatment needed. Crop appears healthy. Continue standard agronomic practices and regular monitoring.", "fertilizer": "Maintain balanced NPK schedule as per soil test recommendation."},
    "deficiency":     {"treatment": "Analyze deficiency type. Apply appropriate micronutrient (Zinc, Iron, Boron, Manganese) as foliar spray.", "fertilizer": "Conduct soil test. Apply missing nutrients — Ferrous Sulfate for iron, Zinc Sulfate for zinc."},
    "default":        {"treatment": "Consult a local agronomist. As a precaution, apply broad-spectrum fungicide Mancozeb (2g/L water).", "fertilizer": "Apply NPK 15-15-15 as a general health booster."},
}

def _get_treatment_from_db(disease_name: str) -> dict:
    """Look up treatment from local database based on disease keywords."""
    dl = disease_name.lower()
    for key in DISEASE_DB:
        if key in dl:
            return DISEASE_DB[key]
    return DISEASE_DB["default"]

def _parse_json_safely(text: str) -> dict:
    """Robustly extract JSON from AI responses that may contain markdown."""
    # Try extracting from code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except:
            pass
    # Try finding raw JSON object
    raw_json = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if raw_json:
        try:
            return json.loads(raw_json.group(0))
        except:
            pass
    raise ValueError("No valid JSON found in response")

def _preprocess_image(image_bytes: bytes, max_size: int = 800) -> bytes:
    """Resize and optimize image for best AI analysis results."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

# ---------------------------------------------------------------------------
# TIER 1: GEMINI (Primary — Best Accuracy)
# ---------------------------------------------------------------------------
def _gemini_predict(image_bytes: bytes, crop: str) -> dict:
    if not GEMINI_KEY:
        raise ValueError("G-Missing")

    expert_prompt = f"""You are a world-class Plant Pathologist AI with 20+ years of expertise.

Analyze this image of {crop or 'a plant/crop'} and provide a highly accurate diagnosis.

CRITICAL RULES:
- If the plant/crop looks HEALTHY (green, normal coloring, no visible lesions), report it as "Healthy".
- If the crop shows MATURITY signs (golden corn, ripened grain), report "Healthy - Mature Crop. Ready for Harvest."
- If diseased, name the SPECIFIC disease (e.g. "Leaf Rust", "Early Blight", "Powdery Mildew", NOT just "fungicide").
- Provide a DETAILED step-by-step treatment plan with specific chemical names and doses.
- Suggest a specific fertilizer recommendation.
- Assign a confidence percentage based on how clear the symptoms are.

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"disease": "...", "confidence": 0.95, "treatment": "...", "fertilizer": "..."}}"""

    b = {
        "contents": [{"parts": [
            {"text": expert_prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode("utf-8")}}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600}
    }
    r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json=b, timeout=25)
    if r.status_code == 429:
        raise Exception("G-429")
    if r.status_code != 200:
        raise Exception(f"G-{r.status_code}")

    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    res = _parse_json_safely(txt)

    # Enrich treatment from DB if AI gives vague response
    db = _get_treatment_from_db(res.get("disease", ""))
    treatment = res.get("treatment", "")
    if len(treatment) < 30:  # if too short/vague, use DB
        treatment = db["treatment"]

    return {
        "disease": res["disease"],
        "confidence": float(res.get("confidence", 0.93)),
        "treatment": treatment,
        "fertilizer": res.get("fertilizer") or db["fertilizer"],
        "method": "Gemini 1.5 Flash Expert"
    }

# ---------------------------------------------------------------------------
# TIER 2: GROQ LLAMA VISION (Secondary fallback)
# ---------------------------------------------------------------------------
def _groq_predict(image_bytes: bytes, crop: str) -> dict:
    if not GROQ_KEY:
        raise ValueError("X-Missing")

    expert_prompt = f"""You are an expert plant pathologist AI. Diagnose the disease in this image of {crop or 'a plant'}.

Rules:
- If healthy/mature crop (golden grain, green leaves with no spots): disease = "Healthy"
- If diseased: provide the specific disease name (e.g. "Corn Leaf Blight", "Rice Brown Spot", "Wheat Rust")
- Give a DETAILED treatment: specific chemical + dosage + application method
- Give a fertilizer recommendation

Return ONLY raw JSON (absolutely no markdown, no ```):
{{"disease": "...", "confidence": 0.9, "treatment": "...", "fertilizer": "..."}}"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": expert_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}}
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 500
    }
    r = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=30)
    if r.status_code != 200:
        raise Exception(f"X-{r.status_code}")

    txt = r.json()["choices"][0]["message"]["content"]
    res = _parse_json_safely(txt)

    db = _get_treatment_from_db(res.get("disease", ""))
    treatment = res.get("treatment", "")
    if len(treatment) < 30:
        treatment = db["treatment"]

    return {
        "disease": res["disease"],
        "confidence": float(res.get("confidence", 0.88)),
        "treatment": treatment,
        "fertilizer": res.get("fertilizer") or db["fertilizer"],
        "method": "Groq Llama 4 Expert"
    }

# ---------------------------------------------------------------------------
# TIER 3: KINDWISE (Specialized Plant Health API)
# ---------------------------------------------------------------------------
def _kindwise_predict(image_bytes: bytes) -> dict:
    if not KINDWISE_KEY:
        raise ValueError("K-Missing")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Api-Key": KINDWISE_KEY, "Content-Type": "application/json"}
    payload = {"images": [f"data:image/jpeg;base64,{b64}"], "details": "local_name,description,treatment"}

    r = requests.post(KINDWISE_URL, json=payload, headers=headers, timeout=15)
    if r.status_code != 200:
        raise Exception(f"K-{r.status_code}")

    data = r.json()
    # Navigate Kindwise response structure
    health = data.get("result", {}).get("disease", {})
    suggestions = health.get("suggestions", [])
    if not suggestions:
        raise Exception("K-NoData")

    top = suggestions[0]
    disease_name = top.get("name", "Unknown Disease").title()
    prob = float(top.get("probability", 0.8))

    db = _get_treatment_from_db(disease_name)
    details = top.get("details", {})
    treatment = details.get("treatment", {}).get("biological", [""]) or [db["treatment"]]
    treatment_str = treatment[0] if isinstance(treatment, list) else treatment
    if len(treatment_str) < 20:
        treatment_str = db["treatment"]

    return {
        "disease": disease_name,
        "confidence": prob,
        "treatment": treatment_str,
        "fertilizer": db["fertilizer"],
        "method": "Kindwise Plant Health AI"
    }

# ---------------------------------------------------------------------------
# TIER 4: LOCAL PIXEL FALLBACK (when all APIs fail)
# ---------------------------------------------------------------------------
def _expert_fallback(image_bytes: bytes, crop: str, errors: list = None) -> dict:
    err_tag = f"[Errors: {', '.join(errors)}]" if errors else ""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((400, 400))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = R.size

    # Color masks
    gold_pct  = ((R > 180) & (G > 160) & (B < 130)).sum() / total   # mature grain
    green_pct = ((G > R + 15) & (G > B + 15)).sum() / total          # healthy green
    rust_pct  = ((R > G + 45) & (R > B + 65) & (R < 215) & (G > 35)).sum() / total  # rust/orange spots
    yellow_pct = ((R > 180) & (G > 160) & (B < 80)).sum() / total    # yellowing (deficiency)
    dark_pct  = ((R < 80) & (G < 80) & (B < 80)).sum() / total       # dark lesions/necrosis

    # Decision tree
    if gold_pct > 0.12:
        return {"disease": "Healthy — Mature Crop", "confidence": 0.96,
                "treatment": "Crop appears to be at maturity stage. Harvest when moisture levels are optimal (14-18% for grains).",
                "fertilizer": "Harvest-ready. No additional fertilizer required.", "method": "Pixel Vision Fallback"}
    if dark_pct > 0.06 and rust_pct > 0.03:
        db = DISEASE_DB["rust"]
        return {"disease": "Foliar Rust / Fungal Lesions", "confidence": 0.82,
                "treatment": db["treatment"], "fertilizer": db["fertilizer"], "method": "Pixel Vision Fallback"}
    if yellow_pct > 0.10 and green_pct < 0.35:
        db = DISEASE_DB["deficiency"]
        return {"disease": "Nutrient Deficiency / Chlorosis", "confidence": 0.78,
                "treatment": db["treatment"], "fertilizer": db["fertilizer"], "method": "Pixel Vision Fallback"}
    if rust_pct > 0.008:
        db = DISEASE_DB["spot"]
        return {"disease": "Leaf Spot / Blight Symptoms", "confidence": 0.75,
                "treatment": db["treatment"], "fertilizer": db["fertilizer"], "method": "Pixel Vision Fallback"}
    if green_pct > 0.50:
        db = DISEASE_DB["healthy"]
        return {"disease": "Healthy", "confidence": 0.88,
                "treatment": f"{db['treatment']} {err_tag}",
                "fertilizer": db["fertilizer"], "method": "Pixel Vision Fallback"}

    db = DISEASE_DB["default"]
    return {"disease": "Inconclusive — Requires Field Inspection", "confidence": 0.55,
            "treatment": f"{db['treatment']} {err_tag}",
            "fertilizer": db["fertilizer"], "method": "Pixel Vision Fallback"}

# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------
def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    c = crop or "Plant"
    errs = []

    # Pre-process image for best results
    try:
        image_bytes = _preprocess_image(image_bytes)
    except Exception:
        pass  # Use original if preprocessing fails

    # TIER 1: Gemini (best accuracy)
    try:
        return _gemini_predict(image_bytes, c)
    except Exception as e:
        errs.append(f"Gemini:{str(e)[:8]}")

    # TIER 2: Groq Llama Vision
    try:
        return _groq_predict(image_bytes, c)
    except Exception as e:
        errs.append(f"Groq:{str(e)[:8]}")

    # TIER 3: Kindwise
    try:
        return _kindwise_predict(image_bytes)
    except Exception as e:
        errs.append(f"Kindwise:{str(e)[:8]}")

    # TIER 4: Local pixel fallback (always works)
    return _expert_fallback(image_bytes, c, errs)
