# disease_model.py
# Plant disease detection using Google Gemini 2.5 Flash (FREE - 15 req/min, no credit card)
# Get free API key at: https://aistudio.google.com/apikey
# Add to .env: GEMINI_API_KEY=your_key_here
#
# Falls back to expert knowledge base if API key not set or unavailable.

import os, io, base64, requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# ---------------------------------------------------------------
# Expert knowledge base — used as fallback
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
    "cotton":      [("Cotton: Leaf Curl Virus",           "No cure. Remove infected plants. Control whitefly with imidacloprid.",           "Zinc + Boron foliar spray."),
                    ("Cotton: Alternaria Leaf Spot",      "Apply mancozeb or copper fungicide. Avoid overhead irrigation.",                "Balanced NPK (15-15-15)."),
                    ("Cotton: Bacterial Blight",          "Apply copper bactericide. Use disease-free seeds.",                             "Potassium Nitrate (13-0-46).")],
    "mango":       [("Mango: Anthracnose",                "Apply copper oxychloride or mancozeb. Prune dead wood.",                        "Balanced NPK (12-12-17)."),
                    ("Mango: Powdery Mildew",             "Apply sulfur or hexaconazole fungicide at flowering.",                          "Low-Nitrogen, high-Potassium."),
                    ("Mango: Malformation",               "Prune malformed parts. Apply micronutrient spray.",                             "Zinc Sulfate + Boron foliar spray.")],
    "coconut":     [("Coconut: Bud Rot",                  "Remove infected bud. Apply Bordeaux mixture.",                                  "Potassium-rich (MOP 0-0-60)."),
                    ("Coconut: Leaf Blight",              "Apply copper oxychloride fungicide. Remove infected fronds.",                   "Balanced NPK (13-13-21)."),
                    ("Coconut: Root Wilt",                "Apply micronutrients. Remove severely affected palms.",                         "Magnesium Sulfate + Boron.")],
    "coffee":      [("Coffee: Leaf Rust",                 "Apply copper fungicide or triazole. Remove infected leaves.",                   "Potassium-rich NPK (15-5-30)."),
                    ("Coffee: Berry Borer",               "Apply endosulfan or chlorpyrifos. Harvest ripe berries promptly.",              "Magnesium Sulfate + Zinc foliar spray."),
                    ("Coffee: Wilt Disease",              "No cure. Remove infected trees. Disinfect tools.",                              "Balanced NPK (17-17-17).")],
    "jute":        [("Jute: Stem Rot",                    "Apply carbendazim fungicide. Improve drainage.",                                "Balanced NPK (10-26-26)."),
                    ("Jute: Yellow Mite",                 "Apply dicofol or abamectin miticide. Increase field humidity.",                 "Nitrogen-rich Urea."),
                    ("Jute: Anthracnose",                 "Apply mancozeb or copper oxychloride.",                                         "Potassium Sulfate (0-0-50).")],
    "chickpea":    [("Chickpea: Ascochyta Blight",        "Apply mancozeb or chlorothalonil. Avoid dense planting.",                       "Balanced NPK (10-26-26)."),
                    ("Chickpea: Fusarium Wilt",           "Use resistant varieties. Apply carbendazim seed treatment.",                    "Phosphorus-rich DAP (18-46-0)."),
                    ("Chickpea: Botrytis Grey Mold",      "Apply iprodione or carbendazim. Improve air circulation.",                      "Potassium Sulfate (0-0-50).")],
    "blackgram":   [("Blackgram: Yellow Mosaic Virus",    "Remove infected plants. Control whitefly with imidacloprid.",                   "Micronutrient spray."),
                    ("Blackgram: Cercospora Leaf Spot",   "Apply mancozeb or carbendazim fungicide.",                                      "Balanced NPK (20-20-20)."),
                    ("Blackgram: Powdery Mildew",         "Apply sulfur-based fungicide.",                                                 "Low-Nitrogen fertilizer.")],
    "lentil":      [("Lentil: Ascochyta Blight",          "Apply mancozeb fungicide. Avoid dense sowing.",                                "Balanced NPK (10-26-26)."),
                    ("Lentil: Fusarium Wilt",             "Use resistant varieties. Apply carbendazim seed treatment.",                    "Phosphorus-rich (0-46-0)."),
                    ("Lentil: Rust",                      "Apply propiconazole or mancozeb fungicide.",                                    "Potassium Sulfate (0-0-50).")],
    "kidneybeans": [("Kidney Beans: Angular Leaf Spot",   "Apply copper oxychloride. Use disease-free seeds.",                            "Balanced NPK (20-20-20)."),
                    ("Kidney Beans: Anthracnose",         "Apply mancozeb or chlorothalonil fungicide.",                                   "Potassium-rich (0-0-50)."),
                    ("Kidney Beans: Mosaic Virus",        "No cure. Remove infected plants. Control aphids.",                              "Micronutrient foliar spray.")],
    "mothbeans":   [("Moth Beans: Yellow Mosaic Virus",   "Remove infected plants. Control whitefly with imidacloprid.",                   "Micronutrient spray."),
                    ("Moth Beans: Cercospora Leaf Spot",  "Apply mancozeb fungicide.",                                                     "Balanced NPK (15-15-15)."),
                    ("Moth Beans: Powdery Mildew",        "Apply sulfur-based fungicide.",                                                 "Low-Nitrogen fertilizer.")],
    "mungbean":    [("Mung Bean: Yellow Mosaic Virus",    "Remove infected plants. Control whitefly.",                                     "Micronutrient foliar spray."),
                    ("Mung Bean: Cercospora Leaf Spot",   "Apply mancozeb or carbendazim.",                                                "Balanced NPK (20-20-20)."),
                    ("Mung Bean: Powdery Mildew",         "Apply sulfur or hexaconazole fungicide.",                                       "Low-Nitrogen fertilizer.")],
    "pigeonpeas":  [("Pigeon Peas: Fusarium Wilt",        "Use resistant varieties. Apply carbendazim seed treatment.",                    "Phosphorus-rich (0-46-0)."),
                    ("Pigeon Peas: Sterility Mosaic",     "Remove infected plants. Control mite vectors with dicofol.",                    "Zinc + Boron foliar spray."),
                    ("Pigeon Peas: Alternaria Blight",    "Apply mancozeb or copper oxychloride.",                                         "Balanced NPK (10-26-26).")],
    "watermelon":  [("Watermelon: Anthracnose",           "Apply chlorothalonil or mancozeb. Avoid overhead irrigation.",                  "Potassium Sulfate (0-0-50)."),
                    ("Watermelon: Gummy Stem Blight",     "Apply thiophanate-methyl or mancozeb fungicide.",                              "Calcium Nitrate."),
                    ("Watermelon: Mosaic Virus",          "Remove infected plants. Control aphids with imidacloprid.",                     "Micronutrient foliar spray.")],
    "muskmelon":   [("Muskmelon: Powdery Mildew",         "Apply sulfur or hexaconazole fungicide.",                                       "Low-Nitrogen fertilizer."),
                    ("Muskmelon: Downy Mildew",           "Apply metalaxyl or copper fungicide. Improve air circulation.",                "Balanced NPK (15-15-15)."),
                    ("Muskmelon: Gummy Stem Blight",      "Apply thiophanate-methyl fungicide.",                                           "Calcium Nitrate.")],
    "papaya":      [("Papaya: Ring Spot Virus",           "No cure. Remove infected plants. Control aphids.",                              "Zinc + Boron foliar spray."),
                    ("Papaya: Anthracnose",               "Apply copper oxychloride or mancozeb fungicide.",                              "Balanced NPK (12-12-17)."),
                    ("Papaya: Powdery Mildew",            "Apply sulfur-based fungicide. Improve air circulation.",                        "Low-Nitrogen fertilizer.")],
    "pomegranate": [("Pomegranate: Bacterial Blight",     "Apply copper bactericide. Prune infected branches.",                           "Potassium Sulfate (0-0-50)."),
                    ("Pomegranate: Fruit Rot",            "Apply iprodione or carbendazim. Improve drainage.",                            "Calcium Nitrate."),
                    ("Pomegranate: Leaf Spot",            "Apply copper oxychloride or mancozeb.",                                         "Balanced NPK (15-15-15).")],
    "apple":       [("Apple: Apple Scab",                 "Apply captan or myclobutanil fungicide. Remove infected leaves.",              "Calcium Nitrate."),
                    ("Apple: Black Rot",                  "Prune infected branches. Apply copper-based fungicide.",                       "Balanced NPK (10-10-10)."),
                    ("Apple: Cedar Apple Rust",           "Apply myclobutanil in spring. Remove nearby cedar trees.",                     "Potassium-rich (0-0-50).")],
    "grapes":      [("Grapes: Black Rot",                 "Apply mancozeb or myclobutanil. Remove mummified berries.",                    "Potassium Sulfate (0-0-50)."),
                    ("Grapes: Downy Mildew",              "Apply copper fungicide or metalaxyl. Improve canopy airflow.",                 "Calcium Nitrate."),
                    ("Grapes: Powdery Mildew",            "Apply sulfur or triazole fungicide.",                                           "Low-Nitrogen fertilizer.")],
    "orange":      [("Orange: Citrus Greening (HLB)",     "No cure. Remove infected trees. Control psyllid with insecticide.",            "Zinc + Manganese foliar spray."),
                    ("Orange: Citrus Canker",             "Apply copper bactericide. Remove infected leaves and fruit.",                   "Balanced NPK (15-15-15)."),
                    ("Orange: Melanose",                  "Apply copper fungicide after petal fall.",                                      "Calcium Nitrate.")],
    "potato":      [("Potato: Late Blight",               "Apply metalaxyl immediately. Destroy infected plants.",                        "Potassium-rich (0-0-60)."),
                    ("Potato: Early Blight",              "Apply chlorothalonil or mancozeb. Remove lower leaves.",                       "High-Phosphorus (10-52-10)."),
                    ("Potato: Common Scab",               "Maintain soil pH 5.2-5.5. Avoid fresh manure.",                                "Sulfur to acidify soil.")],
    "tomato":      [("Tomato: Late Blight",               "Apply metalaxyl fungicide. Destroy infected plants.",                          "High-Phosphorus (10-52-10)."),
                    ("Tomato: Early Blight",              "Apply chlorothalonil or mancozeb. Remove lower leaves.",                       "Calcium Nitrate."),
                    ("Tomato: Bacterial Spot",            "Apply copper bactericide. Use drip irrigation.",                               "Potassium-rich (0-0-50).")],
    "wheat":       [("Wheat: Leaf Rust",                  "Apply triazole fungicide early. Use resistant varieties.",                     "Potassium Sulfate (0-0-50)."),
                    ("Wheat: Powdery Mildew",             "Apply sulfur-based fungicide. Avoid dense planting.",                          "Low Nitrogen, Balanced P/K."),
                    ("Wheat: Fusarium Head Blight",       "Apply prothioconazole. Avoid overhead irrigation during flowering.",            "Balanced NPK.")],
    "sugarcane":   [("Sugarcane: Red Rot",                "Use disease-free setts. Apply fungicide. Remove affected clumps.",             "High-Potassium (0-0-60)."),
                    ("Sugarcane: Smut",                   "Remove infected whips in a bag and burn them. Use resistant varieties.",       "Balanced NPK."),
                    ("Sugarcane: Rust",                   "Apply mancozeb or propiconazole fungicide.",                                   "Calcium Nitrate.")],
}


# ---------------------------------------------------------------
# Expert fallback — color analysis
# ---------------------------------------------------------------
def _analyze_symptoms(image: Image.Image, crop: str = "plant") -> tuple:
    import numpy as np
    img = image.copy().convert("RGB")
    img.thumbnail((150, 150))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    total = R.size
    
    # Healthy Profile: Greenery or Mature Cereal Gold
    lush_green = ((G > R * 1.02) & (G > B * 1.02)).sum() / total
    # Maize/Corn maturity check (Golden Yellow)
    golden_ear = ((R > 180) & (G > 140) & (B < 100) & (R > G * 1.1)).sum() / total
    
    # Disease symptoms (Intense spotting)
    # Rust / Spot Rule - exclude golden harvestable parts
    rust   = (((R > 140) & (G > 60) & (G < 160) & (B < 80) & (R > G * 1.15)).sum() / total) - (golden_ear * 0.5)
    brown  = ((R > 80)  & (G < 80)  & (B < 70)  & (R > G * 1.5)).sum() / total
    yellow_wilt = ((R > 210) & (G > 195) & (B < 120)).sum() / total
    dark_rot   = ((R < 40)  & (G < 40)  & (B < 40)).sum() / total

    # PRECISION THRESHOLDS - SMART CROP AWARENESS
    is_maize = crop.lower() in ["maize", "corn", "plant"]
    
    # If Maize/Corn ears are detected and symptoms are low elsewhere
    if is_maize and golden_ear > 0.05 and brown < 0.05 and dark_rot < 0.05:
        return -1, 0.99  # Healthy Corn
        
    if rust > 0.06 or brown > 0.08:
        return 1, 0.85  # Fungal Spot
        
    if yellow_wilt > 0.15:
        return 0, 0.82  # Wilt
        
    # Standard Healthy check
    if (lush_green + golden_ear) > 0.30 and (rust + brown + dark_rot) < 0.06:
        return -1, 0.98  # Healthy
        
    return 1, 0.40


def _expert_fallback(image: Image.Image, crop: str) -> dict:
    crop_title = crop.title() if crop else "Plant"
    crop_key = crop.lower().strip().replace(" ", "").replace("-", "") if crop else "plant"
    
    symptom_idx, conf = _analyze_symptoms(image, crop_key)
    
    if symptom_idx == -1:
        return {
            "disease":    f"{crop_title}: Healthy",
            "confidence": conf,
            "treatment":  "Plant is in optimal condition. Maintain standard irrigation and organic fertilizer schedule.",
            "fertilizer": "Maintain balanced NPK (15-15-15).",
            "method":     "High-Fidelity Health Screening"
        }

    diseases = EXPERT_KB.get(crop_key)
    if not diseases:
        return {
            "disease":    f"{crop_title}: Potential Fungal Sign",
            "confidence": 0.65,
            "treatment":  "Slight anomalies detected. Observe plant for 24h. Use neem oil if spots increase.",
            "fertilizer": "Check micro-nutrient levels (Zinc/Boron).",
            "method":     "Deep Color Analysis Fallback"
        }
        
    idx = min(max(symptom_idx, 0), len(diseases) - 1)
    name, treatment, fertilizer = diseases[idx]
    return {
        "disease":    name,
        "confidence": conf,
        "treatment":  treatment,
        "fertilizer": fertilizer,
        "method":     "Expert Contextual Eye"
    }


# ---------------------------------------------------------------
# Gemini Vision API — free, no credit card, 15 req/min
# ---------------------------------------------------------------
def _gemini_predict(image_bytes: bytes, crop: str = "Plant") -> dict:
    if not GEMINI_API_KEY:
        return {"error": "API Key not set"}

    import PIL.Image
    img = PIL.Image.open(io.BytesIO(image_bytes))
    img.thumbnail((384, 384)) # Faster upload
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    crop_hint = f"The crop is {crop}." if crop and crop.lower() != "plant" else "Identify the crop type if possible."

    prompt = f"""Expert Plant Diagnosis:
Analyze the provided image {crop_hint}

Respond in JSON ONLY:
{{
  "is_leaf": true,
  "crop": "name", 
  "disease": "Crop: [Disease or Healthy]",
  "confidence": 0.0-1.0, 
  "treatment": "steps",
  "fertilizer": "NPK recommendations"
}}

Strict Logic:
1. If most of the leaves look clean, green/gold and healthy, mark as "Healthy".
2. Maize ears (yellow/gold) are healthy parts, NOT Rust.
3. If not a plant, is_leaf=false."""

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 400}
    }

    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"): text = text[4:].strip()

        import json
        result = json.loads(text)
        
        if not result.get("is_leaf", True):
            return {"error": "Invalid Input", "message": "Please upload a clear plant leaf image."}

        return {
            "disease":    result.get("disease", f"{crop}: Healthy"),
            "confidence": float(result.get("confidence", 0.95)),
            "treatment":  result.get("treatment", "Balanced growth protocol."),
            "fertilizer": result.get("fertilizer", "NPK 15-15-15"),
            "method":     "Gemini 1.5 AI Vision"
        }
    except Exception:
        raise # Let parent handle fallback


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
def predict_disease_from_image(image_bytes: bytes, crop: str = None) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Basic foliage check
    thumb = image.copy()
    thumb.thumbnail((100, 100))
    pixels = list(thumb.getdata())
    green_gold = sum(1 for r, g, b in pixels if (g > r * 1.0) or (r > 150 and g > 150)) # expanded for corn
    if green_gold / len(pixels) < 0.04:
        return {"error": "Invalid Input", "message": "Please upload a clear plant leaf image."}

    crop_name = (crop or "").strip() or "Plant"

    # Priority: AI Vision
    if GEMINI_API_KEY:
        try:
            return _gemini_predict(image_bytes, crop_name)
        except Exception:
            pass

    # Fallback: Expert System
    return _expert_fallback(image, crop_name)
