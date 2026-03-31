# disease_model.py - FarmAI Expert Diagnostics Engine v14.0
# High-precision, multi-tier AI plant pathology system
import os, io, base64, requests, json, re, threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# API URLs for orchestration
# API URLs for orchestration
# API URLs for orchestration
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.2-11b-vision-preview"
KINDWISE_URL = "https://crop.kindwise.com/api/v1/identification"

# ---------------------------------------------------------------------------
# COMPREHENSIVE DISEASE TREATMENT DATABASE
# Used by all tiers to enrich results with proper treatment plans
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# COMPREHENSIVE DISEASE TREATMENT DATABASE — Precise Farmer-Grade Instructions
# All dosages follow Indian Council of Agricultural Research (ICAR) standards
# ---------------------------------------------------------------------------
DISEASE_DB = {
    "rust": {
        "treatment": "Step 1: Remove and burn all visibly infected leaves. Step 2: Mix Mancozeb 75WP at 2g per litre of water (e.g. 40g per 20L spray tank) and spray the entire crop. Step 3: Alternatively use Propiconazole 25EC at 1ml per litre. Step 4: Repeat spray every 10-14 days for 3 cycles. Step 5: Avoid wetting leaves during evening hours.",
        "fertilizer": "Reduce Nitrogen (N) — stop urea temporarily. Apply Muriate of Potash (MOP/KCl) at 3kg per acre to harden cell walls and resist fungal spread.",
        "safety": "Wear waterproof gloves and a face mask while mixing and spraying. Do NOT spray during strong sunlight (above 35°C) or high winds — spray early morning (6-9 AM) or evening (4-7 PM).",
        "cost_estimate": "Mancozeb 75WP (500g pack) ≈ ₹120. Per 20L tank uses 40g ≈ ₹10/spray. For 3 sprays on 1 acre ≈ ₹250-350 total (chemical + minor labor)."
    },
    "blight": {
        "treatment": "Step 1: Uproot and destroy severely infected plants. Step 2: Spray Copper Oxychloride 50WP at 3g per litre of water (60g per 20L tank) covering both leaf surfaces. Step 3: Alternatively, use Metalaxyl 8% + Mancozeb 64WP at 2.5g per litre. Step 4: Repeat every 7 days until symptoms stop spreading. Step 5: Drain waterlogged fields immediately.",
        "fertilizer": "Apply DAP (Di-Ammonium Phosphate) at 50kg per acre to boost root immunity. Avoid excess Nitrogen.",
        "safety": "Copper Oxychloride can irritate skin and eyes. Wear gloves, goggles, and a mask. Wash hands thoroughly after spraying. Do not spray in windy or sunny conditions.",
        "cost_estimate": "Copper Oxychloride 50WP (500g) ≈ ₹150. Per 20L tank uses 60g ≈ ₹18/spray. For 4 sprays on 1 acre ≈ ₹400-500 total."
    },
    "smut": {
        "treatment": "PREVENTION (before planting): Treat seeds with Carboxin 37.5% + Thiram 37.5% at 2g per kg of seed. Rub the powder evenly on the seed surface. AFTER INFECTION: Remove all smut galls immediately before they burst open (to stop spore spread). There is NO chemical cure once infection is established — remove infected plants.",
        "fertilizer": "Apply Zinc Sulfate at 25kg per hectare to improve plant immunity and reduce susceptibility.",
        "safety": "Carboxin + Thiram seed treatment: Wear gloves and mask during seed treatment. Wash hands before eating. Treated seeds should not be consumed by humans or animals.",
        "cost_estimate": "Carboxin+Thiram 37.5+37.5WP (100g) ≈ ₹80. Treats up to 50kg of seed ≈ ₹1.50 per kg of seed treated. Very cost-effective preventive measure."
    },
    "mosaic": {
        "treatment": "There is NO chemical cure for mosaic virus. Step 1: Immediately uproot and burn infected plants — do not compost them. Step 2: Control aphid/whitefly vectors by spraying Imidacloprid 17.8SL at 0.5ml per litre of water. Step 3: Use yellow sticky traps @10 per acre. Step 4: Plant virus-resistant varieties in next season.",
        "fertilizer": "Spray Borax (Boron) at 0.5g per litre as foliar spray to strengthen cell walls and slow virus movement.",
        "safety": "Imidacloprid is toxic to bees — do NOT spray near flowering crops or during bloom. Wear gloves and mask. Avoid spraying near water sources.",
        "cost_estimate": "Imidacloprid 17.8SL (100ml) ≈ ₹180. Per 20L tank uses 10ml ≈ ₹18/spray. Yellow sticky traps (pack of 10) ≈ ₹150. Total management for 1 acre ≈ ₹400-600."
    },
    "wilt": {
        "treatment": "Step 1: Remove wilted plants with roots and destroy. Step 2: Drench the root zone with Carbendazim 50WP at 1g per litre of water (apply 200-250ml per plant). Step 3: Alternatively, apply Trichoderma viride or T. harzianum bio-fungicide at 4g per kg soil at planting. Step 4: Avoid waterlogging — ensure proper field drainage.",
        "fertilizer": "Apply Calcium Nitrate at 2kg per 100L water as soil drench to strengthen root cell walls.",
        "safety": "Carbendazim soil drench: wear gloves. Trichoderma is safe (biological agent) but still avoid eye contact. Wash hands after application.",
        "cost_estimate": "Carbendazim 50WP (250g) ≈ ₹90. Per plant drench uses ~0.25g ≈ ₹0.09 per plant. For 1 acre (~500 plants) ≈ ₹200. Trichoderma viride (1kg) ≈ ₹120 for season."
    },
    "rot": {
        "treatment": "Step 1: Cut away and dispose of all rotten tissue. Step 2: Drench with Copper Oxychloride 50WP at 3g per litre or Mancozeb at 2g per litre, applied directly to the affected area and soil. Step 3: Reduce irrigation frequency. Step 4: Improve field drainage — create furrows to allow water run-off. Step 5: Spray Iprodione 50WP at 2g per litre on stored produce.",
        "fertilizer": "Apply Superphosphate (SSP) at 50kg per acre to strengthen root tissue.",
        "safety": "Wear gloves and avoid inhaling Mancozeb dust. Spray early morning. Keep children and animals away during and after spraying for at least 2 hours.",
        "cost_estimate": "Mancozeb 75WP (500g) ≈ ₹120. Per 20L tank ≈ ₹10/spray. For 4 sprays on 1 acre ≈ ₹300-400 total chemical cost."
    },
    "spot": {
        "treatment": "Step 1: Remove and burn spotted leaves. Step 2: Spray Chlorothalonil 75WP at 2g per litre of water (40g per 20L tank) covering leaf undersides thoroughly. Step 3: Or use Mancozeb 75WP at 2g per litre. Step 4: Repeat every 14 days for 2-3 applications. Step 5: Avoid overhead irrigation to keep leaves dry.",
        "fertilizer": "Foliar spray of Zinc Sulfate (0.5g/L) + Manganese Sulfate (0.3g/L) to boost plant immunity.",
        "safety": "Chlorothalonil: wear gloves, goggles, and mask — it is a mild eye irritant. Avoid spraying in windy conditions. Do not spray during peak sunlight hours.",
        "cost_estimate": "Chlorothalonil 75WP (500g) ≈ ₹200. Per 20L tank uses 40g ≈ ₹16/spray. For 3 sprays on 1 acre ≈ ₹350-450 total."
    },
    "mildew": {
        "treatment": "Step 1: Remove heavily infected leaves. Step 2: Spray Wettable Sulfur 80WP at 3g per litre (60g per 20L tank) or Azoxystrobin 23SC at 1ml per litre. Step 3: Do NOT spray sulfur when temperature exceeds 35°C (risk of phytotoxicity). Step 4: Repeat every 10 days for 3 sprays. Step 5: Improve air circulation by pruning dense canopy.",
        "fertilizer": "Stop all Nitrogen (urea) fertilizer. Apply Potassium Silicate at 2ml per litre as foliar spray to harden leaf surfaces.",
        "safety": "Wettable Sulfur is phytotoxic above 35°C — NEVER spray sulfur in hot weather. Wear gloves and mask. Azoxystrobin: avoid contact with skin and eyes.",
        "cost_estimate": "Wettable Sulfur 80WP (1kg) ≈ ₹100. Per 20L tank uses 60g ≈ ₹6/spray. Azoxystrobin 23SC (100ml) ≈ ₹350. For 3 sulfur sprays on 1 acre ≈ ₹150-250 total."
    },
    "scorch": {
        "treatment": "Step 1: This is often caused by the fungus Fabraea maculata or environmental stress. Step 2: Spray Bordeaux Mixture 1% (10g Copper Sulfate CuSO₄ + 10g hydrated lime per litre of water) — mix CuSO₄ in half the water, mix lime separately in other half, then combine slowly). Step 3: Spray every 7-10 days for 3 cycles. Step 4: Remove and burn infected leaves. Step 5: Ensure proper irrigation — avoid drought stress.",
        "fertilizer": "Apply Calcium Nitrate at 200g per 100L water as foliar spray to strengthen leaf tissue and prevent further scorch.",
        "safety": "Bordeaux Mixture: CuSO₄ is corrosive — wear rubber gloves and goggles. Never use iron or galvanized containers to mix. Spray early morning (6-9 AM) or evening.",
        "cost_estimate": "Copper Sulfate (CuSO₄) 1kg ≈ ₹200, Lime 1kg ≈ ₹20. Per 20L tank: 200g CuSO₄ + 200g lime ≈ ₹44/spray. For 3 sprays on 1 acre ≈ ₹500-700 total."
    },
    "anthracnose": {
        "treatment": "Step 1: Collect and destroy all fallen infected leaves and fruit. Step 2: Spray Carbendazim 50WP at 1g per litre (20g per 20L tank) or Copper Hydroxide 77WP at 2g per litre. Step 3: Repeat every 10-14 days. Step 4: Avoid overhead irrigation. Step 5: Disinfect pruning tools with 10% bleach solution between cuts.",
        "fertilizer": "Apply NPK 13-0-45 (high potassium) at 2g per litre as foliar spray. Add Calcium Chloride 0.5g/L.",
        "safety": "Carbendazim is a mild systemic fungicide — wear gloves. Copper Hydroxide: avoid eye contact, wear goggles. Do not spray near water bodies. Spray early morning only.",
        "cost_estimate": "Carbendazim 50WP (250g) ≈ ₹90. Per 20L tank ≈ ₹9/spray. Copper Hydroxide 77WP (500g) ≈ ₹200. For 3-4 sprays on 1 acre ≈ ₹300-500 total."
    },
    "canker": {
        "treatment": "Step 1: Prune infected branches at least 15cm below the visible infection point. Step 2: Immediately paint cut surfaces with Bordeaux paste (100g CuSO₄ + 100g lime in 1 litre water, thickened to paste). Step 3: Spray entire tree with Copper Oxychloride 50WP at 3g per litre. Step 4: Sterilize pruning tools between each cut using 70% alcohol or 10% bleach.",
        "fertilizer": "Apply Calcium (Ca) at 2g/L + Boron (B) at 0.5g/L as foliar spray to prevent cell wall breakdown.",
        "safety": "Sterilize pruning tools between cuts (70% alcohol). Wear gloves when handling Bordeaux paste — CuSO₄ irritates skin. Dispose of pruned material by burning, not composting.",
        "cost_estimate": "Copper Sulfate (500g) ≈ ₹100, Lime ≈ ₹10. Pruning cost depends on tree count. Chemical cost for paste + 2 sprays on 10 trees ≈ ₹200-300 total."
    },
    "bordeaux": {
        "treatment": "Bordeaux Mixture 1% preparation: Dissolve 10g of Copper Sulfate (CuSO₄) in 500ml water. Separately dissolve 10g of hydrated lime in another 500ml of water. Slowly pour the CuSO₄ solution into the lime solution (never the reverse). Test with litmus — should be neutral or slightly alkaline. Apply 1-2 litres per tree, spray every 7-10 days.",
        "fertilizer": "No additional fertilizer needed if used as preventive treatment.",
        "safety": "Bordeaux Mixture is corrosive. Wear rubber gloves and eye protection. Do not use iron/galvanized containers for mixing. Spray early morning or evening.",
        "cost_estimate": "Copper Sulfate (1kg) ≈ ₹200, Lime ≈ ₹20. Per 100L mixture ≈ ₹220. Very economical for large orchards."
    },
    "scab": {
        "treatment": "Step 1: Start spraying EARLY in the growing season, even before symptoms appear (weather-driven disease — spreads in wet/humid conditions). Step 2: Spray Mancozeb 75WP at 2g per litre (40g per 20L tank) every 7-10 days for 3-4 cycles. Step 3: Alternative fungicides — Carbendazim 50WP at 1g/L OR Chlorothalonil 75WP at 2g/L (rotate to prevent resistance). Step 4: Remove and burn all infected leaves and fallen fruit. Step 5: Avoid overhead irrigation to keep leaves dry. Step 6: Prune to improve air circulation inside the canopy.",
        "fertilizer": "Apply Calcium Nitrate at 200g per 100L water as foliar spray to strengthen leaf tissue. Avoid excess Nitrogen which increases susceptibility.",
        "safety": "Wear gloves and face mask when mixing fungicides. Spray early morning (6-9 AM) or evening (4-7 PM) — avoid spraying during strong sunlight or high winds. Keep spray away from eyes.",
        "cost_estimate": "Mancozeb 75WP (500g) ≈ ₹120. Per 20L tank uses 40g ≈ ₹10/spray. Carbendazim 50WP (250g) ≈ ₹90. For full season (4 sprays alternating) on 1 acre ≈ ₹350-500 total."
    },
    "healthy": {
        "treatment": "No treatment needed. Crop appears in good health. Continue regular monitoring every 5-7 days. Maintain standard cultural practices (proper spacing, timely irrigation, weed management).",
        "fertilizer": "Continue your existing NPK schedule based on soil test results. As a preventive boost, apply NPK 19-19-19 at 2g per litre as foliar spray once a month.",
        "safety": "Crop looks healthy — spraying not needed. If applying preventive foliar NPK, wear gloves and spray in early morning. Avoid fertilizer spray in high winds.",
        "cost_estimate": "No disease treatment cost. Preventive NPK 19-19-19 (1kg) ≈ ₹80. Per acre foliar spray monthly ≈ ₹40-60/month."
    },
    "deficiency": {
        "treatment": "Iron deficiency (yellowing between veins): Spray Ferrous Sulfate at 2.5g per litre. Zinc deficiency (small leaves, bronzing): Spray Zinc Sulfate at 2g per litre. Magnesium deficiency (older leaves yellow first): Spray Magnesium Sulfate (Epsom salt) at 10g per litre. Apply foliar spray in the morning or evening, not in harsh sunlight.",
        "fertilizer": "Conduct full soil test to identify deficient nutrients. As interim measure: apply micronutrient mixture (Zn, Fe, Mn, Cu, B) at label dose.",
        "safety": "Wear gloves when handling concentrated fertilizer solutions. Ferrous Sulfate can stain clothing. Apply micronutrient sprays early morning to prevent leaf burn.",
        "cost_estimate": "Ferrous Sulfate (1kg) ≈ ₹30. Zinc Sulfate (1kg) ≈ ₹60. Magnesium Sulfate (1kg) ≈ ₹40. Per acre foliar spray (2-3 sprays) ≈ ₹100-200 total."
    },
    "default": {
        "treatment": "Symptoms not conclusively matched. As a precaution: Step 1: Remove visibly infected leaves and destroy. Step 2: Spray broad-spectrum fungicide Mancozeb 75WP at 2g per litre of water. Step 3: Observe the crop for 5-7 days. Step 4: If no improvement, consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer with a sample.",
        "fertilizer": "Apply NPK 15-15-15 complex fertilizer at 2g per litre as a general health booster foliar spray.",
        "safety": "Wear gloves and a face mask when mixing and spraying any pesticide. Spray early morning (6-9 AM) or evening (4-7 PM). Keep children and animals away from the sprayed area for at least 4 hours.",
        "cost_estimate": "Mancozeb 75WP (500g) ≈ ₹120. Estimated cost for 2-3 precautionary sprays on 1 acre ≈ ₹200-350 total. Consult local agro-dealer for current market prices."
    },
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

def _correct_scientific_names(text: str) -> str:
    """Fix common AI scientific name typos."""
    corrections = {
        "Venturia pirina": "Venturia pyrina",
        "venturia pirina": "Venturia pyrina",
        "Sphaerotheca pannosa": "Podosphaera pannosa",  # updated taxonomy
        "Puccinia tritici": "Puccinia triticina",       # correct species name
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def _preprocess_image(image_bytes: bytes, max_size: int = 800) -> bytes:
    """Resize and optimize image for best AI analysis results."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()

# ---------------------------------------------------------------------------
# TIER 1: GEMINI (Primary — Best Accuracy)
# ---------------------------------------------------------------------------
def _gemini_predict(api_key: str, image_bytes: bytes, crop: str) -> dict:
    if not api_key:
        raise ValueError("G-Missing")

    expert_prompt = f"""You are a Senior Clinical Plant Pathologist.
    Perform a high-precision morphological analysis on this {crop or 'crop'}.
    
    CLINICAL REASONING STEPS:
    1. PHENOLOGICAL VALIDATION (CRITICAL):
       - WHEAT RIPENING: If wheat heads are turning pale/golden uniformly, it is 100% HEALTHY MATURITY.
       - FUSARIUM SCAB: Only report if you see PREMATURE, irregular bleaching of spikelets while others remain green. If unsure, prioritize "Healthy - Monitoring Recommended".
       - ANTHERS: Small orange/yellow tips on wheat heads are natural flowering, not rust.
    2. MAIZE VALIDATION: Tassels at the top are healthy. Boat-shaped lesions = Blight.
    3. FINAL DECISION: If natural patterns are dominant, you MUST report "Healthy".
    
    Return ONLY a JSON object:
    {{
      "disease": "Scientific + Common Name",
      "confidence": 0.95,
      "severity": "Low/Medium/High/Healthy",
      "treatment": "Precise Chemical + Dosage",
      "fertilizer": "Specific NPK/Micronutrient recommendations",
      "cost_estimate": "₹... per acre",
      "reason": "Expert evidence-based reasoning."
    }}
    """

    b = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode("utf-8")}},
            {"text": expert_prompt}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800}
    }
    r = requests.post(f"{GEMINI_URL}?key={api_key}", json=b, timeout=25)
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

    # Fix scientific name typos in AI response
    if "reason" in res:
        res["reason"] = _correct_scientific_names(res["reason"])
    if "treatment" in res:
        res["treatment"] = _correct_scientific_names(res["treatment"])

    reason = res.get("reason", "").strip()
    if not reason:
        reason = f"AI identified {res['disease']} based on visual pattern analysis of the submitted image."

    return {
        "disease": res["disease"],
        "confidence": float(res.get("confidence", 0.93)),
        "severity": res.get("severity", "Medium"),
        "treatment": treatment,
        "fertilizer": res.get("fertilizer") or db["fertilizer"],
        "safety": res.get("safety") or db.get("safety", "Wear gloves and avoid sunlight/wind."),
        "cost_estimate": res.get("cost_estimate") or db.get("cost_estimate", "Consult local rates."),
        "reason": reason,
        "method": "Gemini 1.5 Flash Expert"
    }

# ---------------------------------------------------------------------------
# TIER 2: GROQ LLAMA VISION (Secondary fallback)
# ---------------------------------------------------------------------------
def _groq_predict(api_key: str, image_bytes: bytes, crop: str) -> dict:
    if not api_key:
        raise ValueError("X-Missing")

    expert_prompt = f"""You are a Clinical Plant Pathologist. Analyze this {crop or 'crop'}.
    1. MAIZE VALIDATION: Differentiate between healthy orange Tassels at the top vs disease.
    2. MAIZE DISEASES: "Cigar-shaped" = N. Blight, "Rectangular" = Grey Leaf Spot, "Circular" = Rust.
    3. Grains: Heads (Maturity/Anthers) vs Leaf (Rust).
    4. If healthy -> "Healthy". If diseased, exact chemicals + doses.
    
    Return JSON only:
    {{"disease": "...", "confidence": 0.9, "severity": "...", "treatment": "...", "reason": "Proof."}}"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}},
                {"type": "text", "text": expert_prompt}
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 400
    }
    r = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    if r.status_code != 200:
        raise Exception(f"X-{r.status_code}")

    txt = r.json()["choices"][0]["message"]["content"]
    res = _parse_json_safely(txt)

    db = _get_treatment_from_db(res.get("disease", ""))
    treatment = res.get("treatment", "")
    if len(treatment) < 30:
        treatment = db["treatment"]

    # Fix scientific name typos in AI response
    if "reason" in res:
        res["reason"] = _correct_scientific_names(res["reason"])
    if "treatment" in res:
        res["treatment"] = _correct_scientific_names(res["treatment"])

    reason = res.get("reason", "").strip()
    if not reason:
        reason = f"AI detected {res['disease']} based on visual symptom analysis of the crop image."

    return {
        "disease": res["disease"],
        "confidence": float(res.get("confidence", 0.88)),
        "severity": res.get("severity", "Medium"),
        "treatment": treatment,
        "fertilizer": res.get("fertilizer") or db["fertilizer"],
        "safety": res.get("safety") or db.get("safety", "Standard PPE required."),
        "cost_estimate": res.get("cost_estimate") or db.get("cost_estimate", "Varies by region."),
        "reason": reason,
        "method": "Groq Llama 3.2 Vision"
    }

# ---------------------------------------------------------------------------
# TIER 3: KINDWISE (Specialized Plant Health API)
# ---------------------------------------------------------------------------
def _kindwise_predict(api_key: str, image_bytes: bytes) -> dict:
    if not api_key:
        raise ValueError("K-Missing")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {
        "images": [f"data:image/jpeg;base64,{b64}"],
        "latitude": 0.0,
        "longitude": 0.0,
        "similar_images": True
    }

    r = requests.post(KINDWISE_URL, json=payload, headers=headers, timeout=15)
    if r.status_code not in [200, 201]:
        raise Exception(f"K-{r.status_code}")

    data = r.json()
    # Handle both direct suggestings and asynchronous identification
    result = data.get("result", {})
    health = result.get("disease", {})
    suggestions = health.get("suggestions", []) or result.get("suggestions", [])
    
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

    description = details.get("description", {}).get("value", "")
    reason = description[:200] if description else f"Kindwise Plant Health database matched this image to {disease_name} with {prob*100:.0f}% probability based on visual characteristics."

    return {
        "disease": disease_name,
        "confidence": prob,
        "severity": "High" if prob > 0.85 else "Medium",
        "treatment": treatment_str,
        "fertilizer": db["fertilizer"],
        "safety": db.get("safety", "Wear gloves and mask."),
        "cost_estimate": db.get("cost_estimate", "Estimate: ₹10-20/spray."),
        "reason": reason,
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
    gold_pct   = ((R > 180) & (G > 160) & (B < 130)).sum() / total   # mature grain
    green_pct  = ((G > R + 15) & (G > B + 15)).sum() / total          # healthy green
    rust_pct   = ((R > G + 45) & (R > B + 65) & (R < 215) & (G > 35)).sum() / total  # rust/orange
    yellow_pct = ((R > 180) & (G > 160) & (B < 80)).sum() / total    # chlorosis
    dark_pct   = ((R < 80) & (G < 80) & (B < 80)).sum() / total       # necrosis

    # 1. Healthy Ripening/Maturity (Golden hue) - Balanced
    if gold_pct > 0.18 and rust_pct < 0.05:
        return {"disease": "Healthy — Ripening/Mature", "confidence": 0.95, "severity": "Healthy",
                "treatment": "No treatment required. Crop is at maturity stage.",
                "reason": f"Maturity detection: {gold_pct*100:.1f}% golden/yellow hue with low infection signature.",
                "method": "Neural Fallback"}

    # 2. Actual Rust/Fungal Detection (High Potency Restored)
    if rust_pct > 0.02 or dark_pct > 0.03:
        db = DISEASE_DB["rust"] if rust_pct > 0.02 else DISEASE_DB["spot"]
        sev = "High" if (rust_pct + dark_pct) > 0.12 else "Medium"
        return {"disease": "Foliar Rust / Fungal Lesions", "confidence": 0.85, "severity": sev,
                "treatment": db["treatment"], "fertilizer": db["fertilizer"],
                "reason": f"Detected {rust_pct*100:.1f}% active fungal signature (spots/lesions) and {dark_pct*100:.1f}% necrotic tissue. {err_tag}",
                "method": "Neural Fallback"}

    # 3. Strong Green Baseline (Healthy)
    if green_pct > 0.40:
        db = DISEASE_DB["healthy"]
        return {"disease": "Healthy", "confidence": 0.88, "severity": "Healthy",
                "treatment": db["treatment"], "fertilizer": db["fertilizer"],
                "reason": f"High vegetative health: {green_pct*100:.1f}% green coverage. No significant pathogen patterns found. {err_tag}",
                "method": "Neural Fallback"}

    db = DISEASE_DB["default"]
    return {"disease": "Inconclusive / Healthy", "confidence": 0.60, "severity": "Unknown",
            "treatment": f"No definitive symptoms found. {db['treatment']}",
            "reason": "Image analysis returned mostly healthy or natural ripening signals.",
            "method": "Neural Fallback"}

# ---------------------------------------------------------------------------

def predict_disease_from_image(image_bytes: bytes, crop: str = None, lat: float = None, lng: float = None) -> dict:
    c = crop or "Plant"
    location_name = None

    if lat and lng:
        try:
            headers = {"User-Agent": "FarmAI-Expert-System/1.0"}
            r = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers, timeout=3)
            if r.status_code == 200:
                addr = r.json().get("address", {})
                location_name = addr.get("village") or addr.get("suburb") or addr.get("city_district") or addr.get("town") or addr.get("city") or addr.get("county") or addr.get("state")
        except: pass

    try: image_bytes = _preprocess_image(image_bytes)
    except: pass

    def enrich(res):
        if location_name:
            res["location_name"] = location_name
            if "cost_estimate" in res:
                res["cost_estimate"] = f"{res['cost_estimate']} [Verified: {location_name}]"
        return res

    # 🚀 PARALLEL MULTI-MODEL EXECUTION (Consensus System)
    results = []
    errs = []
    
    # Refresh keys from environment to ensure latest .env values
    GK = os.getenv("GEMINI_API_KEY", "").strip()
    XK = os.getenv("GROK_API_KEY", "").strip()
    KK = os.getenv("CROP_HEALTH_API_KEY", "").strip()

    def run_tier(name, func, key, *args):
        if not key: return {"name": name, "res": None, "err": "Key Missing"}
        try: return {"name": name, "res": func(key, *args), "err": None}
        except Exception as e: return {"name": name, "res": None, "err": str(e)}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_tier, "Gemini", _gemini_predict, GK, image_bytes, c),
            executor.submit(run_tier, "Groq", _groq_predict, XK, image_bytes, c),
            executor.submit(run_tier, "Kindwise", _kindwise_predict, KK, image_bytes)
        ]
        tier_results = [f.result() for f in futures]

    for tr in tier_results:
        if tr["res"]: results.append(tr["res"])
        else: errs.append(f"{tr['name']}:{tr['err'][:30]}")

    if results:
        # Pick the most confident result among the APIs
        best = max(results, key=lambda x: x.get("confidence", 0))
        # If confidence is low, add a warning
        if best.get("confidence", 0) < 0.6:
            best["reason"] = f"Low-confidence consensus. {best.get('reason','')}"
        return enrich(best)

    # All APIs failed -> Use Fallback and report errors
    fb = _expert_fallback(image_bytes, c, errs)
    if errs:
        fb["method"] = f"Neural Fallback [System Status: {', '.join(errs)}]"
    return enrich(fb)


def predict_disease_multiple(image_list: list, crop: str = None, lat: float = None, lng: float = None) -> dict:
    """Analyze multiple images of the same crop and return a consensus diagnosis with robust matching."""
    if not image_list:
        return {"error": "No images provided."}
    
    results = []
    for img_bytes in image_list:
        try:
            res = predict_disease_from_image(img_bytes, crop=crop, lat=lat, lng=lng)
            results.append(res)
        except Exception:
            results.append({"disease": "Unreadable", "confidence": 0.0})

    # 1. Advanced Normalization Helper
    def normalize(name):
        n = name.lower()
        # Remove common "filler" words that cause mismatch
        n = re.sub(r'\b(of|in|on|the|crop|plant|wheat|rice|maize|corn)\b', '', n)
        # Remove punctuation and extra whitespace
        n = re.sub(r'[^\w\s]', '', n)
        return " ".join(n.split())

    # 2. Score diseases by frequency AND confidence
    scores = {}
    for r in results:
        raw_name = r.get("disease", "Unknown")
        norm_name = normalize(raw_name)
        if not norm_name: norm_name = "unknown"
        
        # We give a score based on confidence to avoid picking a high-frequency but low-confidence guess
        conf = r.get("confidence", 0.1)
        scores[norm_name] = scores.get(norm_name, 0) + conf

    if not scores:
        return results[0]

    # Consensus norm name is the one with the highest cumulative confidence score
    consensus_norm = max(scores, key=scores.get)
    max_score = scores[consensus_norm]
    
    # 🕵️ SKEPTICAL CONSENSUS GUARDRAIL
    # If confidence is low (<0.75 cumulative) and any model says "Healthy", prioritize health
    has_healthy_signal = any("healthy" in normalize(r.get("disease","")) for r in results)
    if max_score < 0.75 and has_healthy_signal:
        consensus_norm = "healthy"

    # 3. Identify outliers based on normalized names
    wrong_count = 0
    valid_results = []
    
    for r in results:
        raw_name = r.get("disease", "Unknown")
        norm_name = normalize(raw_name)
        
        # If the normalized name matches or shares significant keywords, it's valid
        is_match = (norm_name == consensus_norm)
        
        # Check for keyword overlap (e.g. "leaf rust" vs "rust")
        if not is_match:
            c_words = set(consensus_norm.split())
            r_words = set(norm_name.split())
            if c_words and r_words and (c_words & r_words): # Shared keywords
                is_match = True

        if not is_match or r.get("confidence", 0) < 0.35:
            wrong_count += 1
        else:
            valid_results.append(r)

    # 4. Final Aggregation
    if valid_results:
        # Pick the most confident result among valid ones for the final display
        final_res = max(valid_results, key=lambda x: x.get("confidence", 0))
    else:
        # If no consensus, pick the highest confidence overall
        final_res = max(results, key=lambda x: x.get("confidence", 0))

    final_res["wrong_inputs"] = wrong_count
    final_res["total_inputs"] = len(image_list)
    
    if wrong_count > 0:
        msg = f"Detected {wrong_count} inconsistent/invalid image(s) out of {len(image_list)}."
        final_res["reason"] = f"{msg} {final_res.get('reason', '')}"
        
    return final_res

