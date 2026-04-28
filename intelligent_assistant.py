import os
import json
import base64
import requests
from io import BytesIO
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# TIER 1: Gemini
# ─────────────────────────────────────────────────────────────
def _explain_via_gemini(disease: str, confidence: float, stage: str, language: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").split(",")[0].strip()
    if not api_key:
        raise ValueError("No Gemini key")

    prompt = _build_prompt(disease, confidence, stage, language)

    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={api_key}",
    ]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
    }

    for url in endpoints:
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return _parse_explanation(raw)
        except Exception:
            continue

    raise Exception("Gemini all endpoints failed")


# ─────────────────────────────────────────────────────────────
# TIER 2: Groq (llama-3.3-70b-versatile — text only, very fast)
# ─────────────────────────────────────────────────────────────
def _explain_via_groq(disease: str, confidence: float, stage: str, language: str) -> dict:
    api_keys = [k.strip() for k in os.getenv("GROK_API_KEY", "").split(",") if k.strip()]
    if not api_keys:
        raise ValueError("No Groq key")

    prompt = _build_prompt(disease, confidence, stage, language)

    for key in api_keys:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert agricultural assistant. Always respond with valid JSON only — no markdown, no explanation, just the JSON object."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                },
                timeout=20
            )
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"]
                return _parse_explanation(raw)
        except Exception:
            continue

    raise Exception("Groq all keys failed")


# ─────────────────────────────────────────────────────────────
# TIER 3: Static fallback (always works, no API needed)
# ─────────────────────────────────────────────────────────────
def _explain_static_fallback(disease: str, confidence: float, stage: str) -> dict:
    """
    Rule-based fallback — works 100% offline, no API required.
    Returns English content (TTS will still work).
    """
    disease_title = disease.title() if disease else "Unknown Disease"
    conf_pct = f"{confidence * 100:.1f}%"

    stage_lower = (stage or "").lower()
    if "low" in stage_lower or "early" in stage_lower:
        urgency = "low"
        action_extra = "Monitor closely and apply preventive fungicide spray."
    elif "severe" in stage_lower or "high" in stage_lower:
        urgency = "severe"
        action_extra = "Act immediately. Remove infected plant parts and apply fungicide."
    else:
        urgency = "moderate"
        action_extra = "Apply recommended fungicide within 2-3 days."

    return {
        "title": disease_title,
        "summary": (
            f"AI detected {disease_title} with {conf_pct} confidence. "
            f"Infection severity is {stage or 'moderate'}. "
            f"Prompt action is recommended to protect your yield."
        ),
        "actions": [
            action_extra,
            "Remove and burn all visibly infected leaves or plant parts.",
            "Spray Mancozeb 75WP at 2g per litre of water every 7-10 days.",
            "Ensure proper drainage and avoid overhead irrigation.",
            "Consult your local agriculture extension officer for further advice.",
        ],
        "impact": (
            f"If untreated, this {urgency}-severity infection may cause "
            f"10-40% yield loss depending on crop stage and weather."
        ),
        "precautions": [
            "Wear rubber gloves and a face mask when spraying.",
            "Spray early morning (6-9 AM) or evening (4-7 PM) — avoid hot sunlight.",
            "Keep children and animals away from sprayed area for 4 hours.",
            "Store chemicals in a cool, dry, locked location.",
        ],
    }


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────
def _build_prompt(disease: str, confidence: float, stage: str, language: str) -> str:
    return f"""
You are a friendly crop doctor assistant helping farmers understand plant disease.

Disease Name: {disease}
AI Confidence: {confidence * 100:.1f}%
Infection Severity: {stage}
Response Language: {language}

Translate all text into the language code "{language}" (e.g. hi=Hindi, mr=Marathi, te=Telugu, ta=Tamil, en=English).
Keep sentences simple and practical. Avoid technical jargon.

Return ONLY a valid JSON object with this exact structure — no markdown, no code blocks:
{{
  "title": "<Simple friendly disease name in {language}>",
  "summary": "<2 short sentences: what the disease is and its effect on the crop>",
  "actions": [
    "<Step 1: Immediate action>",
    "<Step 2: Chemical/spray recommendation with dosage>",
    "<Step 3: Field management step>"
  ],
  "impact": "<1 sentence: expected yield or crop loss if untreated>",
  "precautions": [
    "<Safety precaution 1>",
    "<Safety precaution 2>"
  ]
}}
"""


def _parse_explanation(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────
# PUBLIC API — called from crop.py
# ─────────────────────────────────────────────────────────────
def generate_ai_explanation(disease: str, confidence: float, stage: str, language: str = "en") -> dict:
    """
    3-tier fallback:
      Tier 1 → Gemini
      Tier 2 → Groq (llama-3.3-70b)
      Tier 3 → Static rule-based (always works)
    """
    errors = []

    # Tier 1: Gemini
    try:
        result = _explain_via_gemini(disease, confidence, stage, language)
        print("[Voice Assistant] Tier 1 (Gemini) SUCCESS")
        return result
    except Exception as e:
        errors.append(f"Gemini: {e}")
        print(f"[Voice Assistant] Tier 1 (Gemini) FAILED — {e}")

    # Tier 2: Groq
    try:
        result = _explain_via_groq(disease, confidence, stage, language)
        print("[Voice Assistant] Tier 2 (Groq) SUCCESS")
        return result
    except Exception as e:
        errors.append(f"Groq: {e}")
        print(f"[Voice Assistant] Tier 2 (Groq) FAILED — {e}")

    # Tier 3: Static fallback
    print(f"[Voice Assistant] Tier 3 (Static Fallback) used. Errors: {errors}")
    return _explain_static_fallback(disease, confidence, stage)


def estimate_cost(disease: str, stage: str, area_in_acres: float) -> dict:
    stage_lower = (stage or "moderate").lower()
    disease_lower = (disease or "").lower()

    if "healthy" in disease_lower:
        range_min, range_max = 0, 0
    elif "low" in stage_lower or "early" in stage_lower:
        range_min, range_max = 300, 600
    elif "severe" in stage_lower or "high" in stage_lower:
        range_min, range_max = 1200, 3000
    else:
        range_min, range_max = 500, 1500

    if range_max == 0:
        return {"estimated_cost": "₹0 per acre", "total_cost": "₹0 total"}

    total_min = int(range_min * area_in_acres)
    total_max = int(range_max * area_in_acres)

    return {
        "estimated_cost": f"₹{range_min}–₹{range_max} per acre",
        "total_cost": f"₹{total_min}–₹{total_max} (for {area_in_acres} acres)"
    }


# ─────────────────────────────────────────────────────────────
# TTS — also fixed: all gTTS languages supported
# ─────────────────────────────────────────────────────────────

# Complete mapping of supported gTTS language codes
GTTS_SUPPORTED = {
    "hi", "en", "mr", "te", "ta", "bn", "gu", "kn", "ml", "pa",
    "ur", "es", "fr", "de", "zh-CN", "ar", "ru", "pt", "ja", "ko",
    "it", "nl", "tr", "vi", "pl", "id", "th",
}


def generate_voice_base64(text: str, language: str = "hi") -> str:
    """
    Convert text to speech using gTTS.
    Falls back to English if the requested language isn't supported.
    """
    # Normalize language code
    lang = language.strip().lower()

    # gTTS uses zh-TW / zh-CN — keep as-is if already correct
    if lang == "zh" or lang == "zh-cn":
        lang = "zh-CN"

    # Fallback to English if not supported
    if lang not in GTTS_SUPPORTED:
        print(f"[TTS] Language '{lang}' not supported by gTTS — falling back to English")
        lang = "en"

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception as e:
        print(f"[TTS] Error: {e}")
        # Last resort: try English
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return f"data:audio/mp3;base64,{base64.b64encode(fp.read()).decode('utf-8')}"
        except Exception as e2:
            print(f"[TTS] English fallback also failed: {e2}")
            return ""