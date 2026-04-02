import os
import json
import base64
import requests
from io import BytesIO
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

def generate_ai_explanation(disease: str, confidence: float, stage: str, language: str = "en") -> dict:
    # 1. Provide an exact prompt for the LLM
    prompt = f"""
You are a highly intelligent but very friendly crop doctor assistant. Explain the following crop diagnosis to a farmer.
Language: {language}
Disease Name: {disease}
AI Confidence: {confidence*100:.1f}%
Infection Severity: {stage}

Provide a structured, accurate translation directly into valid JSON.
Keep sentences short, practical, and highly actionable. NO technical jargon.
Return EXACTLY matching this JSON schema:
{{
  "title": "<Simplified friendly title of disease in the chosen language>",
  "summary": "<2-sentence simple summary of what it is and its effect.>",
  "actions": ["<Action step 1>", "<Action step 2 (e.g. fungicide/dosage if known)>"],
  "impact": "<1-sentence summary of crop/yield expected impact>",
  "precautions": ["<Safety/Prevention step 1>", "<Safety/Prevention step 2>"]
}}
"""
    # Using Gemini 2.5 flash as we know it's incredibly fast and accurate for this workspace
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("AI Key missing for Explanation Engine.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
    }
    
    response = requests.post(url, json=payload, timeout=20)
    if response.status_code != 200:
        raise Exception(f"AI Generation Failed: {response.text}")
        
    try:
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        return json.loads(raw_text.strip())
    except Exception as e:
        raise Exception(f"Parsing AI Response Failed: {str(e)}")

def estimate_cost(disease: str, stage: str, area_in_acres: float) -> dict:
    # Dynamically estimate cost based on stage/disease combination
    stage_lower = stage.lower() if stage else 'moderate'
    disease_lower = disease.lower()
    
    if "healthy" in disease_lower or "safe" in stage_lower:
        range_min, range_max = 0, 0
    elif "low" in stage_lower or "early" in stage_lower:
        range_min, range_max = 300, 600
    elif "severe" in stage_lower or "high" in stage_lower:
        range_min, range_max = 1200, 3000
    else:
        range_min, range_max = 500, 1500
        
    if range_max == 0:
        return {
            "estimated_cost": "₹0 per acre",
            "total_cost": "₹0 total"
        }
        
    total_min = int(range_min * area_in_acres)
    total_max = int(range_max * area_in_acres)
    
    return {
        "estimated_cost": f"₹{range_min}–₹{range_max} per acre",
        "total_cost": f"₹{total_min}–₹{total_max} (for {area_in_acres} acres)"
    }

def generate_voice_base64(text: str, language: str = "hi") -> str:
    try:
        # Generate TTS using the dynamic exact language requested
        tts = gTTS(text=text, lang=language, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # We package it into a raw data stream, bypassing any need for complex static file routing!
        audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception as e:
        print("TTS Engine Alert:", e)
        return ""
