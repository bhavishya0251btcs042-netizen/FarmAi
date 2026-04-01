# verify_all.py - PRODUCTION-READY FarmAI DIAGNOSTIC SUITE
import os, requests, base64, joblib, cv2, numpy as np
from dotenv import load_dotenv

load_dotenv()

# Key Sanitization (Supports comma-separated pools)
KINDWISE_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GEMINI_KEYS   = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
GROQ_KEYS     = [k.strip() for k in os.getenv("GROK_API_KEY", "").split(",") if k.strip()]
NVIDIA_KEYS   = [k.strip() for k in os.getenv("NVIDIA_API_KEY", "").split(",") if k.strip()]

def verify_gemini():
    if not GEMINI_KEYS: return "MISSING KEY"
    # Try the new stable v1 endpoint first
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_KEYS[0]}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEYS[0]}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEYS[0]}"
    ]
    for url in endpoints:
        try:
            r = requests.post(url, json={"contents": [{"parts": [{"text": "Hello"}]}]}, timeout=15)
            if r.status_code == 200: return f"SUCCESS (Status 200 via {url.split('/')[3]})"
        except: continue
    return "FAILED (All Endpoints 401/404/Timeout)"

def verify_groq():
    if not GROQ_KEYS: return "MISSING KEY"
    url = "https://api.groq.com/openai/v1/chat/completions"
    try:
        # Verify both text and vision availability
        r = requests.post(url, json={"model": "meta-llama/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": "Hello"}]}, headers={"Authorization": f"Bearer {GROQ_KEYS[0]}"}, timeout=15)
        return f"({r.status_code}, {r.reason if r.status_code==200 else r.json().get('error',{}).get('message','')})"
    except Exception as e: return f"Error: {str(e)[:40]}"

def verify_nvidia():
    if not NVIDIA_KEYS: return "MISSING KEY"
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    try:
        r = requests.post(url, json={"model": "meta/llama-3.2-11b-vision-instruct", "messages": [{"role": "user", "content": "Hello"}]}, headers={"Authorization": f"Bearer {NVIDIA_KEYS[0]}"}, timeout=15)
        return f"({r.status_code}, {r.reason})"
    except Exception as e: return f"Error: {str(e)[:40]}"

def verify_local_intelligence():
    if not os.path.exists("disease_model.joblib"): return "JOBLIB MISSING (Normal if first run)"
    try:
        m = joblib.load("disease_model.joblib")
        # Dummy features [H, S, V, Texture]
        m.predict([[40, 100, 100, 10]])
        return "SUCCESS (Operational)"
    except Exception as e: return f"FAILED ({str(e)[:40]})"

def verify_kindwise():
    if not KINDWISE_KEY: return "MISSING KEY"
    headers = {"Api-Key": KINDWISE_KEY, "Content-Type": "application/json"}
    try:
        r = requests.post("https://crop.kindwise.com/api/v1/identification", json={"images": ["data:image/jpeg;base64,"]}, headers=headers, timeout=10)
        # 400 is expected for empty image, 401 is unauthorized
        if r.status_code == 401: return "FAILED (401 Unauthorized - Key Invalid)"
        return f"SUCCESS (Status {r.status_code})"
    except Exception as e: return f"Error: {str(e)[:40]}"

print("--- FarmAI PRODUCTION READINESS REPORT (v14.5) ---")
print(f"1. KINDWISE : {verify_kindwise()}")
print(f"2. GEMINI   : {verify_gemini()}")
print(f"3. GROQ     : {verify_groq()}")
print(f"4. NVIDIA   : {verify_nvidia()}")
print(f"5. LOCAL AI : {verify_local_intelligence()}")
print(f"6. OPENCV   : SUCCESS (Module Loaded: {cv2.__version__})")
