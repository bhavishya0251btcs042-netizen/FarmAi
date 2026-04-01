# check_keys.py - PROFESSIONAL API KEY VERIFICATION
import os, requests, base64
from dotenv import load_dotenv

load_dotenv()

# Key Sanitization
KINDWISE_KEY = os.getenv("CROP_HEALTH_API_KEY", "").strip()
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_KEYS     = [k.strip() for k in os.getenv("GROK_API_KEY", "").split(",") if k.strip()]
NVIDIA_KEYS   = [k.strip() for k in os.getenv("NVIDIA_API_KEY", "").split(",") if k.strip()]

# Dummy Image (small red dot)
DUMMY_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def test_kindwise():
    url = "https://crop.kindwise.com/api/v1/identification?details=health"
    payload = {"images": [f"data:image/jpeg;base64,{DUMMY_B64}"]}
    r = requests.post(url, json=payload, headers={"Api-Key": KINDWISE_KEY}, timeout=15)
    return r.status_code, r.reason

def test_gemini():
    # Try V1 instead of V1BETA
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": "Is this plant healthy?"}]}]}
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code == 404: # Fallback
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
        r = requests.post(url, json=payload, timeout=15)
    return r.status_code, r.reason

def test_groq():
    if not GROQ_KEYS: return (0, "No keys found")
    results = []
    for ki, key in enumerate(GROQ_KEYS):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            # Try a standard text model first to see if the KEY is valid
            payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Hello"}]}
            r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=15)
            results.append(f"Key{ki+1}:({r.status_code}, {r.reason})")
        except Exception as e: results.append(f"Key{ki+1}:Error")
    return " | ".join(results)

def test_nvidia():
    if not NVIDIA_KEYS: return (0, "No keys found")
    results = []
    for ki, key in enumerate(NVIDIA_KEYS):
        try:
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            # Try standard text model to verify key
            payload = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "Hello"}]}
            r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=15)
            # If 404, the model name might be strictly scoped
            if r.status_code == 404:
                 payload["model"] = "nvidia/llama-3.1-8b-instruct"
                 r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=15)
            results.append(f"Key{ki+1}:({r.status_code}, {r.reason})")
        except Exception as e: results.append(f"Key{ki+1}:Error")
    return " | ".join(results)

print("--- FARM AI KEY REPORT ---")
print(f"1. KINDWISE : {test_kindwise()}")
print(f"2. GEMINI   : {test_gemini()}")
print(f"3. GROQ     : {test_groq()}")
print(f"4. NVIDIA   : {test_nvidia()}")
