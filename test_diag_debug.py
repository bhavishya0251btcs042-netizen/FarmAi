# test_diag_debug.py - DEBUG CONSENSUS
import os, base64, json, requests
from disease_model import _gemini_predict, _groq_predict, _nvidia_predict, _kindwise_predict, _expert_fallback, _preprocess_image
from dotenv import load_dotenv

load_dotenv()

IMG_PATH = r"C:\Users\DELL\.gemini\antigravity\brain\0d8b226c-78ba-49c2-a150-4a00eec9f0e2\media__1775054718004.jpg"
GK = os.getenv("GEMINI_API_KEY", "").split(",")[0].strip()
XK = os.getenv("GROK_API_KEY", "").split(",")[0].strip()
NK = os.getenv("NVIDIA_API_KEY", "").split(",")[0].strip()

with open(IMG_PATH, "rb") as f:
    img_bytes = _preprocess_image(f.read())

print("--- DEBUG CLUSTER ANALYSIS ---")

try:
    print("\n[GEMINI TIER]")
    print(json.dumps(_gemini_predict(GK, img_bytes, "Pear"), indent=2))
except Exception as e: print(f"Gemini Error: {e}")

try:
    print("\n[GROQ TIER]")
    print(json.dumps(_groq_predict(XK, img_bytes, "Pear"), indent=2))
except Exception as e: print(f"Groq Error: {e}")

try:
    print("\n[NVIDIA TIER]")
    print(json.dumps(_nvidia_predict(NK, img_bytes, "Pear"), indent=2))
except Exception as e: print(f"Nvidia Error: {e}")

try:
    print("\n[FALLBACK TIER]")
    print(json.dumps(_expert_fallback(img_bytes, "Pear"), indent=2))
except Exception as e: print(f"Fallback Error: {e}")
