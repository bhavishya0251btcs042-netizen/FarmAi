# test_diag.py - LIVE PATHOLOGY DEMO
import os, base64, json
from disease_model import predict_disease_from_image

IMG_PATH = r"C:\Users\DELL\.gemini\antigravity\brain\0d8b226c-78ba-49c2-a150-4a00eec9f0e2\media__1775054718004.jpg"

if not os.path.exists(IMG_PATH):
    print(f"Error: Image {IMG_PATH} not found.")
    exit(1)

with open(IMG_PATH, "rb") as f:
    img_bytes = f.read()

print("--- FARM-AI LIVE DIAGNOSTIC INITIATED ---")
print(f"Specimen: {os.path.basename(IMG_PATH)}")
print("Consolidating AI Clusters (Gemini, Groq, Nvidia)...")

res = predict_disease_from_image(img_bytes, crop="Pear")

print("\n--- FINAL MORPHOLOGICAL REPORT ---")
print(json.dumps(res, indent=4))
