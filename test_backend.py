# test_backend.py - 2026 Pathogen Payload Test
import requests, json

url = "http://localhost:8000/predict-disease"
# Simple check for the 401/404 first
try:
    r = requests.post(url, timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Header Check (JSON Expected): {r.headers.get('Content-Type')}")
    print(f"Server Identity: {r.json().get('detail', 'NOT_AUTH')}")
except Exception as e:
    print(f"CRITICAL: Connection Refused. {e}")
