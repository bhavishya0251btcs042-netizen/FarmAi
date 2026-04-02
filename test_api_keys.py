import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
img = open('logo/logo.png', 'rb').read()

print('--- GEMINI ---')
key = os.getenv('GEMINI_API_KEY')
prompt = 'Plant Pathologist Scan. return JSON: disease, confidence, severity, treatment, reason.'
b = {'contents': [{'parts': [{'inline_data': {'mime_type': 'image/jpeg', 'data': base64.b64encode(img).decode("utf-8")}}, {'text': prompt}]}], 'generationConfig': {'temperature': 0.1}}
res = requests.post(f'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}', json=b)
print(res.status_code, res.text[:200])

print('--- GROQ ---')
key2 = os.getenv('GROK_API_KEY').split(',')[0].strip()
b64 = base64.b64encode(img).decode("utf-8")
payload = {'model': 'llama-3.2-90b-vision-preview', 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}, {'type': 'text', 'text': prompt}]}], 'temperature': 0.05}
res2 = requests.post('https://api.groq.com/openai/v1/chat/completions', json=payload, headers={'Authorization': f'Bearer {key2}'})
print(res2.status_code, res2.text[:200])
