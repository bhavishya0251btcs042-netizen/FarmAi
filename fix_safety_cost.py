"""
Fix script: Add safety note + cost estimate to disease_model.py
"""
with open('disease_model.py', encoding='utf-8') as f:
    content = f.read()

# ─────────────────────────────────────────────────────────────────
# 1. Update EVERY DISEASE_DB entry to include safety + cost_estimate
# ─────────────────────────────────────────────────────────────────
db_updates = {
    '"rust"': {
        'safety': 'Wear waterproof gloves and a face mask while mixing and spraying. Do NOT spray during strong sunlight (above 35°C) or high winds — spray early morning (6-9 AM) or evening (4-7 PM).',
        'cost_estimate': 'Mancozeb 75WP (500g pack) ≈ ₹120. Per 20L tank uses 40g ≈ ₹10/spray. For 3 sprays on 1 acre ≈ ₹250-350 total (chemical + minor labor).'
    },
    '"blight"': {
        'safety': 'Copper Oxychloride can irritate skin and eyes. Wear gloves, goggles, and a mask. Wash hands thoroughly after spraying. Do not spray in windy or sunny conditions.',
        'cost_estimate': 'Copper Oxychloride 50WP (500g) ≈ ₹150. Per 20L tank uses 60g ≈ ₹18/spray. For 4 sprays on 1 acre ≈ ₹400-500 total.'
    },
    '"smut"': {
        'safety': 'Carboxin + Thiram seed treatment: Wear gloves and mask during seed treatment. Wash hands before eating. Treated seeds should not be consumed by humans or animals.',
        'cost_estimate': 'Carboxin+Thiram 37.5+37.5WP (100g) ≈ ₹80. Treats up to 50kg of seed ≈ ₹1.50 per kg of seed treated. Very cost-effective preventive measure.'
    },
    '"mosaic"': {
        'safety': 'Imidacloprid is toxic to bees — do NOT spray near flowering crops or during bloom. Wear gloves and mask. Avoid spraying near water sources.',
        'cost_estimate': 'Imidacloprid 17.8SL (100ml) ≈ ₹180. Per 20L tank uses 10ml ≈ ₹18/spray. Yellow sticky traps (pack of 10) ≈ ₹150. Total management for 1 acre ≈ ₹400-600.'
    },
    '"wilt"': {
        'safety': 'Carbendazim soil drench: wear gloves. Trichoderma is safe (biological agent) but still avoid eye contact. Wash hands after application.',
        'cost_estimate': 'Carbendazim 50WP (250g) ≈ ₹90. Per plant drench uses ~0.25g ≈ ₹0.09 per plant. For 1 acre (~500 plants) ≈ ₹200. Trichoderma viride (1kg) ≈ ₹120 for season.'
    },
    '"rot"': {
        'safety': 'Wear gloves and avoid inhaling Mancozeb dust. Spray early morning. Keep children and animals away during and after spraying for at least 2 hours.',
        'cost_estimate': 'Mancozeb 75WP (500g) ≈ ₹120. Per 20L tank ≈ ₹10/spray. For 4 sprays on 1 acre ≈ ₹300-400 total chemical cost.'
    },
    '"spot"': {
        'safety': 'Chlorothalonil: wear gloves, goggles, and mask — it is a mild eye irritant. Avoid spraying in windy conditions. Do not spray during peak sunlight hours.',
        'cost_estimate': 'Chlorothalonil 75WP (500g) ≈ ₹200. Per 20L tank uses 40g ≈ ₹16/spray. For 3 sprays on 1 acre ≈ ₹350-450 total.'
    },
    '"mildew"': {
        'safety': 'Wettable Sulfur is phytotoxic above 35°C — NEVER spray sulfur in hot weather. Wear gloves and mask. Azoxystrobin: avoid contact with skin and eyes.',
        'cost_estimate': 'Wettable Sulfur 80WP (1kg) ≈ ₹100. Per 20L tank uses 60g ≈ ₹6/spray. Azoxystrobin 23SC (100ml) ≈ ₹350. For 3 sulfur sprays on 1 acre ≈ ₹150-250 total.'
    },
    '"scorch"': {
        'safety': 'Bordeaux Mixture: CuSO₄ is corrosive — wear rubber gloves and goggles. Never use iron or galvanized containers to mix. Spray early morning (6-9 AM) or evening.',
        'cost_estimate': 'Copper Sulfate (CuSO₄) 1kg ≈ ₹200, Lime 1kg ≈ ₹20. Per 20L tank: 200g CuSO₄ + 200g lime ≈ ₹44/spray. For 3 sprays on 1 acre ≈ ₹500-700 total.'
    },
    '"scab"': {
        'safety': 'Wear gloves and face mask when mixing fungicides. Spray early morning (6-9 AM) or evening (4-7 PM) — avoid spraying during strong sunlight or high winds. Keep spray away from eyes.',
        'cost_estimate': 'Mancozeb 75WP (500g) ≈ ₹120. Per 20L tank uses 40g ≈ ₹10/spray. Carbendazim 50WP (250g) ≈ ₹90. For full season (4 sprays alternating) on 1 acre ≈ ₹350-500 total.'
    },
    '"anthracnose"': {
        'safety': 'Carbendazim is a mild systemic fungicide — wear gloves. Copper Hydroxide: avoid eye contact, wear goggles. Do not spray near water bodies. Spray early morning only.',
        'cost_estimate': 'Carbendazim 50WP (250g) ≈ ₹90. Per 20L tank ≈ ₹9/spray. Copper Hydroxide 77WP (500g) ≈ ₹200. For 3-4 sprays on 1 acre ≈ ₹300-500 total.'
    },
    '"canker"': {
        'safety': 'Sterilize pruning tools between cuts (70% alcohol). Wear gloves when handling Bordeaux paste — CuSO₄ irritates skin. Dispose of pruned material by burning, not composting.',
        'cost_estimate': 'Copper Sulfate (500g) ≈ ₹100, Lime ≈ ₹10. Pruning cost depends on tree count. Chemical cost for paste + 2 sprays on 10 trees ≈ ₹200-300 total.'
    },
    '"healthy"': {
        'safety': 'Crop looks healthy — spraying not needed. If applying preventive foliar NPK, wear gloves and spray in early morning. Avoid fertilizer spray in high winds.',
        'cost_estimate': 'No disease treatment cost. Preventive NPK 19-19-19 (1kg) ≈ ₹80. Per acre foliar spray monthly ≈ ₹40-60/month.'
    },
    '"deficiency"': {
        'safety': 'Wear gloves when handling concentrated fertilizer solutions. Ferrous Sulfate can stain clothing. Apply micronutrient sprays early morning to prevent leaf burn.',
        'cost_estimate': 'Ferrous Sulfate (1kg) ≈ ₹30. Zinc Sulfate (1kg) ≈ ₹60. Magnesium Sulfate (1kg) ≈ ₹40. Per acre foliar spray (2-3 sprays) ≈ ₹100-200 total.'
    },
    '"default"': {
        'safety': 'Wear gloves and a face mask when mixing and spraying any pesticide. Spray early morning (6-9 AM) or evening (4-7 PM). Keep children and animals away from the sprayed area for at least 4 hours.',
        'cost_estimate': 'Mancozeb 75WP (500g) ≈ ₹120. Estimated cost for 2-3 precautionary sprays on 1 acre ≈ ₹200-350 total. Consult local agro-dealer for current market prices.'
    },
}

# Apply DB updates — inject safety + cost_estimate into each entry
for key, vals in db_updates.items():
    # Find the entry and add fields before the closing }
    # Pattern: find the disease key's treatment entry closing
    old_closing = f'        "fertilizer": '
    
    # Use a different approach: find each key block and inject
    search = f'    {key}: {{\n'
    if search in content:
        # Find the block, inject safety and cost_estimate before the closing brace
        idx = content.find(search)
        block_end = content.find('\n    },', idx) + len('\n    },')
        block = content[idx:block_end]
        
        # Check if already has safety
        if '"safety"' not in block:
            # Inject before the closing },
            new_block = block.replace(
                '\n    },',
                f',\n        "safety": "{vals["safety"]}",\n        "cost_estimate": "{vals["cost_estimate"]}"\n    }},',
                1
            )
            content = content[:idx] + new_block + content[block_end:]
            print(f"  Updated {key}")
        else:
            print(f"  {key} already has safety field")
    else:
        print(f"  WARNING: {key} not found in content")

# ─────────────────────────────────────────────────────────────────
# 2. Update Gemini prompt to request safety + cost_estimate
# ─────────────────────────────────────────────────────────────────
old_gemini_json = '{{\"disease\": \"...\", \"confidence\": 0.95, \"treatment\": \"Step 1: ... Step 2: ... Step 3: ...\", \"fertilizer\": \"...\", \"reason\": \"1-2 sentences describing the exact visual symptoms observed that led to this diagnosis.\"}}'
new_gemini_json = '{{\"disease\": \"...\", \"confidence\": 0.95, \"treatment\": \"Step 1: ... Step 2: ...\", \"fertilizer\": \"...\", \"safety\": \"Wear gloves and avoid spraying during strong sunlight or wind. Spray early morning (6-9 AM) or evening.\", \"cost_estimate\": \"Chemical name (pack size) ≈ ₹XX. Per 20L tank ≈ ₹XX/spray. For N sprays on 1 acre ≈ ₹XXX total.\", \"reason\": \"...\"}}'

if old_gemini_json in content:
    content = content.replace(old_gemini_json, new_gemini_json, 1)
    print("Gemini JSON schema updated with safety + cost_estimate")
else:
    print("WARNING: Gemini JSON schema not found exactly — check manually")

# ─────────────────────────────────────────────────────────────────
# 3. Update Groq prompt to request safety + cost_estimate
# ─────────────────────────────────────────────────────────────────
old_groq_json = '{{\"disease\": \"...\", \"confidence\": 0.9, \"treatment\": \"Step 1: ... Step 2: ...\", \"fertilizer\": \"...\", \"reason\": \"...\"}}'
new_groq_json = '{{\"disease\": \"...\", \"confidence\": 0.9, \"treatment\": \"Step 1: ... Step 2: ...\", \"fertilizer\": \"...\", \"safety\": \"Wear gloves. Spray early morning/evening. Avoid sunlight and wind.\", \"cost_estimate\": \"Chemical (pack size) ≈ ₹XX. Per spray ≈ ₹XX. Full treatment for 1 acre ≈ ₹XXX.\", \"reason\": \"...\"}}'

if old_groq_json in content:
    content = content.replace(old_groq_json, new_groq_json, 1)
    print("Groq JSON schema updated with safety + cost_estimate")
else:
    print("WARNING: Groq JSON schema not found exactly — check manually")

# ─────────────────────────────────────────────────────────────────
# 4. Update get_treatment_from_db to return safety + cost_estimate
# ─────────────────────────────────────────────────────────────────
# Already inheriting from DISEASE_DB dict entries — no change needed

with open('disease_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nAll changes written. Running syntax check...")
